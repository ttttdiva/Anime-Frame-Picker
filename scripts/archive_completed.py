#!/usr/bin/env python3
"""Safely move successfully processed source video + matching subtitles to completed/.

Archiving requires a finished batch report AND verified per-video result marker,
native PNGs and original video fingerprint. It never overwrites destinations.
Can run immediately from a final batch report, or wait for a current Home MCP
Gateway managed batch job to print its final report line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Any, Mapping, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SUBTITLE_EXTS = {".vtt", ".srt", ".ass", ".ssa"}
VIDEO_EXTS = {".mp4", ".mkv"}
FINAL_LINE = re.compile(r"(?m)^\[batch\] (?:completed|incomplete); report: ([^\r\n]+)\r?$")
PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _json(path: Path) -> Mapping[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + os.urandom(4).hex())
    try:
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return f"{stat.st_size}:{stat.st_mtime_ns}:{sha.hexdigest()}"


def _within(target: Path, root: Path) -> Path:
    normalized = target.resolve()
    if not normalized.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes expected root: {target}")
    return normalized


def _relative_output_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {label} path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe {label} path: {value!r}")
    return _within(root / relative, root)


def _public_path_no_reparse(root: Path, relative: Path, label: str) -> Path:
    root_abs = root.absolute()
    target = (root_abs / relative).absolute()
    try:
        parts = target.relative_to(root_abs).parts
    except ValueError as exc:
        raise ValueError(f"{label} escapes public output root") from exc
    current = root_abs
    for component in ("", *parts):
        if component:
            current = current / component
        if not (current.exists() or current.is_symlink()):
            continue
        try:
            attributes = current.lstat().st_file_attributes
        except AttributeError:
            attributes = 0
        is_junction = bool(getattr(current, "is_junction", lambda: False)())
        if current.is_symlink() or is_junction or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError(f"{label} contains a symlink or junction: {current}")
    return _within(target, root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_original_sidecar(
    output_root: Path,
    processed_path: Path,
    pts: int,
    output_entry: Mapping[str, Any],
    metadata_export: Mapping[str, Any],
) -> None:
    """Validate the untouched-frame sidecar and keep no-logo runs explicit."""
    original_file = output_entry.get("original_file")
    output_digest = output_entry.get("original_sha256")
    metadata_relative = metadata_export.get("original_relative_path")
    metadata_digest = metadata_export.get("original_sha256")
    if original_file is None:
        if output_digest is not None or metadata_relative is not None or metadata_digest is not None:
            raise ValueError(f"Original sidecar fields disagree for {processed_path}")
        return

    name = f"frame_pts_{pts}.png"
    expected = _within(processed_path.parent / "originals" / name, output_root)
    original_path = _relative_output_path(output_root, original_file, "original-frame sidecar")
    if original_path != expected or metadata_relative != f"originals/{name}":
        raise ValueError(f"Original sidecar path mismatch for {processed_path}")
    if not isinstance(output_digest, str) or not output_digest or metadata_digest != output_digest:
        raise ValueError(f"Original sidecar digest metadata mismatch for {processed_path}")
    if not original_path.is_file() or original_path.stat().st_size <= len(PNG_HEADER):
        raise ValueError(f"Missing/empty original-frame PNG: {original_path}")
    with original_path.open("rb") as handle:
        if handle.read(8) != PNG_HEADER:
            raise ValueError(f"Invalid original-frame PNG signature: {original_path}")
    try:
        with Image.open(original_path) as image:
            if image.format != "PNG":
                raise ValueError(f"Original-frame file is not PNG: {original_path}")
            image.load()
    except Exception as exc:
        raise ValueError(f"Undecodable original-frame PNG: {original_path}") from exc
    if _sha256(original_path) != output_digest:
        raise ValueError(f"Original-frame PNG digest mismatch: {original_path}")


def _all_mode_files_valid(output: Path, marker: Mapping[str, Any], saved: list[Path]) -> None:
    """Verify the flat all-mode frame manifest and its exact-PTS metadata."""
    outputs = marker.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(saved) or not outputs:
        raise ValueError("All-mode output manifest count mismatch")

    output_entries: dict[Path, Mapping[str, Any]] = {}
    metadata_paths: set[Path] = set()
    for entry in outputs:
        if not isinstance(entry, Mapping):
            raise ValueError("Malformed all-mode output manifest entry")
        path = _relative_output_path(output, entry.get("file"), "all-mode output")
        metadata_path = _relative_output_path(output, entry.get("metadata"), "all-mode metadata")
        if path in output_entries:
            raise ValueError("Duplicate all-mode output manifest entry")
        if path not in saved:
            raise ValueError("All-mode output manifest does not match saved PNGs")
        if path.suffix.lower() != ".png":
            raise ValueError(f"All-mode output is not a PNG: {path}")
        output_entries[path] = entry
        metadata_paths.add(metadata_path)

    if set(output_entries) != set(saved):
        raise ValueError("Saved PNGs do not match the all-mode output manifest")
    if len(metadata_paths) != 1:
        raise ValueError("All-mode outputs do not share one frame metadata manifest")
    metadata_path = next(iter(metadata_paths))
    if metadata_path != _within(output / "frames" / "metadata.json", output):
        raise ValueError("All-mode metadata path must be frames/metadata.json")
    metadata = _json(metadata_path)
    if metadata.get("source_fingerprint") != marker.get("fingerprint"):
        raise ValueError("All-mode metadata fingerprint mismatch")
    exports = metadata.get("exports")
    if not isinstance(exports, list) or not exports or len(exports) != len(saved):
        raise ValueError("All-mode metadata export count mismatch")

    export_entries: dict[Path, Mapping[str, Any]] = {}
    for entry in exports:
        if not isinstance(entry, Mapping):
            raise ValueError("Malformed all-mode metadata export")
        path = _relative_output_path(metadata_path.parent, entry.get("file"), "all-mode metadata export")
        if path in export_entries:
            raise ValueError("Duplicate all-mode metadata export")
        if path not in saved:
            raise ValueError("All-mode metadata does not match saved PNGs")
        export_entries[path] = entry

    if set(export_entries) != set(saved):
        raise ValueError("Saved PNGs do not match all-mode metadata exports")

    for path in saved:
        output_entry = output_entries[path]
        export = export_entries[path]
        try:
            output_pts = int(output_entry["pts"])
            metadata_pts = int(export["pts"])
            source_pts = int(export["source_pts"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid exact-PTS provenance for {path}") from exc
        if output_pts != metadata_pts or metadata_pts != source_pts:
            raise ValueError(f"All-mode PTS provenance mismatch: {path}")
        output_hash = output_entry.get("sha256")
        metadata_hash = export.get("sha256")
        actual_hash = _sha256(path)
        if output_hash != actual_hash or metadata_hash != actual_hash:
            raise ValueError(f"All-mode PNG digest mismatch: {path}")
        _validate_original_sidecar(output, path, output_pts, output_entry, export)


def _versioned_character_files_valid(
    output: Path,
    marker: Mapping[str, Any],
    characters: list[Any],
    saved: list[Path],
) -> None:
    """Validate revisioned character PTS and digest provenance end to end."""
    revision = marker.get("output_manifest_revision")
    exports_by_path: dict[Path, dict[str, Any]] = {}
    for character in characters:
        if not isinstance(character, Mapping) or character.get("status") != "completed":
            raise ValueError("At least one character was not successfully exported")
        index = int(character["index"])
        folder = _within(output / f"character_{index:02d}", output)
        metadata_path = _within(folder / "metadata.json", output)
        metadata = _json(metadata_path)
        if metadata.get("source_fingerprint") != marker.get("fingerprint"):
            raise ValueError(f"Character metadata fingerprint mismatch: {folder}")
        if metadata.get("output_manifest_revision") != revision:
            raise ValueError(f"Character metadata manifest revision mismatch: {folder}")
        exports = metadata.get("exports")
        if not isinstance(exports, list) or not exports or len(exports) != int(character.get("exported", -1)):
            raise ValueError(f"Character metadata count mismatch: {folder}")
        for entry in exports:
            if not isinstance(entry, Mapping):
                raise ValueError("Malformed export metadata")
            try:
                pts = int(entry["pts"])
                source_pts = int(entry["source_pts"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Invalid exact-PTS character provenance") from exc
            expected_name = f"frame_pts_{pts}.png"
            if entry.get("file") != expected_name or source_pts != pts:
                raise ValueError("Character metadata PTS or filename provenance mismatch")
            path = _relative_output_path(folder, entry["file"], "character export")
            if path in exports_by_path:
                raise ValueError("Duplicate character metadata export")
            digest = entry.get("sha256")
            if not isinstance(digest, str) or digest != _sha256(path):
                raise ValueError(f"Character PNG digest mismatch: {path}")
            exports_by_path[path] = {
                "metadata": metadata_path,
                "pts": pts,
                "sha256": digest,
                "entry": entry,
            }

    if set(exports_by_path) != set(saved):
        raise ValueError("Saved PNGs do not match the per-character metadata")

    outputs = marker.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(saved) or not outputs:
        raise ValueError("Character output manifest count mismatch")
    output_paths: set[Path] = set()
    for entry in outputs:
        if not isinstance(entry, Mapping):
            raise ValueError("Malformed character output manifest entry")
        path = _relative_output_path(output, entry.get("file"), "character output")
        metadata_path = _relative_output_path(output, entry.get("metadata"), "character metadata")
        if path in output_paths or path not in exports_by_path:
            raise ValueError("Character output manifest does not match saved PNGs")
        export = exports_by_path[path]
        try:
            output_pts = int(entry["pts"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid character output PTS: {path}") from exc
        if metadata_path != export["metadata"] or output_pts != export["pts"] or entry.get("sha256") != export["sha256"]:
            raise ValueError(f"Character output provenance mismatch: {path}")
        _validate_original_sidecar(output, path, output_pts, entry, export["entry"])
        output_paths.add(path)
    if output_paths != set(saved):
        raise ValueError("Saved PNGs do not match the character output manifest")


def _saved_files_valid(output: Path, marker: Mapping[str, Any]) -> list[Path]:
    """Verify the metadata and *native exported frames* before moving input."""
    if marker.get("status") != "completed":
        raise ValueError("per-video result marker is not completed")
    raw = marker.get("saved_files")
    if not isinstance(raw, list) or not raw or not all(isinstance(x, str) for x in raw):
        raise ValueError("per-video result marker has no saved PNGs")
    saved: list[Path] = []
    for relative in raw:
        path = _relative_output_path(output, relative, "saved output")
        if path.suffix.lower() != ".png" or not path.is_file() or path.stat().st_size <= len(PNG_HEADER):
            raise ValueError(f"Missing/empty native PNG: {path}")
        with path.open("rb") as file:
            if file.read(8) != PNG_HEADER:
                raise ValueError(f"Invalid PNG signature: {path}")
        saved.append(path)
    if len(set(saved)) != len(saved):
        raise ValueError("Duplicate exported PNG entry")
    if marker.get("mode") == "all":
        _all_mode_files_valid(output, marker, saved)
        return saved

    characters = marker.get("characters")
    if not isinstance(characters, list) or not characters:
        raise ValueError("Missing per-character summaries")
    if marker.get("output_manifest_revision") is not None:
        _versioned_character_files_valid(output, marker, characters, saved)
        return saved

    metadata_paths: set[Path] = set()
    for character in characters:
        if not isinstance(character, Mapping) or character.get("status") != "completed":
            raise ValueError("At least one character was not successfully exported")
        index = int(character["index"])
        folder = _within(output / f"character_{index:02d}", output)
        meta = _json(folder / "metadata.json")
        if meta.get("source_fingerprint") != marker.get("fingerprint"):
            raise ValueError(f"Character metadata fingerprint mismatch: {folder}")
        exports = meta.get("exports")
        if not isinstance(exports, list) or not exports or len(exports) != int(character.get("exported", -1)):
            raise ValueError(f"Character metadata count mismatch: {folder}")
        for entry in exports:
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                raise ValueError("Malformed export metadata")
            metadata_paths.add(_within(folder / entry["file"], output))
    if set(saved) != metadata_paths:
        raise ValueError("Saved PNGs do not match the per-character metadata")
    return saved


def _public_files_valid(public_root: Path, marker: Mapping[str, Any]) -> int | None:
    """Verify the user-facing flat image set when the receipt has one."""

    directory_value = marker.get("images_directory")
    public_files = marker.get("public_files")
    if directory_value is None and public_files is None:
        return None  # Backward compatibility for historical receipts.
    if not isinstance(directory_value, str) or not isinstance(public_files, list) or not public_files:
        raise ValueError("Public image manifest is incomplete")
    directory_relative = Path(directory_value)
    parts = directory_relative.parts
    active_layout = len(parts) == 2 and parts[-1] == "images" and parts[0].casefold() != "_archive"
    archived_layout = (
        len(parts) == 5
        and parts[0].casefold() == "_archive"
        and bool(re.fullmatch(r"\d{8}", parts[1]))
        and parts[-1] == "images"
    )
    if directory_relative.is_absolute() or ".." in parts or not (active_layout or archived_layout):
        raise ValueError("Public images directory must be <video>/images or a recorded _archive/<date>/<video>/<run>/images")
    images_dir = _public_path_no_reparse(public_root, directory_relative, "Public images directory")
    if not images_dir.is_dir():
        raise ValueError(f"Public images directory is missing: {images_dir}")
    expected_entries: dict[str, tuple[int, str]] = {}
    outputs = marker.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Internal output manifest is missing")
    for output in outputs:
        if not isinstance(output, Mapping):
            raise ValueError("Malformed internal output entry")
        public_file = output.get("public_file")
        public_digest = output.get("public_sha256")
        if not isinstance(public_file, str) or not isinstance(public_digest, str):
            raise ValueError("Internal output entry has no public image reference")
        expected_entries[public_file] = (int(output["pts"]), public_digest)
    listed: dict[str, tuple[int, str]] = {}
    for entry in public_files:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("file"), str):
            raise ValueError("Malformed public image entry")
        name = str(entry["file"])
        if name in listed:
            raise ValueError(f"Duplicate public image entry: {name}")
        if Path(name).parent.as_posix() != directory_relative.as_posix():
            raise ValueError(f"Public image is outside the video images directory: {name}")
        path = _public_path_no_reparse(public_root, Path(name), "Public output")
        if not path.is_file() or path.suffix.lower() != ".png" or path.stat().st_size <= len(PNG_HEADER):
            raise ValueError(f"Missing/empty public PNG: {path}")
        with path.open("rb") as handle:
            if handle.read(8) != PNG_HEADER:
                raise ValueError(f"Invalid public PNG signature: {path}")
        pts = int(entry["pts"])
        digest = str(entry.get("sha256", ""))
        if not re.fullmatch(rf"frame_pts_{pts}(?:_[a-f0-9]{{8,64}})?\.png", path.name):
            raise ValueError(f"Public filename does not preserve exact PTS: {path.name}")
        if digest != _sha256(path):
            raise ValueError(f"Public PNG digest mismatch: {path}")
        listed[name] = (pts, digest)
    if any(name not in listed or listed[name] != expected for name, expected in expected_entries.items()):
        raise ValueError("Public image manifest does not match internal exact-PTS output entries")
    actual_names = {path.name for path in images_dir.iterdir()}
    listed_names = {Path(name).name for name in listed}
    if actual_names != listed_names:
        raise ValueError("Public images directory contains files outside the active run manifest")
    return len(listed)


def _matching_subtitles(source: Path, video: Path) -> list[Path]:
    wanted_stem = video.stem.casefold()
    return sorted(
        (
            path for path in source.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUBTITLE_EXTS
            and path.stem.casefold() == wanted_stem
        ),
        key=lambda p: p.name.casefold(),
    )


def archive_report(
    report: Mapping[str, Any],
    *,
    source_dir: Path,
    completed_dir: Path,
    output_root: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Archive only videos proven complete, without overwriting anything.

    The report can be 'incomplete' overall; completed video rows may still be
    archived, but failed/incomplete rows remain in source for safe retry.
    """
    source = source_dir.resolve()
    completed = completed_dir.resolve()
    outputs = output_root.resolve()
    state_root_value = report.get("state_root")
    state_root = Path(str(state_root_value)).resolve() if isinstance(state_root_value, str) else outputs
    if not state_root.is_absolute():
        raise ValueError("Batch report state root must be absolute")
    reported_output_root = report.get("output_root")
    if isinstance(reported_output_root, str) and Path(reported_output_root).resolve() != outputs:
        raise ValueError("Batch report public output root does not match the expected output directory")
    if source == completed or completed.is_relative_to(source):
        raise ValueError("Completed folder must not be the source or nested inside it")
    if Path(str(report.get("source", ""))).resolve() != source:
        raise ValueError("Batch report source does not match the expected source directory")
    jobs = report.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Batch report has no job entries")
    results: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, Mapping):
            results.append({"status": "error", "error": "Invalid batch job entry"})
            continue
        detail: dict[str, Any] = {"video": job.get("video")}
        results.append(detail)
        if job.get("status") not in {"completed", "skipped"}:
            detail["status"] = "left_in_source"
            detail["reason"] = f"Processing status: {job.get('status')}"
            continue
        try:
            video = Path(str(job.get("video", ""))).resolve()
            if video.parent != source or video.suffix.lower() not in VIDEO_EXTS:
                raise ValueError("Video must be a direct source MP4/MKV")
            output = _within(Path(str(job.get("output", ""))), state_root)
            if output == state_root:
                raise ValueError("Output must be a per-video folder")
            marker = _json(output / "batch-result.json")
            marker_video = Path(str(marker.get("video", ""))).resolve()
            if marker_video != video:
                raise ValueError("Per-video marker points to a different source")
            if job.get("status") == "completed":
                if job.get("fingerprint") != marker.get("fingerprint"):
                    raise ValueError("Batch report and result marker fingerprints differ")
            frames = _saved_files_valid(output, marker)
            public_count = _public_files_valid(outputs, marker)
            expected = str(marker.get("fingerprint", ""))
            destination = completed / video.name
            if not video.is_file():
                if destination.is_file() and _fingerprint(destination) == expected:
                    associates = _matching_subtitles(source, video)
                    pending: list[Path] = []
                    collisions: list[Path] = []
                    for subtitle in associates:
                        if (completed / subtitle.name).exists():
                            collisions.append(subtitle)
                        else:
                            pending.append(subtitle)
                    if dry_run:
                        detail.update({
                            "status": "already_archived",
                            "moved": [],
                            "would_move": [p.name for p in pending],
                            "collisions": [p.name for p in collisions],
                            "pngs_verified": len(frames),
                            "public_pngs_verified": public_count,
                        })
                        continue

                    moved_subtitles: list[Path] = []
                    try:
                        for subtitle in pending:
                            target = completed / subtitle.name
                            if target.exists():
                                collisions.append(subtitle)
                                continue
                            try:
                                subtitle.rename(target)
                            except FileExistsError:
                                collisions.append(subtitle)
                                continue
                            moved_subtitles.append(subtitle)
                    except OSError as exc:
                        rollback_errors: list[str] = []
                        for subtitle in reversed(moved_subtitles):
                            try:
                                (completed / subtitle.name).rename(subtitle)
                            except OSError as rollback_error:
                                rollback_errors.append(f"{subtitle.name}: {rollback_error}")
                        raise OSError(f"Subtitle reconciliation failed: {exc}; rollback issues: {rollback_errors}") from exc
                    detail.update({
                        "status": "already_archived",
                        "moved": [p.name for p in moved_subtitles],
                        "collisions": [p.name for p in collisions],
                        "pngs_verified": len(frames),
                    })
                    if moved_subtitles:
                        print(f"[archive] {video.name}: reconciled {len(moved_subtitles)} subtitle(s)", flush=True)
                else:
                    raise ValueError("Source missing and no matching archived video was found")
                continue
            if _fingerprint(video) != expected:
                raise ValueError("Source video was modified after analysis")
            associates = _matching_subtitles(source, video)
            group = [video, *associates]
            existing_destinations = [file for file in group if (completed / file.name).exists()]
            if existing_destinations:
                detail["collisions"] = [file.name for file in existing_destinations]
                raise FileExistsError(
                    "Destination already exists; original left untouched: "
                    + ", ".join(str(completed / file.name) for file in existing_destinations)
                )
            if dry_run:
                detail.update({"status": "would_archive", "moved": [p.name for p in group], "pngs_verified": len(frames), "public_pngs_verified": public_count})
                continue
            completed.mkdir(parents=True, exist_ok=True)
            moved: list[Path] = []
            try:
                for file in group:
                    file.rename(completed / file.name)
                    moved.append(file)
            except OSError as exc:
                rollback_errors: list[str] = []
                for file in reversed(moved):
                    try:
                        (completed / file.name).rename(file)
                    except OSError as rollback_error:
                        rollback_errors.append(f"{file.name}: {rollback_error}")
                raise OSError(f"Archive move failed: {exc}; rollback issues: {rollback_errors}") from exc
            detail.update({"status": "archived", "moved": [p.name for p in moved], "pngs_verified": len(frames), "public_pngs_verified": public_count})
            print(f"[archive] {video.name}: video + {len(associates)} subtitles, {len(frames)} PNGs verified", flush=True)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            detail.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[archive:error] {job.get('video')}: {detail['error']}", file=sys.stderr, flush=True)
    summary = {
        "batch_id": report.get("batch_id"),
        "source": str(source),
        "completed": str(completed),
        "dry_run": dry_run,
        "videos_archived": sum(x["status"] == "archived" for x in results),
        "videos_already_archived": sum(x["status"] == "already_archived" for x in results),
        "videos_left_in_source": sum(x["status"] == "left_in_source" for x in results),
        "errors": sum(x["status"] == "error" for x in results),
        "files_moved": sum(len(x.get("moved", [])) for x in results if x["status"] in {"archived", "already_archived"}),
        "collisions": sum(len(x.get("collisions", [])) for x in results),
        "jobs": results,
    }
    return summary


def archive_and_record(report_path: Path, *, dry_run: bool = False) -> tuple[int, dict[str, Any]]:
    report_path = report_path.resolve(strict=True)
    report = _json(report_path)
    source = Path(str(report["source"]))
    output_root = Path(str(report.get("output_root", report_path.parent))).resolve()
    state_root = Path(str(report.get("state_root", report_path.parent))).resolve()
    if report.get("state_root") is None:
        report = {**report, "state_root": str(state_root)}
    result = archive_report(
        report,
        source_dir=source,
        completed_dir=source.parent / "completed",
        output_root=output_root,
        dry_run=dry_run,
    )
    if not dry_run:
        receipt = state_root / f"archive_batch_{report.get('batch_id', 'unknown')}.json"
        _write_json(receipt, result)
        print(f"[archive] receipt: {receipt}", flush=True)
    print(
        f"[archive] moved videos={result['videos_archived']}, moved files={result['files_moved']}, "
        f"already archived={result['videos_already_archived']}, unfinished={result['videos_left_in_source']}, "
        f"errors={result['errors']}",
        flush=True,
    )
    return (1 if result["errors"] else 0), result


def watch_managed_job(job_dir: Path, *, poll_seconds: float, timeout_hours: float) -> int:
    job_dir = job_dir.resolve(strict=True)
    metadata_path = job_dir / "metadata.json"
    metadata = _json(metadata_path)
    started_at = float(metadata["started_at"])
    stdout = job_dir / "stdout.log"
    print(f"[archive:watch] awaiting the final report of job {metadata['job_id']}", flush=True)
    deadline = time.monotonic() + timeout_hours * 3600
    while time.monotonic() <= deadline:
        if stdout.exists():
            # Bounded tail: the final report line must be near the end.
            with stdout.open("rb") as stream:
                stream.seek(max(0, stdout.stat().st_size - 16384))
                content = stream.read().decode("utf-8", errors="replace")
            matches = list(FINAL_LINE.finditer(content))
            if matches:
                report_path = Path(matches[-1].group(1).strip())
                if report_path.is_file() and report_path.stat().st_mtime >= started_at - 1:
                    code, _ = archive_and_record(report_path)
                    return code
        metadata = _json(metadata_path)
        if metadata.get("finished_at") is not None or str(metadata.get("status", "")).lower() in {"completed", "failed", "cancelled", "error", "stopped"}:
            print(f"[archive:watch:error] Job terminated without a verifiable final report; no files moved: {metadata.get('status')}", file=sys.stderr, flush=True)
            return 2
        time.sleep(max(1, poll_seconds))
    print("[archive:watch:error] Timed out without a final batch report; no files moved", file=sys.stderr, flush=True)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Archive finished batch input videos and paired subtitles safely")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--report", type=Path, help="Existing batch_run_*.json with per-video completion evidence")
    choice.add_argument("--watch-job-dir", type=Path, help="Home MCP Gateway .runtime/jobs/<id> folder")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--timeout-hours", type=float, default=72)
    parser.add_argument("--dry-run", action="store_true", help="Verify and display intended moves without moving")
    args = parser.parse_args(argv)
    if args.timeout_hours <= 0 or args.poll_seconds <= 0:
        parser.error("--poll-seconds and --timeout-hours must be positive")
    if args.dry_run and args.watch_job_dir is not None:
        parser.error("--dry-run requires --report")
    try:
        if args.watch_job_dir:
            return watch_managed_job(args.watch_job_dir, poll_seconds=args.poll_seconds, timeout_hours=args.timeout_hours)
        status, _ = archive_and_record(args.report, dry_run=args.dry_run)
        return status
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[archive:error] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
