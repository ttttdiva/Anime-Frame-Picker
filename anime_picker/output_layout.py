"""Shared, human-facing output paths for extracted native PNG frames.

Internal run manifests stay in a caller-owned state directory.  This module
only allocates a stable per-source public folder and publishes verified PNGs
directly into its ``images`` child.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import threading
from typing import Any, Iterator, Mapping, Sequence


_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?$", re.IGNORECASE)
_LOCK = threading.RLock()


def _stable_source_identity(value: str) -> str:
    """Prefer the content digest and size over volatile mtime fields."""

    raw = str(value)
    digests = re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", raw, re.IGNORECASE)
    size = re.search(r"(?:^|:)size:(\d+)(?=:|$)", raw, re.IGNORECASE)
    if not size:
        leading = re.match(r"^(\d+):", raw)
        size_value = leading.group(1) if leading else ""
    else:
        size_value = size.group(1)
    if digests:
        return f"sha256:{digests[-1].lower()}:size:{size_value}"
    return raw


def safe_video_name(value: str, *, fallback: str = "video", limit: int = 96) -> str:
    """Return a readable Windows-safe directory component."""

    cleaned = _INVALID.sub("_", str(value or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = fallback
    if _RESERVED.fullmatch(cleaned):
        cleaned = f"_{cleaned}"
    return cleaned[:limit].rstrip(" .") or fallback


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _registry_lock(path: Path) -> Iterator[None]:
    """Lock one registry across threads and processes without a dependency."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _LOCK, lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _registry_path(output_root: Path, state_root: Path) -> Path:
    root_key = hashlib.sha256(os.fspath(output_root.resolve()).casefold().encode("utf-8")).hexdigest()[:16]
    return state_root / "output-layout" / root_key / "sources.json"


def _layout_root(output_root: Path, state_root: Path) -> Path:
    root_key = hashlib.sha256(os.fspath(output_root.resolve()).casefold().encode("utf-8")).hexdigest()[:16]
    return state_root / "output-layout" / root_key


def _source_identity(display_name: str, source_identity: str) -> str:
    return hashlib.sha256((str(display_name).casefold() + "\0" + _stable_source_identity(source_identity)).encode("utf-8")).hexdigest()


def _is_reparse_point(path: Path) -> bool:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return False
    if path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)()):
        return True
    attributes = getattr(current, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _assert_public_path(output_root: Path, target: Path) -> Path:
    """Reject symlinks/junctions and prove the resolved target stays in root."""

    root = Path(output_root).absolute()
    path = Path(target).absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Public output path escapes its root: {path}") from exc
    current = root
    if current.exists() and _is_reparse_point(current):
        raise ValueError(f"Public output root is a symlink or junction: {current}")
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if _is_reparse_point(current):
                raise ValueError(f"Public output path contains a symlink or junction: {current}")
    resolved_root = root.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Resolved public output escapes its root: {resolved_path}") from exc
    return resolved_path


def default_state_root(output_root: str | os.PathLike[str], project_root: str | os.PathLike[str]) -> Path:
    """Choose a private state location beside the public outputs root."""

    root = Path(output_root).resolve(strict=False)
    project = Path(project_root).resolve(strict=False)
    if root == (project / "outputs").resolve(strict=False):
        return project / ".local" / "anime-picker"
    root_key = hashlib.sha256(os.fspath(root).casefold().encode("utf-8")).hexdigest()[:12]
    return root.parent / ".local" / f"anime-picker-{root_key}"


def publication_state_root(output_root: str | os.PathLike[str]) -> Path:
    """Publication registry/manifest location, a function of the public root only.

    Every writer of one public root (UI or batch, any checkout, any
    ``--state-root``) must share the name registry and active manifests, or
    one writer allocates duplicate folders and leaves another's receipts
    pointing at images it replaced. Private run state stays per caller.
    """

    root = Path(output_root).resolve(strict=False)
    root_key = hashlib.sha256(os.fspath(root).casefold().encode("utf-8")).hexdigest()[:12]
    return root.parent / ".local" / f"anime-picker-{root_key}"


def video_output_directory(
    output_root: str | os.PathLike[str],
    display_name: str,
    source_identity: str,
    *,
    state_root: str | os.PathLike[str],
    create_images: bool = True,
) -> Path:
    """Allocate a stable ``<video name>/images`` path without claiming strangers.

    A readable unsuffixed name is used when available.  If the sanitized name
    is already owned by another source (or an unmanaged legacy directory), a
    deterministic content-identity suffix is used.
    """

    root = Path(output_root).absolute()
    state = Path(state_root).resolve(strict=False)
    _assert_public_path(root, root)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=False)
    registry_path = _registry_path(root, state)
    identity = _source_identity(display_name, source_identity)
    base = safe_video_name(Path(str(display_name)).stem)

    with _registry_lock(registry_path):
        if registry_path.is_file():
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("sources"), dict):
                raise ValueError(f"Invalid output layout registry: {registry_path}")
            sources = payload["sources"]
        else:
            sources = {}

        existing = sources.get(identity)
        legacy_identity = hashlib.sha256((str(display_name).casefold() + "\0" + str(source_identity)).encode("utf-8")).hexdigest()
        if existing is None and legacy_identity != identity and legacy_identity in sources:
            legacy_folder = sources.pop(legacy_identity)
            sources[identity] = legacy_folder
            _atomic_json(registry_path, {"revision": 1, "sources": sources})
            existing = legacy_folder
        if existing is not None:
            if not isinstance(existing, str) or Path(existing).name != existing:
                raise ValueError("Unsafe output layout registry entry")
            folder = root / existing
        else:
            occupied = {str(value).casefold() for value in sources.values() if isinstance(value, str)}
            if base.casefold() not in occupied and not (root / base).exists():
                folder_name = base
            else:
                for width in range(8, 65, 4):
                    candidate = safe_video_name(base[: max(1, 96 - width - 1)] + "_" + identity[:width])
                    if candidate.casefold() not in occupied and not (root / candidate).exists():
                        folder_name = candidate
                        break
                else:
                    raise FileExistsError(f"No safe public output directory available for {display_name!r}")
            sources[identity] = folder_name
            payload = {"revision": 1, "sources": sources}
            _atomic_json(registry_path, payload)
            folder = root / folder_name

        _assert_public_path(root, folder)
        folder.mkdir(parents=True, exist_ok=True)
        _assert_public_path(root, folder)
        images = folder / "images"
        _assert_public_path(root, images)
        if create_images:
            images.mkdir(exist_ok=True)
        return _assert_public_path(root, images)


def publish_png(
    source: str | os.PathLike[str],
    images_directory: str | os.PathLike[str],
    preferred_name: str,
    *,
    state_root: str | os.PathLike[str],
) -> tuple[Path, str, bool]:
    """Hard-link (or safely copy) one verified PNG without replacing anything.

    If a different existing image occupies the preferred PTS filename, a
    stable digest suffix is added.  Returns the public path, SHA-256, and
    whether bytes were copied instead of linked.
    """

    src = Path(source).resolve(strict=True)
    raw_images = Path(images_directory).absolute()
    public_root = raw_images.parent.parent
    images = _assert_public_path(public_root, raw_images)
    if src.suffix.lower() != ".png" or not src.is_file():
        raise ValueError(f"Only existing PNG files may be published: {src}")
    images.mkdir(parents=True, exist_ok=True)
    digest = _digest_file(src)
    stem = Path(preferred_name).stem
    if Path(preferred_name).name != preferred_name or Path(preferred_name).suffix.lower() != ".png":
        raise ValueError(f"Unsafe public image name: {preferred_name!r}")

    copied = False
    state = Path(state_root).resolve(strict=False)
    with _registry_lock(_registry_path(images.parent.parent, state)):
        target = images / preferred_name
        _assert_public_path(public_root, target)
        if target.exists():
            if target.is_file() and _digest_file(target) == digest:
                return target, digest, copied
            target = images / f"{stem}_{digest[:8]}.png"
            width = 8
            while target.exists():
                if target.is_file() and _digest_file(target) == digest:
                    return target, digest, copied
                width += 4
                if width > 64:
                    raise FileExistsError(f"Public image hash collision at {target}")
                target = images / f"{stem}_{digest[:width]}.png"

        try:
            os.link(src, target)
        except OSError:
            # Cross-volume or unsupported hard links: copy to a temp in the
            # destination directory, then link that temp into place so a
            # concurrent writer can never be overwritten.
            scratch_root = _layout_root(public_root, state) / "publishing-temp"
            if os.stat(public_root).st_dev == os.stat(state).st_dev:
                scratch_root.mkdir(parents=True, exist_ok=True)
            else:
                scratch_root = images
            descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".publishing", dir=scratch_root)
            temp_path = Path(temporary)
            try:
                with os.fdopen(descriptor, "wb") as destination, src.open("rb") as original:
                    shutil.copyfileobj(original, destination, length=1024 * 1024)
                    destination.flush()
                    os.fsync(destination.fileno())
                if _digest_file(temp_path) != digest:
                    raise IOError(f"Copied image digest mismatch: {temp_path}")
                if target.exists() or _is_reparse_point(target):
                    raise FileExistsError(f"Public image target appeared during publication: {target}")
                os.rename(temp_path, target)
                copied = True
            finally:
                temp_path.unlink(missing_ok=True)
        if not target.is_file() or _digest_file(target) != digest:
            raise IOError(f"Published image digest mismatch: {target}")
    return target, digest, copied


def _active_manifest_path(output_root: Path, state_root: Path, identity: str) -> Path:
    return _layout_root(output_root, state_root) / "active" / f"{identity}.json"


def _image_rows(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    if _is_reparse_point(directory) or not directory.is_dir():
        raise ValueError(f"Images directory is not a normal directory: {directory}")
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if _is_reparse_point(path) or not path.is_file() or path.suffix.lower() != ".png":
            raise ValueError(f"Unexpected entry in images directory: {path}")
        match = re.fullmatch(r"frame_pts_(-?\d+)(?:_[a-f0-9]{8,64})?\.png", path.name)
        if not match:
            raise ValueError(f"Image filename does not preserve exact PTS: {path.name}")
        with path.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"Invalid PNG signature in images directory: {path}")
        rows.append({"file": path.name, "pts": int(match.group(1)), "sha256": _digest_file(path), "size": path.stat().st_size})
    return rows


def _rows_match(actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> bool:
    actual_map = {str(row.get("file")): (int(row.get("pts", -1)), str(row.get("sha256", ""))) for row in actual}
    expected_map = {str(row.get("file")): (int(row.get("pts", -1)), str(row.get("sha256", ""))) for row in expected}
    return actual_map == expected_map


def _archive_directory(output_root: Path, video_directory: Path, run_id: str) -> Path:
    date = datetime.now().astimezone().strftime("%Y%m%d")
    token = hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()[:10]
    base = output_root / "_archive" / date / video_directory.name / f"{safe_video_name(run_id, limit=40)}_{token}"
    for attempt in range(20):
        candidate = base if attempt == 0 else base.with_name(f"{base.name}_{attempt:02d}")
        if not candidate.exists():
            return candidate / "images"
    raise FileExistsError(f"Could not allocate an archive folder for {video_directory.name}")


def _rewrite_reference_value(value: Any, mappings: Sequence[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_reference_value(item, mappings) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_reference_value(item, mappings) for item in value]
    if isinstance(value, str):
        for prefix, new_prefix in sorted((item for item in mappings if item[0]), key=lambda item: len(item[0]), reverse=True):
            variants = (
                (prefix.replace("/", "\\"), new_prefix.replace("/", "\\")),
                (prefix.replace("\\", "/"), new_prefix.replace("\\", "/")),
            )
            for old, new in variants:
                if value.casefold().startswith(old.casefold()):
                    return new + value[len(old):]
    return value


def _rewrite_receipts(
    receipt_paths: Sequence[str],
    *,
    output_root: Path,
    state_root: Path,
    old_relative: str,
    new_relative: str,
) -> list[str]:
    updated: list[str] = []
    old_absolute = os.fspath(output_root / Path(old_relative))
    new_absolute = os.fspath(output_root / Path(new_relative))
    for raw_path in receipt_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = state_root / path
        if not path.is_file():
            continue
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            rewritten = _rewrite_reference_value(value, ((old_relative, new_relative), (old_absolute, new_absolute)))
            if rewritten != value:
                if not isinstance(rewritten, Mapping):
                    raise ValueError(f"Receipt must be a JSON object: {path}")
                _atomic_json(path, rewritten)
                updated.append(os.fspath(path))
        elif path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8-sig")
            rewritten_text = text
            for old, new in ((old_relative, new_relative), (old_absolute, new_absolute)):
                rewritten_text = rewritten_text.replace(old.replace("\\", "/"), new.replace("\\", "/"))
                rewritten_text = rewritten_text.replace(old.replace("/", "\\"), new.replace("/", "\\"))
            if rewritten_text != text:
                path.write_text(rewritten_text, encoding="utf-8", newline="\n")
                updated.append(os.fspath(path))
    return updated


def _stage_png(source: Path, destination: Path, expected_digest: str) -> bool:
    """Add one digest-verified hard link to an internal pending image set."""

    if destination.exists():
        if _digest_file(destination) == expected_digest:
            return False
        raise FileExistsError(f"Pending image already contains different bytes: {destination}")
    try:
        os.link(source, destination)
        copied = False
    except OSError:
        # Keep fallback-copy scratch outside the exact PNG set so a crash does
        # not leave a non-PTS file that blocks same-token publication resume.
        descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".staging", dir=destination.parent.parent)
        temp_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as target, source.open("rb") as original:
                shutil.copyfileobj(original, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            if _digest_file(temp_path) != expected_digest:
                raise IOError(f"Staged image digest mismatch: {temp_path}")
            if destination.exists() or _is_reparse_point(destination):
                raise FileExistsError(f"Pending image target appeared during staging: {destination}")
            os.rename(temp_path, destination)
            copied = True
        finally:
            temp_path.unlink(missing_ok=True)
    if _digest_file(destination) != expected_digest:
        raise IOError(f"Staged image digest mismatch: {destination}")
    return copied


def _write_active_history(layout_root: Path, identity: str, previous: Mapping[str, Any]) -> None:
    if not previous.get("files"):
        return
    old_id = safe_video_name(str(previous.get("run_id") or "previous"), limit=50)
    history_dir = layout_root / "history" / identity
    history_path = history_dir / f"{old_id}_{hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()[:8]}.json"
    if history_path.exists():
        return
    archived = dict(previous)
    archived["status"] = "archived"
    archived["archived_at"] = datetime.now().astimezone().isoformat()
    _atomic_json(history_path, archived)


def _finish_pending_publication(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    output_root: Path,
    state_root: Path,
    images_directory: Path,
    layout_root: Path,
) -> dict[str, Any]:
    staging_directory = Path(str(manifest.get("staging_directory", "")))
    new_files = manifest.get("files")
    previous_files = manifest.get("previous_files", [])
    if not isinstance(new_files, list) or not new_files:
        raise ValueError("Pending public image set is missing its file manifest")
    archive_relative = manifest.get("previous_archive_directory")
    archive_images = output_root / Path(str(archive_relative)) if isinstance(archive_relative, str) and archive_relative else None

    def rewrite_archived_receipts() -> None:
        if archive_images is None or not archive_images.exists() or not manifest.get("previous_images_existed"):
            return
        archived_rows = _image_rows(archive_images)
        if not _rows_match(archived_rows, previous_files):
            raise ValueError("Archived previous image set does not match its active manifest")
        manifest["previous_archived_directory"] = archive_images.relative_to(output_root).as_posix()
        if not manifest.get("rewritten_previous_receipts"):
            old_directory = str(manifest.get("previous_images_directory", ""))
            new_directory = archive_images.relative_to(output_root).as_posix()
            rewritten = _rewrite_receipts(
                manifest.get("previous_receipt_paths", []),
                output_root=output_root,
                state_root=state_root,
                old_relative=old_directory,
                new_relative=new_directory,
            )
            manifest["rewritten_previous_receipts"] = rewritten
            _atomic_json(manifest_path, manifest)

    images_rows = _image_rows(images_directory) if images_directory.exists() else None
    if images_rows is not None and _rows_match(images_rows, new_files):
        pass  # Directory switch completed before a previous interruption.
    else:
        if not staging_directory.is_dir() or _is_reparse_point(staging_directory):
            raise ValueError(f"Pending image staging directory is unavailable: {staging_directory}")
        staged_rows = _image_rows(staging_directory)
        if not _rows_match(staged_rows, new_files):
            raise ValueError("Pending staged PNG set does not match its manifest")
        if images_rows is not None:
            if not isinstance(previous_files, list) or not _rows_match(images_rows, previous_files):
                raise ValueError("Existing public images no longer match the previous active manifest")
            if archive_images is None:
                raise ValueError("Pending publication has no archive destination for existing images")
            _assert_public_path(output_root, archive_images)
            if archive_images.exists():
                raise FileExistsError(f"Archive destination already exists: {archive_images}")
            archive_images.parent.mkdir(parents=True, exist_ok=True)
            _assert_public_path(output_root, archive_images)
            images_directory.rename(archive_images)
            manifest["previous_images_existed"] = True
            _atomic_json(manifest_path, manifest)
            rewrite_archived_receipts()
        elif archive_images is not None and not archive_images.exists():
            if manifest.get("previous_images_existed") and previous_files:
                raise ValueError("Previous active images disappeared before they were archived")
        staging_directory.rename(images_directory)
        images_rows = _image_rows(images_directory)
        if not _rows_match(images_rows, new_files):
            raise ValueError("New public image set failed post-switch verification")

    rewrite_archived_receipts()

    previous_manifest = manifest.get("previous_manifest")
    if isinstance(previous_manifest, Mapping):
        previous_completed = dict(previous_manifest)
        if manifest.get("previous_archived_directory"):
            previous_completed["archived_from_images_directory"] = previous_manifest.get("images_directory")
            previous_completed["images_directory"] = manifest["previous_archived_directory"]
            previous_completed["status"] = "archived"
            previous_completed["archived_at"] = datetime.now().astimezone().isoformat()
        _write_active_history(layout_root, str(manifest["identity"]), previous_completed)

    completed = dict(manifest)
    completed.update({
        "status": "complete",
        "images_directory": images_directory.relative_to(output_root).as_posix(),
        "completed_at": datetime.now().astimezone().isoformat(),
        "staging_directory": None,
    })
    _atomic_json(manifest_path, completed)
    return completed


def publish_image_set(
    output_root: str | os.PathLike[str],
    display_name: str,
    source_identity: str,
    files: Sequence[Mapping[str, Any]],
    *,
    state_root: str | os.PathLike[str],
    run_id: str,
    receipt_paths: Sequence[str | os.PathLike[str]] = (),
) -> tuple[Path, list[dict[str, Any]], dict[tuple[int, str], str], dict[str, Any]]:
    """Publish a complete image set; one publication per video at a time.

    The cross-process lock covers the whole stage-verify-switch transaction,
    so the UI and the batch runner can never interleave on one images
    directory. Other videos publish concurrently.
    """

    from .exclusion import publish_lock

    root = Path(output_root).resolve(strict=False)
    identity = _source_identity(display_name, source_identity)
    lock = publish_lock(root, identity)
    try:
        return _publish_image_set_locked(
            output_root, display_name, source_identity, files,
            state_root=state_root, run_id=run_id, receipt_paths=receipt_paths,
        )
    finally:
        lock.release()


def _publish_image_set_locked(
    output_root: str | os.PathLike[str],
    display_name: str,
    source_identity: str,
    files: Sequence[Mapping[str, Any]],
    *,
    state_root: str | os.PathLike[str],
    run_id: str,
    receipt_paths: Sequence[str | os.PathLike[str]] = (),
) -> tuple[Path, list[dict[str, Any]], dict[tuple[int, str], str], dict[str, Any]]:
    """Publish a complete flat image set, archiving the previous set as one run.

    All images are staged and verified before the public ``images`` directory
    is switched. Prior successful images move to ``outputs/_archive``; no old
    file is removed or overwritten.
    """

    root = Path(output_root).resolve(strict=False)
    state = Path(state_root).resolve(strict=False)
    images = video_output_directory(root, display_name, source_identity, state_root=state, create_images=False)
    video_directory = images.parent
    identity = _source_identity(display_name, source_identity)
    layout_root = _layout_root(root, state)
    manifest_path = _active_manifest_path(root, state, identity)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    if os.stat(root).st_dev != os.stat(state).st_dev:
        raise ValueError("Public outputs and private state must be on the same volume for safe atomic folder switching")

    ordered: list[dict[str, Any]] = []
    names_by_key: dict[tuple[int, str], str] = {}
    names_used: dict[str, str] = {}
    for raw in files:
        source = Path(str(raw.get("source", ""))).resolve(strict=True)
        pts = int(raw["pts"])
        digest = str(raw.get("sha256", "")) or _digest_file(source)
        if source.suffix.lower() != ".png" or not source.is_file() or _digest_file(source) != digest:
            raise ValueError(f"Input PNG does not match publication digest: {source}")
        key = (pts, digest)
        if key in names_by_key:
            continue
        preferred = f"frame_pts_{pts}.png"
        used_digest = names_used.get(preferred.casefold())
        if used_digest is not None and used_digest != digest:
            stem = Path(preferred).stem
            width = 8
            preferred = f"{stem}_{digest[:width]}.png"
            while preferred.casefold() in names_used and names_used[preferred.casefold()] != digest:
                width += 4
                if width > 64:
                    raise FileExistsError("Unable to disambiguate same-PTS PNGs")
                preferred = f"{stem}_{digest[:width]}.png"
        names_by_key[key] = preferred
        names_used[preferred.casefold()] = digest
        ordered.append({"file": preferred, "pts": pts, "sha256": digest, "size": source.stat().st_size, "source": os.fspath(source)})
    if not ordered:
        raise ValueError("Cannot publish an empty image set")

    content_token = hashlib.sha256(json.dumps({"identity": identity, "files": [{k: item[k] for k in ("file", "pts", "sha256")} for item in ordered]}, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    staging_images = layout_root / "staging" / identity / content_token / "images"
    stage_receipt = staging_images.parent / "transaction.json"
    public_receipts = [os.fspath(Path(value).resolve(strict=False)) for value in receipt_paths]

    with _registry_lock(manifest_path):
        active: dict[str, Any] | None = None
        if manifest_path.is_file():
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"Invalid active output manifest: {manifest_path}")
            active = loaded
            if active.get("status") == "publishing":
                active = _finish_pending_publication(
                    manifest_path, active, output_root=root, state_root=state,
                    images_directory=images, layout_root=layout_root,
                )
        if active and active.get("status") == "complete":
            images_existed = images.exists()
            live_rows = _image_rows(images) if images_existed else []
            active_rows = active.get("files", [])
            if _rows_match(live_rows, active_rows) and _rows_match(live_rows, ordered):
                active["receipt_paths"] = sorted(set([*active.get("receipt_paths", []), *public_receipts]))
                _atomic_json(manifest_path, active)
                public = [
                    {"file": (images.relative_to(root) / Path(item["file"])).as_posix(), "pts": item["pts"], "sha256": item["sha256"], "copied": False}
                    for item in ordered
                ]
                return images, public, names_by_key, active
            if not images_existed:
                if not _rows_match(active_rows, ordered):
                    raise ValueError("The active images directory is missing; restore the last active set before publishing a different set")
                active["integrity_repair"] = "missing_public_directory_rebuilt_from_verified_sources"
                active["observed_files"] = []
            elif not _rows_match(live_rows, active_rows):
                active["integrity_repair"] = "modified_or_incomplete_public_set_archived_before_republish"
                active["observed_files"] = live_rows
            else:
                active.pop("integrity_repair", None)
                active.pop("observed_files", None)

        if not active and images.exists():
            existing = _image_rows(images)
            if existing:
                active = {
                    "revision": 1,
                    "status": "complete",
                    "identity": identity,
                    "source_name": display_name,
                    "source_identity": _stable_source_identity(source_identity),
                    "run_id": "adopted-existing",
                    "publication_id": content_token,
                    "images_directory": images.relative_to(root).as_posix(),
                    "files": existing,
                    "receipt_paths": [],
                    "adopted_existing": True,
                }
                _atomic_json(manifest_path, active)

        staging_images.parent.mkdir(parents=True, exist_ok=True)
        staging_images.mkdir(parents=True, exist_ok=True)
        if _is_reparse_point(staging_images) or _is_reparse_point(staging_images.parent):
            raise ValueError(f"Pending image path is a symlink or junction: {staging_images}")
        for item in ordered:
            _stage_png(Path(item["source"]), staging_images / item["file"], item["sha256"])
        staged_rows = _image_rows(staging_images)
        expected_rows = [{key: item[key] for key in ("file", "pts", "sha256", "size")} for item in ordered]
        if not _rows_match(staged_rows, expected_rows):
            raise ValueError("Staged image set failed verification")
        _atomic_json(stage_receipt, {
            "status": "staged",
            "identity": identity,
            "run_id": run_id,
            "images_directory": os.fspath(staging_images),
            "files": expected_rows,
            "created_at": datetime.now().astimezone().isoformat(),
        })

        previous_files = list(active.get("observed_files", active.get("files", []))) if active else []
        previous_receipts = list(active.get("receipt_paths", [])) if active else []
        previous_archive = _archive_directory(root, video_directory, str(active.get("run_id", "adopted-existing")) if active else "empty-output") if images.exists() else None
        previous_manifest = dict(active) if active else None
        next_manifest: dict[str, Any] = {
            "revision": 1,
            "status": "publishing",
            "identity": identity,
            "source_name": display_name,
            "source_identity": _stable_source_identity(source_identity),
            "run_id": run_id,
            "publication_id": content_token,
            "images_directory": images.relative_to(root).as_posix(),
            "staging_directory": os.fspath(staging_images),
            "files": expected_rows,
            "receipt_paths": public_receipts,
            "previous_images_directory": active.get("images_directory", images.relative_to(root).as_posix()) if active else images.relative_to(root).as_posix(),
            "previous_files": previous_files,
            "previous_receipt_paths": previous_receipts,
            "previous_manifest": previous_manifest,
            "previous_images_existed": images.exists(),
            "previous_archive_directory": previous_archive.relative_to(root).as_posix() if previous_archive else None,
            "started_at": datetime.now().astimezone().isoformat(),
        }
        _atomic_json(manifest_path, next_manifest)
        active = _finish_pending_publication(
            manifest_path, next_manifest, output_root=root, state_root=state,
            images_directory=images, layout_root=layout_root,
        )
    public_files = [
        {"file": (images.relative_to(root) / Path(item["file"])).as_posix(), "pts": item["pts"], "sha256": item["sha256"], "copied": not os.path.samefile(Path(item["source"]), images / item["file"])}
        for item in ordered
    ]
    return images, public_files, names_by_key, active


def register_public_receipts(
    output_root: str | os.PathLike[str],
    display_name: str,
    source_identity: str,
    *,
    state_root: str | os.PathLike[str],
    receipt_paths: Sequence[str | os.PathLike[str]],
    publication_id: str | None = None,
) -> None:
    root = Path(output_root).resolve(strict=False)
    state = Path(state_root).resolve(strict=False)
    identity = _source_identity(display_name, source_identity)
    manifest_path = _active_manifest_path(root, state, identity)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Active public image manifest is missing: {manifest_path}")
    with _registry_lock(manifest_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Cannot register receipts for a malformed public image set")
        if manifest.get("status") == "publishing":
            images = root / Path(str(manifest.get("images_directory", "")))
            manifest = _finish_pending_publication(
                manifest_path, manifest, output_root=root, state_root=state,
                images_directory=images, layout_root=_layout_root(root, state),
            )
        paths = {os.fspath(Path(item).resolve(strict=False)) for item in receipt_paths}
        if publication_id is None or manifest.get("publication_id") == publication_id:
            if manifest.get("status") != "complete":
                raise ValueError("Cannot register receipts for an incomplete public image set")
            manifest["receipt_paths"] = sorted(set([*manifest.get("receipt_paths", []), *paths]))
            _atomic_json(manifest_path, manifest)
            return

        history_root = _layout_root(root, state) / "history" / identity
        history_manifest_path = None
        history_manifest: dict[str, Any] | None = None
        if history_root.is_dir():
            for candidate in history_root.glob("*.json"):
                value = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(value, dict) and value.get("publication_id") == publication_id:
                    history_manifest_path = candidate
                    history_manifest = value
                    break
        if history_manifest is None or history_manifest_path is None or history_manifest.get("status") != "archived":
            raise ValueError(f"Publication {publication_id} is neither active nor in output history")
        old_directory = str(history_manifest.get("archived_from_images_directory", ""))
        archive_directory = str(history_manifest.get("images_directory", ""))
        rewritten = _rewrite_receipts(
            list(paths),
            output_root=root,
            state_root=state,
            old_relative=old_directory,
            new_relative=archive_directory,
        )
        history_manifest["receipt_paths"] = sorted(set([*history_manifest.get("receipt_paths", []), *paths]))
        history_manifest["late_receipt_rewrites"] = sorted(set([*history_manifest.get("late_receipt_rewrites", []), *rewritten]))
        _atomic_json(history_manifest_path, history_manifest)
