"""No-GPU tests for safe, post-batch input archiving and live-job watcher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import pytest
from PIL import Image

from scripts.archive_completed import _fingerprint, _public_files_valid, archive_report, watch_managed_job


def test_public_manifest_rejects_extra_images_from_superseded_runs(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    images = output_root / "Episode" / "images"
    images.mkdir(parents=True)
    expected = images / "frame_pts_100.png"
    extra = images / "frame_pts_200.png"
    Image.new("RGB", (8, 6), (1, 2, 3)).save(expected, format="PNG")
    digest = hashlib.sha256(expected.read_bytes()).hexdigest()
    marker = {
        "images_directory": "Episode/images",
        "public_files": [{"file": "Episode/images/frame_pts_100.png", "pts": 100, "sha256": digest}],
        "outputs": [{"file": "frames/frame_pts_100.png", "public_file": "Episode/images/frame_pts_100.png", "public_sha256": digest, "pts": 100}],
    }
    assert _public_files_valid(output_root, marker) == 1
    Image.new("RGB", (8, 6), (3, 2, 1)).save(extra, format="PNG")
    with pytest.raises(ValueError, match="outside the active run manifest"):
        _public_files_valid(output_root, marker)


def test_public_manifest_rejects_symlinked_images_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    actual = output_root / "Episode" / "images"
    alias = output_root / "Alias" / "images"
    actual.mkdir(parents=True)
    alias.parent.mkdir(parents=True)
    frame = actual / "frame_pts_100.png"
    Image.new("RGB", (8, 6), (1, 2, 3)).save(frame, format="PNG")
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink creation is unavailable: {error}")
    digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    marker = {
        "images_directory": "Alias/images",
        "public_files": [{"file": "Alias/images/frame_pts_100.png", "pts": 100, "sha256": digest}],
        "outputs": [{"file": "frames/frame_pts_100.png", "public_file": "Alias/images/frame_pts_100.png", "public_sha256": digest, "pts": 100}],
    }
    with pytest.raises(ValueError, match="symlink or junction"):
        _public_files_valid(output_root, marker)


def test_public_manifest_rejects_run_id_or_version_layers_in_active_path(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    images = output_root / "Episode" / "v5" / "images"
    images.mkdir(parents=True)
    frame = images / "frame_pts_100.png"
    Image.new("RGB", (8, 6), (1, 2, 3)).save(frame, format="PNG")
    digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    marker = {
        "images_directory": "Episode/v5/images",
        "public_files": [{"file": "Episode/v5/images/frame_pts_100.png", "pts": 100, "sha256": digest}],
        "outputs": [{"file": "frames/frame_pts_100.png", "public_file": "Episode/v5/images/frame_pts_100.png", "public_sha256": digest, "pts": 100}],
    }
    with pytest.raises(ValueError, match="<video>/images"):
        _public_files_valid(output_root, marker)


def test_public_manifest_accepts_the_recorded_archive_layout(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    relative = Path("_archive/20261020/Episode/run-123/images")
    images = output_root / relative
    images.mkdir(parents=True)
    frame = images / "frame_pts_100.png"
    Image.new("RGB", (8, 6), (1, 2, 3)).save(frame, format="PNG")
    digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    filename = (relative / frame.name).as_posix()
    marker = {
        "images_directory": relative.as_posix(),
        "public_files": [{"file": filename, "pts": 100, "sha256": digest}],
        "outputs": [{"file": "frames/frame_pts_100.png", "public_file": filename, "public_sha256": digest, "pts": 100}],
    }
    assert _public_files_valid(output_root, marker) == 1


def fixture(tmp_path: Path, *, status: str = "completed"):
    source = tmp_path / "source"
    output_root = tmp_path / "outputs" / "batch"
    complete = tmp_path / "completed"
    source.mkdir()
    output_root.mkdir(parents=True)
    video = source / "A special episode 1.mp4"
    video.write_bytes(b"mock video content")
    subtitle = source / "A special episode 1.vtt"
    subtitle.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n")
    fingerprint = _fingerprint(video)
    folder = output_root / "results"
    ch = folder / "character_01"
    ch.mkdir(parents=True)
    Image.new("RGB", (48, 29), (3, 4, 5)).save(ch / "frame_0001_pts_100.png", format="PNG")
    (ch / "metadata.json").write_text(
        json.dumps(
            {
                "source_fingerprint": fingerprint,
                "exports": [{"file": "frame_0001_pts_100.png", "pts": 100}],
            }
        ),
        encoding="utf-8",
    )
    marker = {
        "video": str(video),
        "status": status,
        "fingerprint": fingerprint,
        "saved_files": ["character_01/frame_0001_pts_100.png"],
        "characters": [{"index": 1, "status": status, "exported": 1}],
    }
    (folder / "batch-result.json").write_text(json.dumps(marker), encoding="utf-8")
    report = {
        "batch_id": "test-batch",
        "status": status,
        "source": str(source),
        "jobs": [{"video": str(video), "output": str(folder), "status": status, "fingerprint": fingerprint}],
    }
    return source, output_root, complete, video, subtitle, folder, report


def all_mode_fixture(tmp_path: Path, *, with_original: bool = False):
    source, output_root, complete, video, subtitle, folder, report = fixture(tmp_path)
    frame_dir = folder / "frames"
    frame_dir.mkdir()
    frame = frame_dir / "frame_pts_100.png"
    Image.new("RGB", (48, 29), (6, 7, 8)).save(frame, format="PNG")
    digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    original_file = None
    original_relative = None
    original_digest = None
    if with_original:
        original_dir = frame_dir / "originals"
        original_dir.mkdir()
        original = original_dir / frame.name
        Image.new("RGB", (48, 29), (12, 13, 14)).save(original, format="PNG")
        original_file = "frames/originals/" + frame.name
        original_relative = "originals/" + frame.name
        original_digest = hashlib.sha256(original.read_bytes()).hexdigest()
    (frame_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_fingerprint": _fingerprint(video),
                "output_manifest_revision": 1,
                "exports": [{
                    "file": frame.name,
                    "pts": 100,
                    "source_pts": 100,
                    "sha256": digest,
                    "original_relative_path": original_relative,
                    "original_sha256": original_digest,
                }],
            }
        ),
        encoding="utf-8",
    )
    marker = {
        "video": str(video),
        "mode": "all",
        "status": "completed",
        "fingerprint": _fingerprint(video),
        "saved_files": ["frames/frame_pts_100.png"],
        "outputs": [
            {
                "file": "frames/frame_pts_100.png",
                "metadata": "frames/metadata.json",
                "pts": 100,
                "sha256": digest,
                "original_file": original_file,
                "original_sha256": original_digest,
                "identity_group": {"mode": "all", "index": 1},
            }
        ],
    }
    (folder / "batch-result.json").write_text(json.dumps(marker), encoding="utf-8")
    return source, output_root, complete, video, subtitle, folder, report


def versioned_character_fixture(tmp_path: Path, *, with_original: bool = False):
    source, output_root, complete, video, subtitle, folder, report = fixture(tmp_path)
    character_dir = folder / "character_01"
    frame = character_dir / "frame_pts_100.png"
    Image.new("RGB", (48, 29), (9, 10, 11)).save(frame, format="PNG")
    digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    original_file = None
    original_relative = None
    original_digest = None
    if with_original:
        original_dir = character_dir / "originals"
        original_dir.mkdir()
        original = original_dir / frame.name
        Image.new("RGB", (48, 29), (15, 16, 17)).save(original, format="PNG")
        original_file = "character_01/originals/" + frame.name
        original_relative = "originals/" + frame.name
        original_digest = hashlib.sha256(original.read_bytes()).hexdigest()
    revision = 3
    (character_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_fingerprint": _fingerprint(video),
                "output_manifest_revision": revision,
                "exports": [{
                    "file": frame.name,
                    "pts": 100,
                    "source_pts": 100,
                    "sha256": digest,
                    "original_relative_path": original_relative,
                    "original_sha256": original_digest,
                }],
            }
        ),
        encoding="utf-8",
    )
    marker = {
        "video": str(video),
        "mode": "character",
        "status": "completed",
        "fingerprint": _fingerprint(video),
        "output_manifest_revision": revision,
        "saved_files": ["character_01/frame_pts_100.png"],
        "outputs": [
            {
                "file": "character_01/frame_pts_100.png",
                "metadata": "character_01/metadata.json",
                "pts": 100,
                "sha256": digest,
                "original_file": original_file,
                "original_sha256": original_digest,
                "identity_group": {"mode": "character", "index": 1},
            }
        ],
        "characters": [{"index": 1, "status": "completed", "exported": 1}],
    }
    (folder / "batch-result.json").write_text(json.dumps(marker), encoding="utf-8")
    return source, output_root, complete, video, subtitle, folder, report


def test_completed_video_and_paired_subtitle_moved_after_png_verification(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = fixture(tmp_path)
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["videos_archived"] == 1
    assert summary["files_moved"] == 2
    assert summary["errors"] == 0
    assert not video.exists() and not subtitle.exists()
    assert (complete / video.name).read_bytes() == b"mock video content"
    assert (complete / subtitle.name).is_file()
    assert (folder / "character_01" / "frame_0001_pts_100.png").is_file()
    # Re-run: no overwriting and already archived is verified by fingerprint.
    next_summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert next_summary["videos_already_archived"] == 1
    assert next_summary["files_moved"] == 0


def test_all_mode_frame_manifest_is_verified_and_archived(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = all_mode_fixture(tmp_path)
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["videos_archived"] == 1
    assert summary["errors"] == 0
    assert not video.exists() and not subtitle.exists()
    assert (folder / "frames" / "frame_pts_100.png").is_file()


def test_all_mode_original_sidecar_is_verified_and_archived(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = all_mode_fixture(tmp_path, with_original=True)
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["videos_archived"] == 1
    assert summary["errors"] == 0
    assert (folder / "frames" / "originals" / "frame_pts_100.png").is_file()


def test_all_mode_missing_original_sidecar_is_not_archived(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = all_mode_fixture(tmp_path, with_original=True)
    (folder / "frames" / "originals" / "frame_pts_100.png").unlink()
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["errors"] == 1
    assert video.is_file() and subtitle.is_file()
    assert not complete.exists()


def test_all_mode_original_sidecar_path_must_stay_in_expected_folder(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = all_mode_fixture(tmp_path, with_original=True)
    marker_path = folder / "batch-result.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["outputs"][0]["original_file"] = "frames/originals/../frame_pts_100.png"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["errors"] == 1
    assert video.is_file() and subtitle.is_file()
    assert not complete.exists()


def test_all_mode_manifest_must_match_saved_files_exactly(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = all_mode_fixture(tmp_path)
    marker_path = folder / "batch-result.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["outputs"][0]["file"] = "frames/frame_pts_101.png"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["errors"] == 1
    assert video.is_file() and subtitle.is_file()
    assert not complete.exists()


def test_all_mode_metadata_checks_exact_pts_provenance(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = all_mode_fixture(tmp_path)
    metadata_path = folder / "frames" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["exports"][0]["source_pts"] = 101
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["errors"] == 1
    assert video.is_file() and subtitle.is_file()
    assert not complete.exists()


def test_all_mode_png_digest_mismatch_is_not_archived(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = all_mode_fixture(tmp_path)
    marker_path = folder / "batch-result.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["outputs"][0]["sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["errors"] == 1
    assert video.is_file() and subtitle.is_file()
    assert not complete.exists()


def test_versioned_character_manifest_is_verified_and_archived(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = versioned_character_fixture(tmp_path)
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["videos_archived"] == 1
    assert summary["errors"] == 0
    assert not video.exists() and not subtitle.exists()
    assert (folder / "character_01" / "frame_pts_100.png").is_file()


def test_versioned_character_original_sidecar_is_verified_and_archived(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = versioned_character_fixture(tmp_path, with_original=True)
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["videos_archived"] == 1
    assert summary["errors"] == 0
    assert (folder / "character_01" / "originals" / "frame_pts_100.png").is_file()


def test_versioned_character_original_sidecar_digest_tampering_is_rejected(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = versioned_character_fixture(tmp_path, with_original=True)
    original = folder / "character_01" / "originals" / "frame_pts_100.png"
    Image.new("RGB", (48, 29), (20, 21, 22)).save(original, format="PNG")
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["errors"] == 1
    assert video.is_file() and subtitle.is_file()
    assert not complete.exists()


def test_no_logo_sidecar_absence_must_match_metadata(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = all_mode_fixture(tmp_path)
    metadata_path = folder / "frames" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["exports"][0]["original_sha256"] = "1" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["errors"] == 1
    assert video.is_file() and subtitle.is_file()
    assert not complete.exists()


def test_versioned_character_pts_mismatch_is_not_archived(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = versioned_character_fixture(tmp_path)
    metadata_path = folder / "character_01" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["exports"][0]["source_pts"] = 101
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["errors"] == 1
    assert video.is_file() and subtitle.is_file()
    assert not complete.exists()


def test_versioned_character_requires_exact_pts_filename(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = versioned_character_fixture(tmp_path)
    metadata_path = folder / "character_01" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["exports"][0]["file"] = "frame_0001_pts_100.png"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["errors"] == 1
    assert video.is_file() and subtitle.is_file()
    assert not complete.exists()


def test_versioned_character_output_digest_mismatch_is_not_archived(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = versioned_character_fixture(tmp_path)
    marker_path = folder / "batch-result.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["outputs"][0]["sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["errors"] == 1
    assert video.is_file() and subtitle.is_file()
    assert not complete.exists()


def test_incomplete_processing_keeps_source_and_subtitle(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = fixture(tmp_path, status="incomplete")
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["videos_left_in_source"] == 1
    assert summary["files_moved"] == 0
    assert video.is_file() and subtitle.is_file()
    assert not complete.exists()


def test_dry_run_verifies_but_does_not_move(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = fixture(tmp_path)
    result = archive_report(report, source_dir=source, completed_dir=complete, output_root=out, dry_run=True)
    assert result["jobs"][0]["status"] == "would_archive"
    assert result["files_moved"] == 0
    assert video.is_file() and subtitle.is_file()


def test_destination_collision_does_not_overwrite_and_does_not_move_video(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = fixture(tmp_path)
    complete.mkdir()
    (complete / subtitle.name).write_text("existing unrelated data")
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["errors"] == 1
    assert summary["files_moved"] == 0
    assert summary["collisions"] == 1
    assert summary["jobs"][0]["collisions"] == [subtitle.name]
    assert video.is_file() and subtitle.is_file()
    assert (complete / subtitle.name).read_text() == "existing unrelated data"


def test_retry_reconciles_subtitle_after_video_was_already_archived(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = fixture(tmp_path)
    complete.mkdir()
    video.rename(complete / video.name)
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["videos_already_archived"] == 1
    assert summary["files_moved"] == 1
    assert summary["collisions"] == 0
    assert summary["jobs"][0]["moved"] == [subtitle.name]
    assert not subtitle.exists()
    assert (complete / video.name).read_bytes() == b"mock video content"
    assert (complete / subtitle.name).is_file()


def test_retry_reports_subtitle_collision_without_overwriting(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = fixture(tmp_path)
    complete.mkdir()
    video.rename(complete / video.name)
    existing = complete / subtitle.name
    existing.write_text("existing unrelated subtitle")
    summary = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert summary["videos_already_archived"] == 1
    assert summary["files_moved"] == 0
    assert summary["collisions"] == 1
    assert summary["jobs"][0]["collisions"] == [subtitle.name]
    assert subtitle.read_text().startswith("WEBVTT")
    assert existing.read_text() == "existing unrelated subtitle"


def test_missing_export_is_not_eligible_for_move(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = fixture(tmp_path)
    (folder / "character_01" / "frame_0001_pts_100.png").unlink()
    result = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert result["errors"] == 1
    assert video.is_file() and subtitle.is_file()


def test_source_changed_after_export_is_not_archived(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = fixture(tmp_path)
    video.write_bytes(b"changed video")
    result = archive_report(report, source_dir=source, completed_dir=complete, output_root=out)
    assert result["errors"] == 1
    assert video.is_file() and subtitle.is_file()


def test_watcher_only_acts_when_final_batch_report_is_written(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = fixture(tmp_path)
    report_file = out / "batch_run_test.json"
    report_file.write_text(json.dumps(report), encoding="utf-8")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "stdout.log").write_text(f"[batch] completed; report: {report_file}\n", encoding="utf-8")
    (job_dir / "metadata.json").write_text(
        json.dumps({"job_id": "fixture", "started_at": time.time() - 120, "status": "running", "finished_at": None}),
        encoding="utf-8",
    )
    assert watch_managed_job(job_dir, poll_seconds=0.01, timeout_hours=0.01) == 0
    assert not video.exists()
    assert (complete / subtitle.name).is_file()
    receipt = out / "archive_batch_test-batch.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["videos_archived"] == 1


def test_watcher_does_not_move_without_a_final_report(tmp_path: Path):
    source, out, complete, video, subtitle, folder, report = fixture(tmp_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "stdout.log").write_text("processing in progress\n")
    (job_dir / "metadata.json").write_text(
        json.dumps({"job_id": "fixture", "started_at": time.time() - 120, "status": "failed", "finished_at": time.time()}),
        encoding="utf-8",
    )
    assert watch_managed_job(job_dir, poll_seconds=0.01, timeout_hours=0.01) == 2
    assert video.is_file() and subtitle.is_file()
