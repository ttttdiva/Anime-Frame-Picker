from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from PIL import Image

import anime_picker.output_layout as output_layout
from anime_picker.output_layout import _atomic_json, _registry_path, _source_identity, publish_image_set, publish_png, register_public_receipts, safe_video_name, video_output_directory


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_safe_video_name_handles_windows_invalid_and_reserved_names() -> None:
    assert safe_video_name('A/B: "special"?') == "A_B_ _special__"
    assert safe_video_name("CON") == "_CON"
    assert safe_video_name("LPT1.txt") == "_LPT1.txt"
    assert safe_video_name("... ") == "video"
    assert len(safe_video_name("a" * 400)) == 96


def test_video_output_directory_is_stable_and_collision_safe(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    state = tmp_path / ".local" / "state"
    first = video_output_directory(outputs, "Episode 1.mp4", "fingerprint-A", state_root=state)
    assert first == outputs / "Episode 1" / "images"
    assert video_output_directory(outputs, "Episode 1.mp4", "fingerprint-A", state_root=state) == first

    second = video_output_directory(outputs, "Episode 1.mkv", "fingerprint-B", state_root=state)
    assert second != first
    assert second.parent.name.startswith("Episode 1_")
    assert video_output_directory(outputs, "Episode 1.mkv", "fingerprint-B", state_root=state) == second
    assert all(child.name not in {"sources.json", "sources.json.lock"} for child in outputs.iterdir())


def test_video_identity_ignores_volatile_mtime_when_content_hash_matches(tmp_path: Path) -> None:
    digest = "a" * 64
    outputs = tmp_path / "outputs"
    state = tmp_path / ".local" / "state"
    first = video_output_directory(outputs, "Same.mp4", f"100:200:{digest}", state_root=state)
    repeated = video_output_directory(outputs, "Same.mp4", f"100:999:{digest}", state_root=state)
    assert repeated == first


def test_video_directory_migrates_legacy_raw_fingerprint_registry_key(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    state = tmp_path / ".local" / "state"
    display_name = "Episode.mp4"
    fingerprint = f"100:200:{'b' * 64}"
    legacy_identity = hashlib.sha256((display_name.casefold() + "\0" + fingerprint).encode()).hexdigest()
    legacy_folder = outputs / "Episode"
    (legacy_folder / "images").mkdir(parents=True)
    registry = _registry_path(outputs, state)
    _atomic_json(registry, {"revision": 1, "sources": {legacy_identity: "Episode"}})

    resolved = video_output_directory(outputs, display_name, fingerprint, state_root=state)
    payload = json.loads(registry.read_text(encoding="utf-8"))
    stable_identity = _source_identity(display_name, fingerprint)
    assert resolved == legacy_folder / "images"
    assert payload["sources"] == {stable_identity: "Episode"}


def test_publish_png_is_idempotent_and_never_overwrites_pts_conflicts(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    state = tmp_path / ".local" / "state"
    images = video_output_directory(outputs, "Episode.mp4", "fingerprint", state_root=state)
    source_one = tmp_path / "one.png"
    source_two = tmp_path / "two.png"
    Image.new("RGB", (8, 6), (1, 2, 3)).save(source_one)
    Image.new("RGB", (8, 6), (4, 5, 6)).save(source_two)

    first, first_digest, copied_first = publish_png(source_one, images, "frame_pts_24.png", state_root=state)
    repeated, repeated_digest, _ = publish_png(source_one, images, "frame_pts_24.png", state_root=state)
    second, second_digest, _ = publish_png(source_two, images, "frame_pts_24.png", state_root=state)

    assert first == repeated == images / "frame_pts_24.png"
    assert first_digest == repeated_digest == _sha(source_one)
    assert second == images / f"frame_pts_24_{second_digest[:8]}.png"
    assert second_digest == _sha(source_two)
    assert _sha(first) == first_digest
    assert _sha(second) == second_digest
    assert len(list(images.iterdir())) == 2
    if os.stat(source_one).st_dev == os.stat(first).st_dev:
        assert os.stat(source_one).st_ino == os.stat(first).st_ino
        assert copied_first is False


def test_copy_fallback_uses_atomic_rename_when_hardlinks_are_unavailable(tmp_path: Path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    state = tmp_path / ".local" / "state"
    images = video_output_directory(outputs, "Episode.mp4", "fingerprint", state_root=state)
    source = tmp_path / "source.png"
    Image.new("RGB", (9, 7), (2, 4, 6)).save(source)

    def no_hardlinks(*_args, **_kwargs):
        raise OSError("fixture volume does not support hardlinks")

    monkeypatch.setattr(output_layout.os, "link", no_hardlinks)
    public, digest, copied = publish_png(source, images, "frame_pts_5.png", state_root=state)
    assert copied is True
    assert _sha(public) == digest == _sha(source)

    stage_images = state / "staging" / "images"
    stage_images.mkdir(parents=True)
    staged = stage_images / "frame_pts_6.png"
    copied_stage = output_layout._stage_png(source, staged, _sha(source))
    assert copied_stage is True
    assert _sha(staged) == _sha(source)
    assert not list(stage_images.glob("*.staging"))


def test_changed_image_set_archives_old_set_and_rewrites_its_receipt(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    state = tmp_path / ".local" / "state"
    source_one = tmp_path / "first.png"
    source_two = tmp_path / "second.png"
    Image.new("RGB", (8, 6), (1, 2, 3)).save(source_one)
    Image.new("RGB", (8, 6), (7, 8, 9)).save(source_two)
    images = video_output_directory(outputs, "Episode.mp4", "same-source", state_root=state, create_images=False)
    old_receipt = state / "batch" / "old-run.json"
    old_receipt.parent.mkdir(parents=True)
    readme = outputs / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("Accepted set: `Episode/images/`\n", encoding="utf-8")
    old_receipt.write_text(json.dumps({
        "images_directory": "Episode/images",
        "public_files": [{"file": "Episode/images/frame_pts_10.png"}],
        "absolute": str(images / "frame_pts_10.png"),
    }) + "\n", encoding="utf-8")

    first_images, first_public, _, _ = publish_image_set(
        outputs,
        "Episode.mp4",
        "same-source",
        [{"source": source_one, "pts": 10}],
        state_root=state,
        run_id="full-run",
        receipt_paths=[old_receipt, readme],
    )
    assert [row["file"] for row in first_public] == ["Episode/images/frame_pts_10.png"]
    register_public_receipts(outputs, "Episode.mp4", "same-source", state_root=state, receipt_paths=[old_receipt, readme])

    second_images, second_public, _, active = publish_image_set(
        outputs,
        "Episode.mp4",
        "same-source",
        [{"source": source_two, "pts": 20}],
        state_root=state,
        run_id="short-run",
    )
    assert second_images == first_images
    assert [row["file"] for row in second_public] == ["Episode/images/frame_pts_20.png"]
    assert sorted(path.name for path in second_images.iterdir()) == ["frame_pts_20.png"]
    archived_images = outputs / active["previous_archived_directory"]
    assert archived_images.is_dir()
    assert sorted(path.name for path in archived_images.iterdir()) == ["frame_pts_10.png"]
    receipt = json.loads(old_receipt.read_text(encoding="utf-8"))
    assert receipt["images_directory"] == active["previous_archived_directory"]
    assert receipt["public_files"][0]["file"].startswith(active["previous_archived_directory"] + "/")
    assert Path(receipt["absolute"]).is_file()
    assert active["previous_archived_directory"] in readme.read_text(encoding="utf-8")


def test_interrupted_publication_resumes_from_private_staging(tmp_path: Path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    state = tmp_path / ".local" / "state"
    old_source = tmp_path / "old.png"
    new_source = tmp_path / "new.png"
    Image.new("RGB", (8, 6), (1, 1, 1)).save(old_source)
    Image.new("RGB", (8, 6), (2, 2, 2)).save(new_source)
    images, _, _, _ = publish_image_set(
        outputs,
        "Episode.mp4",
        "same-source",
        [{"source": old_source, "pts": 10}],
        state_root=state,
        run_id="old",
    )

    original_rename = Path.rename
    interrupted = {"done": False}

    def fail_stage_swap(path: Path, target: Path):
        if "staging" in path.parts and path.name == "images" and Path(target) == images and not interrupted["done"]:
            interrupted["done"] = True
            raise OSError("fixture interruption after archiving old set")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_stage_swap)
    with pytest.raises(OSError, match="fixture interruption"):
        publish_image_set(
            outputs,
            "Episode.mp4",
            "same-source",
            [{"source": new_source, "pts": 20}],
            state_root=state,
            run_id="new",
        )
    monkeypatch.setattr(Path, "rename", original_rename)
    assert not images.exists()

    resumed, public, _, manifest = publish_image_set(
        outputs,
        "Episode.mp4",
        "same-source",
        [{"source": new_source, "pts": 20}],
        state_root=state,
        run_id="new",
    )
    assert resumed == images
    assert [item["file"] for item in public] == ["Episode/images/frame_pts_20.png"]
    assert sorted(path.name for path in images.iterdir()) == ["frame_pts_20.png"]
    assert manifest["status"] == "complete"
    assert (outputs / manifest["previous_archived_directory"] / "frame_pts_10.png").is_file()


def test_missing_active_public_png_is_archived_and_restored_from_internal_source(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    state = tmp_path / ".local" / "state"
    source = tmp_path / "frame.png"
    Image.new("RGB", (8, 6), (5, 6, 7)).save(source)
    images, _, _, _ = publish_image_set(
        outputs,
        "Episode.mp4",
        "same-source",
        [{"source": source, "pts": 10}],
        state_root=state,
        run_id="same-run",
    )
    target = images / "frame_pts_10.png"
    digest = _sha(target)
    target.unlink()

    repaired, public, _, active = publish_image_set(
        outputs,
        "Episode.mp4",
        "same-source",
        [{"source": source, "pts": 10}],
        state_root=state,
        run_id="same-run",
    )
    assert repaired == images
    assert _sha(target) == digest
    assert [item["file"] for item in public] == ["Episode/images/frame_pts_10.png"]
    assert active["previous_manifest"]["integrity_repair"] == "modified_or_incomplete_public_set_archived_before_republish"
    assert (outputs / active["previous_archived_directory"]).is_dir()


def test_receipt_written_after_concurrent_archive_is_reconciled_by_publication_id(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    state = tmp_path / ".local" / "state"
    source_a = tmp_path / "a.png"
    source_b = tmp_path / "b.png"
    Image.new("RGB", (8, 6), (11, 12, 13)).save(source_a)
    Image.new("RGB", (8, 6), (14, 15, 16)).save(source_b)
    late_receipt = state / "batch" / "late-report.json"

    old_images, old_public, _, old_manifest = publish_image_set(
        outputs,
        "Episode.mp4",
        "same-source",
        [{"source": source_a, "pts": 30}],
        state_root=state,
        run_id="first-publication",
        receipt_paths=[late_receipt],
    )
    new_images, _, _, _ = publish_image_set(
        outputs,
        "Episode.mp4",
        "same-source",
        [{"source": source_b, "pts": 40}],
        state_root=state,
        run_id="second-publication",
    )
    assert old_images == new_images
    late_receipt.parent.mkdir(parents=True)
    late_receipt.write_text(json.dumps({
        "images_directory": "Episode/images",
        "public_files": [old_public[0]["file"]],
        "absolute_images": str(old_images),
    }) + "\n", encoding="utf-8")

    register_public_receipts(
        outputs,
        "Episode.mp4",
        "same-source",
        state_root=state,
        receipt_paths=[late_receipt],
        publication_id=old_manifest["publication_id"],
    )
    receipt = json.loads(late_receipt.read_text(encoding="utf-8"))
    archived = receipt["images_directory"]
    assert archived.startswith("_archive/")
    assert receipt["public_files"][0].startswith(archived + "/")
    assert Path(receipt["absolute_images"]).is_dir()


def test_video_output_directory_rejects_symlink_or_junction_escape(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    state = tmp_path / ".local" / "state"
    outside = tmp_path / "outside"
    outside.mkdir()
    original = video_output_directory(outputs, "Episode.mp4", "fingerprint", state_root=state)
    original.rmdir()
    original.parent.rmdir()
    link = outputs / "Episode"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink creation is unavailable: {error}")
    with pytest.raises(ValueError, match="symlink or junction"):
        video_output_directory(outputs, "Episode.mp4", "fingerprint", state_root=state)
