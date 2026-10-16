from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import anime_picker.logo_removal as logo_removal
from anime_picker.logo_removal import LogoProcessor, ProfileStore, build_mask


def _profile(template: Image.Image, *, reference_size=(128, 96), threshold=0.55):
    width, height = reference_size
    return {
        "version": 1,
        "id": "test-logo",
        "name": "Test logo",
        "reference_size": [width, height],
        "roi": [80, 0, width, 40],
        "mask_coordinates": "full_frame",
        "mask": {"rectangles": [[96, 10, 120, 24]], "polygons": []},
        "template_image": template,
        "detector": {"threshold": threshold, "aspect_tolerance": 0.025},
    }


def _logo_frame(size=(128, 96), logo=True):
    image = Image.new("RGB", size, (80, 80, 80))
    if logo:
        draw = ImageDraw.Draw(image)
        draw.rectangle((96, 10, 119, 23), fill=(245, 245, 245))
        draw.line((98, 12, 116, 21), fill=(20, 20, 20), width=2)
    return image


def test_profile_store_validates_profile_and_keeps_assets_user_local(tmp_path: Path):
    profile_dir = tmp_path / "test-logo"
    profile_dir.mkdir()
    template = _logo_frame().crop((80, 0, 128, 40))
    template.save(profile_dir / "template.png")
    (profile_dir / "profile.json").write_text(
        json.dumps({
            **_profile(template),
            "template_image": "template.png",
        }),
        encoding="utf-8",
    )

    store = ProfileStore(tmp_path)
    listed = store.list()
    loaded = store.load()
    assert [item["id"] for item in listed] == ["test-logo"]
    assert loaded[0]["_template_path"].endswith("template.png")
    assert "_template_path" not in listed[0]


def test_absent_logo_is_skipped_without_loading_model():
    template = _logo_frame().crop((80, 0, 128, 40))
    processor = LogoProcessor(model_root="missing-model", require_gpu=False)

    def fail_if_loaded():
        raise AssertionError("a no-hit frame must not load Anime-LaMa")

    processor._load_model = fail_if_loaded  # type: ignore[method-assign]
    result = processor.process(_logo_frame(logo=False), [_profile(template)])

    assert result.metadata["status"] == "skipped"
    assert result.metadata["profile_hits"] == []
    assert result.image.tobytes() == _logo_frame(logo=False).tobytes()
    assert np.asarray(result.edit_mask).max() == 0


def test_match_preserves_dimensions_and_pixels_outside_declared_edit_mask():
    source = _logo_frame()
    template = source.crop((80, 0, 128, 40))
    processor = LogoProcessor(model_root="unused", require_gpu=False)

    def fake_inpaint(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.full_like(patch, (10, 20, 30))

    processor._inpaint = fake_inpaint  # type: ignore[method-assign]
    result = processor.process(source, [_profile(template)])

    assert result.metadata["status"] == "processed"
    assert result.image.size == source.size
    assert result.edit_mask.size == source.size
    before = np.asarray(source.convert("RGB"))
    after = np.asarray(result.image.convert("RGB"))
    edit_mask = np.asarray(result.edit_mask) > 0
    assert np.array_equal(before[~edit_mask], after[~edit_mask])
    assert np.any(before[edit_mask] != after[edit_mask])
    hit = result.metadata["profile_hits"][0]
    assert hit["bounds"] == result.metadata["edit_bounds"][0]
    assert hit["bounds"] == result.metadata["bounds"][0]
    assert hit["mask_bounds"] == [96, 10, 120, 24]
    # The 11x11 Gaussian blend expands support by five pixels on every side.
    assert hit["bounds"] == [91, 5, 125, 29]


def test_unsupported_aspect_is_skipped_without_model_lookup():
    template = _logo_frame().crop((80, 0, 128, 40))
    processor = LogoProcessor(model_root="unused", require_gpu=False)
    processor._load_model = lambda: (_ for _ in ()).throw(AssertionError("must not load"))  # type: ignore[method-assign]
    result = processor.process(Image.new("RGB", (256, 96), (80, 80, 80)), [_profile(template)])

    assert result.metadata["status"] == "skipped"
    assert "unsupported aspect" in " ".join(result.metadata.get("skipped_profiles", []))


def test_build_mask_is_binary_and_rejects_out_of_roi_coordinates():
    mask = build_mask((20, 10), (10, 5, 30, 15), (12, 6, 18, 9))
    assert mask.shape == (10, 20)
    assert set(np.unique(mask)) == {0, 255}
    assert int(mask.sum()) == 6 * 3 * 255

    try:
        build_mask((20, 10), (10, 5, 30, 15), (9, 6, 18, 9))
    except ValueError:
        pass
    else:
        raise AssertionError("coordinates outside the ROI must be rejected")


def test_prepare_reuses_verified_existing_model_without_network(tmp_path: Path, monkeypatch):
    model_root = tmp_path / "anime-lama"
    model_root.mkdir()
    model_path = model_root / logo_removal.MODEL_FILENAME
    payload = b"verified-test-model"
    model_path.write_bytes(payload)
    md5 = hashlib.md5(payload).hexdigest()
    sha256 = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(logo_removal, "MODEL_SIZE", len(payload))
    monkeypatch.setattr(logo_removal, "MODEL_MD5", md5)
    monkeypatch.setattr(logo_removal, "MODEL_SHA256", sha256)
    monkeypatch.setattr(
        logo_removal.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("valid bytes must be reused")),
    )

    result = LogoProcessor.prepare(model_root)

    assert result["ready"] is True
    assert model_path.read_bytes() == payload
    manifest = json.loads((model_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == sha256


def test_model_status_digest_cache_invalidates_on_file_change(tmp_path: Path, monkeypatch):
    model_root = tmp_path / "anime-lama"
    model_root.mkdir()
    model_path = model_root / logo_removal.MODEL_FILENAME
    model_path.write_bytes(b"first")
    original_md5, original_sha256 = logo_removal._md5, logo_removal._sha256
    calls = {"md5": 0, "sha256": 0}

    def counted_md5(path):
        calls["md5"] += 1
        return original_md5(path)

    def counted_sha256(path):
        calls["sha256"] += 1
        return original_sha256(path)

    monkeypatch.setattr(logo_removal, "_md5", counted_md5)
    monkeypatch.setattr(logo_removal, "_sha256", counted_sha256)
    LogoProcessor.status(model_root)
    LogoProcessor.status(model_root)
    assert calls == {"md5": 1, "sha256": 1}

    model_path.write_bytes(b"second")
    LogoProcessor.status(model_root)
    assert calls == {"md5": 2, "sha256": 2}
