"""Offline contract tests for :mod:`anime_picker.models`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anime_picker import models


def _write_ready_bundle(root: Path, config: dict) -> None:
    tagger = root / "pixai-tagger-v1.0"
    tagger.mkdir(parents=True)
    (tagger / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"revision": "test-revision", "files": []}), encoding="utf-8")
    (root / ".status").write_text(json.dumps({"ready": True, "revision": "test-revision"}), encoding="utf-8")


def test_status_requires_complete_local_marker(tmp_path: Path) -> None:
    assert models.ModelService.status(tmp_path)["ready"] is False
    _write_ready_bundle(tmp_path, {"tags": ["1girl"]})
    status = models.ModelService.status(tmp_path)
    assert status["ready"] is True
    assert "test-revision" in status["details"]


def test_vocabulary_reads_pixai_categories_and_splits(tmp_path: Path) -> None:
    _write_ready_bundle(
        tmp_path,
        {
            "tags": ["1girl", "blue hair", "artist name"],
            "tags_split": [["general", 2], ["character", 1]],
        },
    )
    service = models.ModelService(tmp_path, require_gpu=False)
    assert service.vocabulary() == {"1girl": "general", "blue hair": "general", "artist name": "character"}


def test_detect_returns_pixel_boxes_and_kinds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_ready_bundle(tmp_path, {"tags": []})
    calls: list[dict] = []

    def person(image, **kwargs):
        calls.append(kwargs)
        return [((1, 2, 30, 40), "person", 0.91)]

    def heads(image, **kwargs):
        return [{"box": [0.1, 0.2, 0.5, 0.6], "score": 0.8, "kind": "head"}]

    monkeypatch.setattr(models, "detect_person", person)
    monkeypatch.setattr(models, "detect_heads", heads)
    monkeypatch.setattr(models, "detect_head", None)
    image = pytest.importorskip("PIL.Image").new("RGB", (100, 80))
    result = models.ModelService(tmp_path, require_gpu=False).detect(image)
    assert result == [
        {"box": [1.0, 2.0, 30.0, 40.0], "score": 0.91, "kind": "person"},
        {"box": [10.0, 16.0, 50.0, 48.0], "score": 0.8, "kind": "head"},
    ]
    assert calls and calls[0]["level"] == "m" and calls[0]["version"] == "v1.1"


def test_distance_uses_official_batch_ccip_and_returns_reference_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_ready_bundle(tmp_path, {"tags": []})
    seen: list[np.ndarray] = []

    def differences(values, **kwargs):
        seen.append(np.asarray(values))
        count = len(values)
        return np.arange(count * count, dtype=np.float32).reshape(count, count)

    monkeypatch.setattr(models, "ccip_batch_differences", differences)
    monkeypatch.setattr(models, "ccip_batch_extract_features", lambda images, **kwargs: np.ones((len(images), 4), dtype=np.float32))
    monkeypatch.setattr(models, "ccip_difference", None)
    service = models.ModelService(tmp_path, require_gpu=False)
    result = service.distance(np.ones((2, 4)), np.ones((3, 4)))
    assert result.shape == (2, 3)
    assert np.array_equal(result, np.arange(25, dtype=np.float32).reshape(5, 5)[:2, 2:])
    assert seen and seen[0].shape == (5, 4)


def test_tag_keeps_low_cutoff_category_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_ready_bundle(tmp_path, {"tags": ["1girl", "blue hair"], "tags_split": [["general", 2]]})

    class FakeTagger:
        def __call__(self, image, **kwargs):
            assert kwargs["threshold"] == 0.0
            return {"results": {"general": {"1girl": 0.02, "blue hair": 0.88}}}

    service = models.ModelService(tmp_path, require_gpu=False)
    monkeypatch.setattr(service, "_tagger", FakeTagger())
    assert service.tag(object()) == {"general": {"1girl": 0.02, "blue hair": 0.88}}


def test_cpu_precision_is_explicit_and_part_of_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_ready_bundle(tmp_path, {"tags": []})
    monkeypatch.setenv("ANIME_PICKER_TAG_DTYPE", "bfloat16")
    service = models.ModelService(tmp_path, require_gpu=False)
    assert service.tag_precision == "float32"
    assert ":tag-float32" in service.revision
    assert service.identity_revision == service.manifest_revision
    assert service.tag_revision == service.revision
    diagnostics = service.diagnostics()
    assert diagnostics["tag_precision"] == "float32"
    assert diagnostics["require_gpu"] is False
