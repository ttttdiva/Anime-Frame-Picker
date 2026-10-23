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
    monkeypatch.setattr(models.ModelService, "_ensure_detector", lambda self: None)
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
    monkeypatch.setattr(models.ModelService, "_ensure_ccip", lambda self, **kwargs: None)
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
    service._tag_precision = "float32"
    monkeypatch.setattr(service, "_tagger", FakeTagger())
    assert service.tag(object()) == {"general": {"1girl": 0.02, "blue hair": 0.88}}


def test_tag_many_preserves_order_normalizes_rows_and_records_per_image_telemetry(tmp_path: Path) -> None:
    _write_ready_bundle(tmp_path, {"tags": []})
    images = [object(), object(), object()]

    class FakeTagger:
        def __init__(self) -> None:
            self.images = None
            self.kwargs = None

        def __call__(self, values, **kwargs):
            self.images = values
            self.kwargs = kwargs
            return [
                {"results": {"general": {f"tag-{index}": score}}}
                for index, score in enumerate((0.2, 0.3, 0.4))
            ]

    service = models.ModelService(tmp_path, require_gpu=False)
    service._tag_precision = "float32"
    fake = FakeTagger()
    service._tagger = fake

    result = service.tag_many(images, batch_size=2)

    assert fake.images == images
    assert fake.kwargs == {"threshold": 0.0, "min_threshold": 0.0, "batch_size": 2}
    assert result == [
        {"general": {"tag-0": 0.2}},
        {"general": {"tag-1": 0.3}},
        {"general": {"tag-2": 0.4}},
    ]
    telemetry = service.telemetry()["pixai_tag"]
    assert telemetry["calls"] == len(images)
    assert telemetry["seconds"] >= 0.0


def test_tag_many_requires_gpu_device_and_surfaces_batch_failure_without_cpu_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_ready_bundle(tmp_path, {"tags": []})
    pipeline_kwargs: list[dict] = []
    inference_calls: list[dict] = []

    class FakeTagger:
        def __call__(self, images, **kwargs):
            inference_calls.append(kwargs)
            raise RuntimeError("synthetic CUDA inference failure")

    fake_tagger = FakeTagger()

    def fake_pipeline(**kwargs):
        pipeline_kwargs.append(kwargs)
        return fake_tagger

    fake_transformers = type("FakeTransformers", (), {"pipeline": staticmethod(fake_pipeline)})()
    fake_importlib = type(
        "FakeImportlib",
        (),
        {"import_module": staticmethod(lambda name: fake_transformers if name == "transformers" else None)},
    )()
    monkeypatch.setattr(models, "importlib", fake_importlib)

    service = models.ModelService(tmp_path, require_gpu=True)
    service._tag_precision = "float32"
    monkeypatch.setattr(service, "_prepare_runtime", lambda: None)

    with pytest.raises(RuntimeError, match="synthetic CUDA inference failure"):
        service.tag_many([object(), object()], batch_size=2)

    assert pipeline_kwargs and pipeline_kwargs[0]["device"] == 0
    assert len(inference_calls) == 1
    assert inference_calls[0] == {"threshold": 0.0, "min_threshold": 0.0, "batch_size": 2}
    assert service.telemetry()["pixai_tag"]["calls"] == 0


def test_cpu_precision_is_explicit_and_part_of_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_ready_bundle(tmp_path, {"tags": []})
    monkeypatch.setenv("ANIME_PICKER_TAG_DTYPE", "bfloat16")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch = type("FakeTorch", (), {"cuda": FakeCuda()})()
    fake_onnxruntime = type(
        "FakeOnnxRuntime",
        (),
        {"get_available_providers": staticmethod(lambda: ["CPUExecutionProvider"])},
    )()

    def fake_import_module(name: str):
        if name == "torch":
            return fake_torch
        if name == "onnxruntime":
            return fake_onnxruntime
        raise AssertionError(f"unexpected optional runtime import: {name}")

    fake_importlib = type(
        "FakeImportlib",
        (),
        {"import_module": staticmethod(fake_import_module)},
    )()
    monkeypatch.setattr(models, "importlib", fake_importlib)
    service = models.ModelService(tmp_path, require_gpu=False)
    assert service.tag_precision == "float32"
    assert ":tag-float32" in service.revision
    assert service.identity_revision == service.manifest_revision
    assert service.tag_revision == service.revision
    diagnostics = service.diagnostics()
    assert diagnostics["tag_precision"] == "float32"
    assert diagnostics["require_gpu"] is False


def test_telemetry_returns_independent_nested_copies(tmp_path: Path) -> None:
    _write_ready_bundle(tmp_path, {"tags": []})
    service = models.ModelService(tmp_path, require_gpu=False)

    snapshot = service.telemetry()
    snapshot["detector"]["calls"] = 17
    snapshot["detector"]["extra"] = "caller-owned"
    snapshot["caller-owned"] = {"calls": 1}

    fresh = service.telemetry()
    assert fresh["detector"] == {"calls": 0, "seconds": 0.0}
    assert "caller-owned" not in fresh


def test_close_reports_accumulated_telemetry_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_ready_bundle(tmp_path, {"tags": []})
    service = models.ModelService(tmp_path, require_gpu=False)
    service._tag_precision = "float32"

    class FakeTagger:
        def __call__(self, image, **kwargs):
            return {"results": {"general": {"1girl": 0.5}}}

    service._tagger = FakeTagger()
    service.tag(object())
    service.tag(object())
    expected = service.telemetry()
    assert expected["pixai_tag"]["calls"] == 2
    assert expected["pixai_tag"]["seconds"] >= 0.0

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch = type("FakeTorch", (), {"cuda": FakeCuda()})()
    fake_importlib = type(
        "FakeImportlib",
        (),
        {"import_module": staticmethod(lambda name: fake_torch)},
    )()
    monkeypatch.setattr(models, "importlib", fake_importlib)
    monkeypatch.setattr(models, "sys", type("FakeSys", (), {"modules": {}})())

    first = service.close()
    assert first["performed"] is True
    assert first["model_telemetry"] == expected
    assert first["cuda_before_empty_cache"] == {"available": False}
    assert first["cuda_after_empty_cache"] == {"available": False}
    assert first["errors"] == []

    second = service.close()
    assert second["performed"] is False
    assert second["already_closed"] is True
    assert second["model_telemetry"] == expected


def test_runtime_cache_root_is_separate_from_prepared_model_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    runtime_cache = tmp_path / "staging-cache"
    _write_ready_bundle(bundle, {"tags": []})

    service = models.ModelService(
        bundle,
        require_gpu=False,
        runtime_cache_root=runtime_cache,
    )

    assert service.diagnostics()["cache_root"] == str(runtime_cache)
    assert models.os.environ["HF_HOME"] == str(runtime_cache / "huggingface")
    assert models.os.environ["HF_HUB_CACHE"] == str(runtime_cache / "huggingface" / "hub")
    assert models.os.environ["IMGUTILS_CACHE"] == str(runtime_cache / "imgutils")
    assert not (bundle / "cache").exists()
