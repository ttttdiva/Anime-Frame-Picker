from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from anime_picker import curation, pipeline


def _anime_frame(color: str, *, variation: bool = False) -> Image.Image:
    image = Image.new("RGB", (96, 72), color)
    draw = ImageDraw.Draw(image)
    draw.ellipse((28, 9, 68, 49), fill="#ffd9bd", outline="#301b24", width=2)
    draw.ellipse((38, 23, 43, 28), fill="#10101a")
    draw.ellipse((54, 23, 59, 28), fill="#10101a")
    if variation:
        draw.ellipse((43, 34, 54, 40), fill="#a8315b")
    else:
        draw.line((44, 37, 53, 37), fill="#a8315b", width=2)
    return image


class _SparseReader:
    def __init__(self, _path: str):
        self.info = SimpleNamespace(duration=4.0, fps=24.0)

    def sample(self, start, end, step, max_side=640):
        assert (start, end, step) == (0.0, 4.0, 1.0)
        assert max_side == 640
        images = [
            _anime_frame("#5a3c86"),
            _anime_frame("#5a3c86"),
            Image.new("RGB", (96, 72), "#303030"),
            _anime_frame("#164e67", variation=True),
        ]
        for pts, image in enumerate(images, 100):
            yield SimpleNamespace(pts=pts, time=float(pts - 100), stream_index=0, image=image)


class _Tagger:
    revision = "pixai-fixture-v1"

    def __init__(self, *, fail: bool = False):
        self.calls = []
        self.fail = fail

    def tag_many(self, images, batch_size=8):
        self.calls.append((len(images), batch_size))
        if self.fail:
            raise RuntimeError("fixture CUDA batch failure")
        rows = []
        for image in images:
            person_present = image.getpixel((0, 0)) != (48, 48, 48)
            rows.append(
                {
                    "general": {
                        "1girl": 0.92 if person_present else 0.01,
                        "smile": 0.73 if person_present else 0.01,
                    }
                }
            )
        return rows


def test_automatic_mode_tags_sparse_samples_and_curates_character_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "VideoReader", _SparseReader)
    captured: dict[str, list[dict]] = {}
    curate = curation.curate_candidates

    def capture_records(records, **kwargs):
        records = list(records)
        captured["records"] = records
        return curate(records, **kwargs)

    monkeypatch.setattr(curation, "curate_candidates", capture_records)
    model = _Tagger()
    result = pipeline.run_search(
        "fixture.mp4",
        [],
        [],
        [],
        {
            "selection_mode": "automatic",
            "start": 0.0,
            "end": 4.0,
            "coarse_step": 1.0,
            "auto_sample_max_side": 640,
        },
        model,
        tmp_path,
    )

    stats = result["stats"]
    assert stats["sampled_frames"] == 4
    assert stats["tagged_frames"] == 4
    assert stats["character_candidates"] == 3
    assert stats["excluded_no_character"] == 1
    assert stats["excluded_near_duplicate"] == 1
    assert stats["returned"] == 2
    assert stats["identity_model_used"] is False
    assert stats["identity_clustering_used"] is False
    assert stats["detector_calls"] == 0
    assert stats["curation_report_saved"] is True
    assert (tmp_path / "curation-report.json").is_file()
    first_input = captured["records"][0]
    assert first_input["technical_quality"] == pytest.approx(first_input["quality_metrics"]["quality"])
    assert first_input["source_score"] == pytest.approx(0.5)
    assert first_input["tags"]["general"]["smile"] == pytest.approx(0.73)
    assert model.calls == [(4, 8)]
    assert len({item["pts"] for item in result["candidates"]}) == 2
    assert any(item["pts"] in {100, 101} for item in result["candidates"])
    assert any(item["pts"] == 103 for item in result["candidates"])
    for item in result["candidates"]:
        assert item["selection_mode"] == "automatic_character"
        assert item["character_presence"] is True
        assert item["group_id"]
        assert item["tags"]["frame"]["1girl"] == pytest.approx(0.92)
        assert Path(item["preview_path"]).is_file()
        assert Path(item["crop_path"]).is_file()


def test_automatic_mode_propagates_gpu_tagging_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "VideoReader", _SparseReader)
    model = _Tagger(fail=True)
    with pytest.raises(pipeline.PipelineError, match="GPU inference was not downgraded"):
        pipeline.run_search(
            "fixture.mp4",
            [],
            [],
            [],
            {"selection_mode": "automatic", "start": 0.0, "end": 4.0, "coarse_step": 1.0},
            model,
            tmp_path,
        )


def test_automatic_mode_defaults_to_one_second_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "VideoReader", _SparseReader)
    result = pipeline.run_search(
        "fixture.mp4",
        [],
        [],
        [],
        {"selection_mode": "automatic", "start": 0.0, "end": 4.0},
        _Tagger(),
        tmp_path,
    )
    assert result["stats"]["sample_step"] == pytest.approx(1.0)


def test_automatic_mode_rejects_subject_scope_instead_of_relabeling_it(tmp_path):
    with pytest.raises(pipeline.PipelineError, match="frame-scoped tag conditions only"):
        pipeline.run_search(
            "fixture.mp4",
            [],
            [{"tag": "smile", "scope": "subject"}],
            [],
            {"selection_mode": "automatic"},
            _Tagger(),
            tmp_path,
        )


def test_automatic_mode_rejects_explicit_cpu_service(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "VideoReader", _SparseReader)
    model = _Tagger()
    model.require_gpu = False
    with pytest.raises(pipeline.PipelineError, match="CPU fallback is disabled"):
        pipeline.run_search(
            "fixture.mp4",
            [],
            [],
            [],
            {"selection_mode": "automatic", "start": 0.0, "end": 4.0, "coarse_step": 1.0},
            model,
            tmp_path,
        )


def test_reference_mode_still_requires_references(tmp_path, monkeypatch):
    class IdentityModel:
        def embed(self, _images):
            raise AssertionError("no reference should exit before feature extraction")

        def distance(self, _left, _right):
            raise AssertionError("no reference should exit before distance scoring")

        def detect(self, _image):
            return []

    monkeypatch.setattr(pipeline, "VideoReader", None)
    result = pipeline.run_search(
        "fixture.mp4", [], [], [], {"selection_mode": "reference"}, IdentityModel(), tmp_path
    )
    assert result["stats"]["error"] == "no_references"
    assert result["candidates"] == []
