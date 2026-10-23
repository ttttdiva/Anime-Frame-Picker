from __future__ import annotations

import io
import json
import time
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anime_picker import server


def _client() -> TestClient:
    return TestClient(server.create_app())


def test_bootstrap_is_local_contract(monkeypatch):
    monkeypatch.setattr(server, "_discover_videos", lambda: [])
    with _client() as client:
        response = client.get("/api/bootstrap")
    assert response.status_code == 200
    payload = response.json()
    assert {"videos", "defaults", "models", "aliases", "settings"} <= payload.keys()
    assert payload["defaults"]["model_root"]
    assert payload["settings"]["auto_sample_step"] == 1.0
    assert payload["settings"]["cut_quality_min"] == 0.55


def test_cut_quality_floor_default_and_range_validation():
    settings = server._parse_settings({}, duration=10.0)
    assert settings["cut_quality_min"] == 0.55
    with pytest.raises(server.HTTPException, match="cut_quality_min"):
        server._parse_settings({"cut_quality_min": 1.01}, duration=10.0)


def test_ui_defaults_to_guided_character_selection():
    with _client() as client:
        response = client.get("/")
        script_response = client.get("/static/app.js")
    assert response.status_code == 200
    assert script_response.status_code == 200
    page = response.text
    assert 'id="selectionModeReference" value="reference" checked' in page
    assert 'id="selectionModeAutomatic" value="automatic"' in page
    assert 'id="suggestFrames"' in page
    assert 'data-scope="subject" type="button" disabled' in page
    assert 'class="scope-button active" data-scope="frame"' in page
    assert 'id="tagScopeHelp" class="panel-help scope-help"' in page
    assert "自動選別のタグ条件はフレーム全体に適用します" in page
    assert server._parse_settings({}, duration=10.0)["candidate_count"] == 0
    script = script_response.text
    init = script.split("async function init() {", 1)[1].split("\n  function bindEvents()", 1)[0]
    assert init.index("bindEvents();") < init.index("updateSelectionMode();") < init.index("await loadBootstrap();")
    assert "state.scope = referenceMode ? state.referenceScope : 'frame';" in script
    assert "addTag(name, state.intent, state.scope, alias)" in script


def test_default_ui_exposes_guided_character_selection_and_unlimited_results():
    with _client() as client:
        response = client.get("/")
    assert response.status_code == 200
    assert 'id="suggestFrames"' in response.text
    assert 'id="addSuggestedFrames"' in response.text
    assert 'id="candidateCount" type="number" min="0"' in response.text
    assert "同じキャラの別カット" in response.text
    assert "固定キーで区別" in response.text
    assert "服装・表情・角度の違いだけでは別キャラに分けません" in response.text
    assert "全動画自動quality抽出" in response.text
    assert "quality-owner batch" in response.text
    app_js = (Path(__file__).parents[1] / "anime_picker" / "static" / "app.js").read_text(encoding="utf-8")
    renderer = app_js.split("function renderSuggestedFrames()", 1)[1].split("function addSelectedSuggestedReferences(", 1)[0]
    assert "createCharacterCard(group)" in renderer
    assert "dataset.characterKey" in renderer and "dataset.detectionKey" in renderer
    assert "innerHTML" not in renderer
    css = (Path(__file__).parents[1] / "anime_picker" / "static" / "style.css").read_text(encoding="utf-8")
    assert ".suggestion-image-wrap > img" in css
    assert "height: auto; aspect-ratio: auto; object-fit: contain;" in css


def test_reference_upload_is_bounded_registry_file():
    image = Image.new("RGB", (12, 8), "#ff00aa")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    with _client() as client:
        response = client.post(
            "/api/references",
            files={"files": ("face.png", stream, "image/png")},
        )
        assert response.status_code == 200
        item = response.json()["references"][0]
        assert item["width"] == 12 and item["height"] == 8
        image_response = client.get(item["url"])
    assert image_response.status_code == 200
    assert image_response.headers["content-type"].startswith("image/")


def test_frame_preview_is_small_and_does_not_register_a_reference(monkeypatch):
    video = server.VideoRecord(
        id="a" * 32,
        path=Path("episode.mp4"),
        info={"width": 1920, "height": 1080, "duration": 12.0, "fps": 24.0, "path": "episode.mp4"},
        fingerprint="fingerprint",
        created_at="now",
    )
    monkeypatch.setattr(server, "_get_video", lambda _video_id: video)
    monkeypatch.setattr(server, "_read_video_frame", lambda _path, _time: Image.new("RGB", (1920, 1080), "#456789"))
    reference_ids = set(server.REFERENCE_REGISTRY)
    with _client() as client:
        response = client.get(f"/api/videos/{video.id}/frame-preview?time=4.5")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert server.REFERENCE_REGISTRY.keys() == reference_ids
    with Image.open(io.BytesIO(response.content)) as preview:
        assert preview.size == (480, 270)


def test_source_stat_guard_detects_size_and_mtime_changes(tmp_path):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"fixture")
    stat = source.stat()
    video = server.VideoRecord(
        id="b" * 32,
        path=source,
        info={},
        fingerprint=f"{stat.st_size}:{stat.st_mtime_ns}:fixture-sha",
        created_at="now",
    )
    assert server._source_stat_unchanged(video)
    video.fingerprint = f"{stat.st_size + 1}:{stat.st_mtime_ns}:fixture-sha"
    assert not server._source_stat_unchanged(video)
    video.fingerprint = f"{stat.st_size}:{stat.st_mtime_ns + 1}:fixture-sha"
    assert not server._source_stat_unchanged(video)


def test_character_suggestions_keep_real_person_detections_and_both_people(monkeypatch):
    video = server.VideoRecord(
        id="d" * 32,
        path=Path("episode.mp4"),
        info={"width": 100, "height": 100, "duration": 3.0, "fps": 1.0, "path": "episode.mp4"},
        fingerprint="fingerprint",
        created_at="now",
    )

    class Reader:
        def __init__(self, _path):
            self.info = SimpleNamespace(duration=3.0, fps=1.0)

        def sample(self, start, _end, _step, max_side=960):
            index = round(start)
            colors = {0: "#102030", 1: "#000000", 2: "#405060"}
            image = Image.new("RGB", (100, 100), colors[index])
            yield SimpleNamespace(pts=100 + index, time=float(index), image=image)

    class Models:
        require_gpu = True
        identity_revision = "cached-local-model-revision"

        @staticmethod
        def detect(image):
            pixel = image.getpixel((0, 0))
            if pixel == (16, 32, 48):
                return [
                    {"box": [5, 10, 45, 90], "score": 0.91, "kind": "person"},
                    {"box": [55, 10, 95, 90], "score": 0.87, "kind": "person"},
                ]
            if pixel == (64, 80, 96):
                return [{"box": [1, 1, 15, 15], "score": 0.99, "kind": "person"}]
            return []

    monkeypatch.setattr(server, "VideoReader", Reader)
    monkeypatch.setattr(server, "_get_video", lambda _video_id: video)
    monkeypatch.setattr(server, "_load_model_service", lambda _root: Models())
    monkeypatch.setattr(server, "_source_stat_unchanged", lambda _video: True)
    with _client() as client:
        response = client.post(
            f"/api/videos/{video.id}/character-suggestions",
            json={"times": [0.0, 1.0, 2.0]},
        )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["checked_frames"] == 3
    assert result["detector"] == "person_head"
    assert result["identity_grouping"] is False
    assert len(result["frames"]) == 1
    frame = result["frames"][0]
    assert frame["pts"] == 100
    assert len(frame["detections"]) == 2
    assert frame["preview_url"].endswith("frame-preview?pts=100")
    assert all(item["kind"] == "person" and item["score"] >= 0.3 for item in frame["detections"])


def test_archived_detector_ccip_evidence_is_source_matched_and_paginated(tmp_path, monkeypatch):
    video = server.VideoRecord(
        id="9" * 32,
        path=Path("episode.mp4"),
        info={"width": 100, "height": 100, "duration": 30.0, "fps": 24.0, "path": "episode.mp4"},
        fingerprint="exact-source-fingerprint",
        created_at="now",
    )
    metadata_path = tmp_path / "outputs" / "_archive" / "day" / "batch" / "run" / "character_01" / "metadata.json"
    metadata_path.parent.mkdir(parents=True)
    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "manifest.json").write_text(json.dumps({"revision": "cached-model-revision", "identity_model": "ccip-caformer_b36-24"}), encoding="utf-8")
    metadata_path.write_text(json.dumps({
        "source_fingerprint": video.fingerprint,
        "mode": "automatic",
        "stats": {"detections": 100, "identity_matches": 40, "identity_model": "ccip-caformer_b36-24", "model_revision": "cached-model-revision"},
        "exports": [
            {"pts": 100 + index, "time": float(index), "bbox": [0.1, 0.1, 0.7, 0.9], "identity": 0.9, "distance": 0.1}
            for index in range(13)
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(server, "DEFAULT_OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(server, "DEFAULT_MODEL_ROOT", model_root)
    monkeypatch.setattr(server, "_get_video", lambda _video_id: video)
    monkeypatch.setattr(server, "_load_model_service", lambda _root: (_ for _ in ()).throw(AssertionError("cache page must not invoke detector")))
    monkeypatch.setattr(server, "_source_stat_unchanged", lambda _video: True)
    with _client() as client:
        first = client.post(
            f"/api/videos/{video.id}/character-suggestions",
            json={"page": 0, "times": [float(index) for index in range(12)]},
        )
        second = client.post(
            f"/api/videos/{video.id}/character-suggestions",
            json={"page": 1, "times": [float(index) for index in range(12)]},
        )
    assert first.status_code == second.status_code == 200
    assert first.json()["detector"] == "archived_detector_ccip"
    assert first.json()["cache_total"] == 13
    assert len(first.json()["frames"]) == 12
    assert len(second.json()["frames"]) == 1
    assert first.json()["identity_grouping"] is False
    assert "group" not in first.json()["frames"][0]
    assert first.json()["frames"][0]["source"] == "archived_detector_ccip"


def test_character_suggestions_refuse_cpu_model_and_oversized_page(monkeypatch):
    video = server.VideoRecord(
        id="e" * 32,
        path=Path("episode.mp4"),
        info={"width": 100, "height": 100, "duration": 30.0, "fps": 24.0, "path": "episode.mp4"},
        fingerprint="fingerprint",
        created_at="now",
    )
    calls: list[str] = []

    class CpuModels:
        require_gpu = False

        @staticmethod
        def detect(_image):
            calls.append("detect")
            return []

    monkeypatch.setattr(server, "_get_video", lambda _video_id: video)
    monkeypatch.setattr(server, "_load_model_service", lambda _root: CpuModels())
    monkeypatch.setattr(server, "_source_stat_unchanged", lambda _video: True)
    with _client() as client:
        cpu_response = client.post(
            f"/api/videos/{video.id}/character-suggestions",
            json={"times": [0.0]},
        )
        page_response = client.post(
            f"/api/videos/{video.id}/character-suggestions",
            json={"times": [float(index) for index in range(13)]},
        )
    assert cpu_response.status_code == 503
    assert "CPU fallback was not used" in cpu_response.json()["detail"]
    assert page_response.status_code == 422
    assert calls == []


def test_reference_capture_uses_exact_pts_and_keeps_detection_box(tmp_path, monkeypatch):
    video = server.VideoRecord(
        id="f" * 32,
        path=tmp_path / "episode.mp4",
        info={"width": 100, "height": 100, "duration": 3.0, "fps": 24.0, "path": "episode.mp4"},
        fingerprint="fingerprint",
        created_at="now",
    )
    read_pts: list[int] = []

    class Reader:
        def __init__(self, _path):
            pass

        def get_frame(self, pts, max_side=None):
            read_pts.append(int(pts))
            return SimpleNamespace(pts=int(pts), time=1.25, image=Image.new("RGB", (100, 100), "#123456"))

    monkeypatch.setattr(server, "VideoReader", Reader)
    monkeypatch.setattr(server, "_get_video", lambda _video_id: video)
    monkeypatch.setattr(server, "_source_stat_unchanged", lambda _video: True)
    monkeypatch.setattr(server, "REFERENCE_ROOT", tmp_path / "references")
    monkeypatch.setattr(server, "_executor", lambda: SimpleNamespace(submit=lambda *_args, **_kwargs: None))
    server.REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
    with _client() as client:
        response = client.post(
            f"/api/videos/{video.id}/reference",
            json={"pts": 777, "box": [0.1, 0.1, 0.5, 0.9]},
        )
        created = client.post(
            "/api/jobs",
            json={
                "video_id": video.id,
                "references": [{"id": response.json()["id"], "box": response.json()["box"]}],
                "settings": {"candidate_count": 0},
            },
        )
    assert response.status_code == 200, response.text
    assert read_pts == [777]
    assert response.json()["source_pts"] == 777
    assert response.json()["box"] == [0.1, 0.1, 0.5, 0.9]
    assert response.json()["time"] == 1.25
    assert created.status_code == 200, created.text
    job = server.JOB_REGISTRY[created.json()["job_id"]]
    assert job.reference_items[0]["box"] == [0.1, 0.1, 0.5, 0.9]
    monkeypatch.delitem(server.JOB_REGISTRY, job.id)


def test_character_suggestions_reject_stale_source_before_cache_or_model(monkeypatch):
    video = server.VideoRecord(
        id="g" * 32,
        path=Path("replaced.mp4"),
        info={"width": 100, "height": 100, "duration": 3.0, "fps": 24.0, "path": "replaced.mp4"},
        fingerprint="old-source-fingerprint",
        created_at="now",
    )
    monkeypatch.setattr(server, "_get_video", lambda _video_id: video)
    monkeypatch.setattr(server, "_source_stat_unchanged", lambda _video: False)
    monkeypatch.setattr(server, "_load_model_service", lambda _root: (_ for _ in ()).throw(AssertionError("stale source must not run GPU inference")))
    monkeypatch.setattr(server, "_read_video_frame_at_pts", lambda *_args: (_ for _ in ()).throw(AssertionError("stale source PTS must not be decoded")))
    with _client() as client:
        response = client.post(
            f"/api/videos/{video.id}/character-suggestions",
            json={"times": [0.0]},
        )
        preview = client.get(f"/api/videos/{video.id}/frame-preview?pts=123")
        reference = client.post(f"/api/videos/{video.id}/reference", json={"pts": 123, "box": [0.1, 0.1, 0.5, 0.9]})
    assert response.status_code == 409
    assert "source changed" in response.json()["detail"]
    assert preview.status_code == 409
    assert reference.status_code == 409


def test_candidate_count_zero_is_unlimited_and_has_no_small_hard_ceiling():
    assert server._parse_settings({"candidate_count": 0}, 10.0)["candidate_count"] == 0
    assert server._parse_settings({"candidate_count": 5001}, 10.0)["candidate_count"] == 5001


def test_job_keeps_all_selected_reference_images(monkeypatch):
    video = server.VideoRecord(
        id="c" * 32,
        path=Path("episode.mp4"),
        info={"width": 1920, "height": 1080, "duration": 12.0, "fps": 24.0, "path": "episode.mp4"},
        fingerprint="fingerprint",
        created_at="now",
    )
    references = [
        server.ReferenceRecord(f"{index:032x}", Path(f"ref-{index}.png"), f"ref-{index}.png", 100, 100, "now")
        for index in (1, 2, 3)
    ]
    monkeypatch.setattr(server, "_get_video", lambda _video_id: video)
    monkeypatch.setattr(server, "_executor", lambda: SimpleNamespace(submit=lambda *_args, **_kwargs: None))
    monkeypatch.setitem(server.REFERENCE_REGISTRY, references[0].id, references[0])
    monkeypatch.setitem(server.REFERENCE_REGISTRY, references[1].id, references[1])
    monkeypatch.setitem(server.REFERENCE_REGISTRY, references[2].id, references[2])
    with _client() as client:
        response = client.post(
            "/api/jobs",
            json={
                "video_id": video.id,
                "references": [{"id": item.id, "box": None} for item in references],
                "settings": {"candidate_count": 0},
            },
        )
    assert response.status_code == 200
    job = server.JOB_REGISTRY[response.json()["job_id"]]
    assert len(job.reference_items) == 3
    assert job.settings["candidate_count"] == 0
    monkeypatch.delitem(server.JOB_REGISTRY, job.id)


def test_cross_origin_and_non_loopback_hosts_are_rejected():
    with _client() as client:
        assert client.get("/api/bootstrap", headers={"host": "example.com"}).status_code == 400
        assert client.get(
            "/api/bootstrap", headers={"origin": "https://example.com"}
        ).status_code == 403


def test_unknown_tag_is_rejected_before_queue(monkeypatch):
    record = server.VideoRecord(
        id="a" * 32,
        path=server.PACKAGE_ROOT / "source" / "placeholder.mp4",
        info={"width": 10, "height": 10, "duration": 1.0, "fps": 24.0, "path": "placeholder.mp4"},
        fingerprint="fingerprint",
        created_at="now",
    )
    reference = server.ReferenceRecord(
        id="b" * 32,
        path=server.REFERENCE_ROOT / "placeholder.png",
        name="placeholder.png",
        width=10,
        height=10,
        created_at="now",
    )
    monkeypatch.setattr(server, "_get_video", lambda _video_id: record)
    server.VIDEO_REGISTRY[record.id] = record
    server.REFERENCE_REGISTRY[reference.id] = reference
    with _client() as client:
        response = client.post(
            "/api/jobs",
            json={
                "video_id": record.id,
                "references": [{"id": reference.id, "box": [0, 0, 1, 1]}],
                "wanted": [{"tag": "definitely-unknown", "scope": "frame"}],
                "avoided": [],
                "settings": {},
            },
        )
    assert response.status_code == 422


def test_automatic_job_rejects_subject_scoped_tag_filters(monkeypatch):
    record = server.VideoRecord(
        id="g" * 32,
        path=server.PACKAGE_ROOT / "source" / "placeholder.mp4",
        info={"width": 10, "height": 10, "duration": 1.0, "fps": 24.0, "path": "placeholder.mp4"},
        fingerprint="fingerprint",
        created_at="now",
    )
    monkeypatch.setattr(server, "_get_video", lambda _video_id: record)
    monkeypatch.setattr(server, "_normalize_tags", lambda items, *, field: list(items))
    monkeypatch.setattr(server, "_executor", lambda: pytest.fail("invalid filters must be rejected before queueing"))
    with _client() as client:
        response = client.post(
            "/api/jobs",
            json={
                "video_id": record.id,
                "selection_mode": "automatic",
                "references": [],
                "wanted": [{"tag": "smile", "scope": "subject"}],
                "avoided": [],
                "settings": {},
            },
        )
    assert response.status_code == 422
    assert "frame-scoped" in response.json()["detail"]
    assert "subject-scoped tags" in response.json()["detail"]


def test_automatic_job_accepts_frame_filters_and_defaults_to_one_second_sampling(monkeypatch):
    record = server.VideoRecord(
        id="h" * 32,
        path=server.PACKAGE_ROOT / "source" / "placeholder.mp4",
        info={"width": 10, "height": 10, "duration": 1.0, "fps": 24.0, "path": "placeholder.mp4"},
        fingerprint="fingerprint",
        created_at="now",
    )

    class Queue:
        def submit(self, _function, _job_id):
            return object()

    monkeypatch.setattr(server, "_get_video", lambda _video_id: record)
    monkeypatch.setattr(server, "_normalize_tags", lambda items, *, field: list(items))
    monkeypatch.setattr(server, "_executor", Queue)
    with _client() as client:
        response = client.post(
            "/api/jobs",
            json={
                "video_id": record.id,
                "selection_mode": "automatic",
                "references": [],
                "wanted": [{"tag": "smile", "scope": "frame"}],
                "avoided": [],
                "settings": {},
            },
        )
    assert response.status_code == 200
    job = server.JOB_REGISTRY[response.json()["job_id"]]
    assert job.wanted == [{"tag": "smile", "scope": "frame"}]
    assert job.settings["auto_sample_step"] == 1.0


def test_automatic_job_preserves_an_explicit_sampling_interval(monkeypatch):
    record = server.VideoRecord(
        id="i" * 32,
        path=server.PACKAGE_ROOT / "source" / "placeholder.mp4",
        info={"width": 10, "height": 10, "duration": 1.0, "fps": 24.0, "path": "placeholder.mp4"},
        fingerprint="fingerprint",
        created_at="now",
    )

    class Queue:
        def submit(self, _function, _job_id):
            return object()

    monkeypatch.setattr(server, "_get_video", lambda _video_id: record)
    monkeypatch.setattr(server, "_normalize_tags", lambda items, *, field: list(items))
    monkeypatch.setattr(server, "_executor", Queue)
    with _client() as client:
        response = client.post(
            "/api/jobs",
            json={
                "video_id": record.id,
                "selection_mode": "automatic",
                "references": [],
                "wanted": [],
                "avoided": [],
                "settings": {"coarse_step": 0.6},
            },
        )
    assert response.status_code == 200
    job = server.JOB_REGISTRY[response.json()["job_id"]]
    assert job.settings["auto_sample_step"] == pytest.approx(0.6)


def test_export_uses_candidate_pts_and_writes_metadata(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"video-source")
    output = tmp_path / "exports"
    seen_pts: list[int] = []

    class Reader:
        def __init__(self, _path):
            pass

        def export_png(self, pts, destination):
            seen_pts.append(int(pts))
            Image.new("RGB", (4, 3), "#123456").save(destination, format="PNG")

    monkeypatch.setattr(server, "VideoReader", Reader)
    record = server.VideoRecord(
        id="c" * 32,
        path=source,
        info={"width": 4, "height": 3, "duration": 2.0, "fps": 24.0, "path": str(source), "time_base": "1/24"},
        fingerprint=server._fingerprint(source),
        created_at="now",
        display_name="Friendly Episode.mp4",
    )
    job = server.JobRecord(
        id="d" * 32,
        video_id=record.id,
        reference_items=[],
        wanted=[],
        avoided=[],
        settings={"output_root": str(output)},
        status="done",
        candidates=[{"id": "candidate-1", "pts": 37, "time": 1.541, "bbox": [0.1, 0.2, 0.8, 0.9], "score": 0.91}],
    )
    server.VIDEO_REGISTRY[record.id] = record
    server.JOB_REGISTRY[job.id] = job
    with _client() as client:
        response = client.post(f"/api/jobs/{job.id}/export", json={"candidate_ids": ["candidate-1"]})
    assert response.status_code == 200
    payload = response.json()
    assert seen_pts == [37]
    public_images = Path(payload["directory"])
    assert public_images == output / "Friendly Episode" / "images"
    assert (public_images / "frame_pts_37.png").is_file()
    assert sorted(path.name for path in public_images.iterdir()) == ["frame_pts_37.png"]
    assert not Path(payload["metadata_path"]).is_relative_to(output.resolve())
    metadata = json.loads(Path(payload["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["source_pts"] == [37]
    assert metadata["candidates"][0]["crop"] == [0.1, 0.2, 0.8, 0.9]
    assert metadata["public_files"] == [str(public_images / "frame_pts_37.png")]


def test_export_without_output_root_uses_configured_default(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"video-source")
    default_root = tmp_path / "configured-default"

    class Reader:
        def __init__(self, _path):
            pass

        def export_png(self, _pts, destination):
            Image.new("RGB", (4, 3), "#123456").save(destination, format="PNG")

    monkeypatch.setattr(server, "DEFAULT_OUTPUT_ROOT", default_root)
    monkeypatch.setattr(server, "VideoReader", Reader)
    record = server.VideoRecord(
        id="7" * 32,
        path=source,
        info={"width": 4, "height": 3, "duration": 2.0, "fps": 24.0, "path": str(source), "time_base": "1/24"},
        fingerprint=server._fingerprint(source),
        created_at="now",
    )
    job = server.JobRecord(
        id="8" * 32,
        video_id=record.id,
        reference_items=[],
        wanted=[],
        avoided=[],
        settings={"logo_removal": False},
        status="done",
        candidates=[{"id": "candidate-1", "pts": 37}],
    )
    server.VIDEO_REGISTRY[record.id] = record
    server.JOB_REGISTRY[job.id] = job

    result = server._export_job(job, ["candidate-1"])

    directory = Path(result["directory"])
    assert directory == default_root / "episode" / "images"
    assert (directory / "frame_pts_37.png").is_file()
    assert not (directory / "metadata.json").exists()
    assert Path(result["metadata_path"]).is_file()


def test_export_rejects_source_changed_after_search(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"before")
    record = server.VideoRecord(
        id="e" * 32,
        path=source,
        info={"width": 4, "height": 3, "duration": 2.0, "fps": 24.0, "path": str(source)},
        fingerprint=server._fingerprint(source),
        created_at="now",
    )
    job = server.JobRecord(
        id="f" * 32,
        video_id=record.id,
        reference_items=[],
        wanted=[],
        avoided=[],
        settings={"output_root": str(tmp_path / "exports")},
        status="done",
        candidates=[{"id": "candidate-1", "pts": 2}],
    )
    server.VIDEO_REGISTRY[record.id] = record
    server.JOB_REGISTRY[job.id] = job
    source.write_bytes(b"after")
    with _client() as client:
        response = client.post(f"/api/jobs/{job.id}/export", json={"candidate_ids": ["candidate-1"]})
    assert response.status_code == 409


def test_failed_ui_export_keeps_previous_public_set_and_only_stages_partial_frames(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"partial-export-source")
    output = tmp_path / "outputs"
    fingerprint = server._fingerprint(source)
    state_root = server.default_state_root(output, server.PACKAGE_ROOT)
    images = server.video_output_directory(output, source.name, fingerprint, state_root=state_root)
    Image.new("RGB", (5, 4), (9, 8, 7)).save(images / "frame_pts_99.png", format="PNG")

    class Reader:
        def __init__(self, _path):
            pass

        def export_png(self, pts, destination):
            if int(pts) == 2:
                raise RuntimeError("fixture interrupted after one staged frame")
            Image.new("RGB", (5, 4), (1, 2, 3)).save(destination, format="PNG")

    monkeypatch.setattr(server, "VideoReader", Reader)
    record = server.VideoRecord(
        id="a" * 32,
        path=source,
        info={"width": 5, "height": 4, "duration": 1.0, "fps": 24.0, "path": str(source)},
        fingerprint=fingerprint,
        created_at="now",
    )
    job = server.JobRecord(
        id="b" * 32,
        video_id=record.id,
        reference_items=[],
        wanted=[],
        avoided=[],
        settings={"output_root": str(output), "logo_removal": False},
        status="done",
        candidates=[{"id": "first", "pts": 1}, {"id": "second", "pts": 2}],
    )
    server.VIDEO_REGISTRY[record.id] = record
    with pytest.raises(server.HTTPException) as raised:
        server._export_job(job, ["first", "second"])
    assert raised.value.status_code == 500
    assert sorted(path.name for path in images.iterdir()) == ["frame_pts_99.png"]
    status_files = list((state_root / "exports").rglob("export-status.json"))
    assert len(status_files) == 1
    status = json.loads(status_files[0].read_text(encoding="utf-8"))
    assert status["status"] == "incomplete"
    assert status["published_files"] == []
    assert len(list(status_files[0].parent.glob("frame_*.png"))) == 1


def test_job_http_lifecycle_with_mocked_model_pipeline(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"video-source")
    export_root = tmp_path / "exports"
    seen_pts: list[int] = []

    class Reader:
        def __init__(self, _path):
            self.info = {"width": 16, "height": 12, "duration": 1.0, "fps": 24.0, "path": str(source), "time_base": "1/24"}

        @property
        def fingerprint(self):
            return server._fingerprint(source)

        def export_png(self, pts, destination):
            seen_pts.append(int(pts))
            Image.new("RGB", (16, 12), "#123456").save(destination, format="PNG")

    class Models:
        @staticmethod
        def status(_root):
            return {"ready": True, "details": "test model"}

        def __init__(self, **_kwargs):
            pass

        def vocabulary(self):
            return {"smile": "general"}

    def pipeline(**kwargs):
        result_root = Path(kwargs["cache_dir"])
        result_root.mkdir(parents=True, exist_ok=True)
        preview = result_root / "preview.png"
        crop = result_root / "crop.png"
        Image.new("RGB", (16, 12), "#111111").save(preview, format="PNG")
        Image.new("RGB", (8, 8), "#222222").save(crop, format="PNG")
        kwargs["progress"](1.0, "done")
        return {"candidates": [{"id": "mock-1", "pts": 9, "time": 0.375, "bbox": [0, 0, 1, 1], "score": 0.9, "preview_path": str(preview), "crop_path": str(crop)}], "warnings": [], "stats": {"mock": True}}

    monkeypatch.setattr(server, "VideoReader", Reader)
    monkeypatch.setattr(server, "ModelService", Models)
    monkeypatch.setattr(server, "run_search", pipeline)
    monkeypatch.setattr(server, "_MODEL_INSTANCE", None)
    monkeypatch.setattr(server, "_MODEL_MANIFEST_ROOT", None)
    monkeypatch.setattr(server, "_MODEL_MANIFEST_VOCABULARY", None)
    with _client() as client:
        inspected = client.post("/api/videos/inspect", json={"path": str(source)})
        assert inspected.status_code == 200
        video_id = inspected.json()["video_id"]
        image = io.BytesIO()
        Image.new("RGB", (16, 12), "#ff00aa").save(image, format="PNG")
        image.seek(0)
        reference = client.post("/api/references", files={"files": ("ref.png", image, "image/png")}).json()["references"][0]
        created = client.post(
            "/api/jobs",
            json={
                "video_id": video_id,
                "references": [{"id": reference["id"], "box": [0, 0, 1, 1]}],
                "wanted": [{"tag": "smile", "scope": "frame"}],
                "avoided": [],
                "settings": {"candidate_count": 1, "output_root": str(export_root)},
            },
        )
        assert created.status_code == 200
        job_id = created.json()["job_id"]
        for _ in range(100):
            state = client.get(f"/api/jobs/{job_id}").json()
            if state["status"] in {"done", "error", "cancelled"}:
                break
            time.sleep(0.01)
        assert state["status"] == "done", state
        assert state["candidates"][0]["preview_url"].endswith("/preview")
        exported = client.post(f"/api/jobs/{job_id}/export", json={"candidate_ids": ["mock-1"]})
    assert exported.status_code == 200
    assert seen_pts == [9]


def test_automatic_job_needs_no_reference_and_exports_to_video_images(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"automatic-mode-video")
    export_root = tmp_path / "outputs"
    private_cache = tmp_path / "private-cache"
    monkeypatch.setattr(server, "CACHE_ROOT", private_cache)
    monkeypatch.setattr(server, "RESULT_ROOT", private_cache / "results")
    seen: dict[str, object] = {}
    seen_pts: list[int] = []

    class Reader:
        def __init__(self, _path):
            self.info = {"width": 16, "height": 12, "duration": 1.0, "fps": 24.0, "path": str(source), "time_base": "1/24"}

        @property
        def fingerprint(self):
            return server._fingerprint(source)

        def export_png(self, pts, destination):
            seen_pts.append(int(pts))
            Image.new("RGB", (16, 12), "#123456").save(destination, format="PNG")

    class Models:
        @staticmethod
        def status(_root):
            return {"ready": True, "details": "test GPU model"}

        def __init__(self, **kwargs):
            assert kwargs.get("require_gpu") is True
            self.revision = "fixture-model"

    def pipeline(**kwargs):
        seen["references"] = kwargs["references"]
        seen["mode"] = kwargs["settings"]["selection_mode"]
        result_root = Path(kwargs["cache_dir"])
        result_root.mkdir(parents=True, exist_ok=True)
        preview = result_root / "candidate-preview.jpg"
        crop = result_root / "candidate-crop.jpg"
        Image.new("RGB", (16, 12), "#f0b0c0").save(preview, format="JPEG")
        Image.new("RGB", (16, 12), "#f0b0c0").save(crop, format="JPEG")
        return {
            "candidates": [
                {
                    "id": "auto-pts-9",
                    "pts": 9,
                    "time": 0.375,
                    "score": 0.91,
                    "selection_mode": "automatic_character",
                    "character_presence": True,
                    "cut_quality": 0.91,
                    "preview_path": str(preview),
                    "crop_path": str(crop),
                }
            ],
            "warnings": [],
            "stats": {"selection_mode": "automatic_character", "identity_model_used": False},
        }

    monkeypatch.setattr(server, "VideoReader", Reader)
    monkeypatch.setattr(server, "ModelService", Models)
    monkeypatch.setattr(server, "run_search", pipeline)
    monkeypatch.setattr(server, "_MODEL_INSTANCE", None)
    monkeypatch.setattr(server, "_MODEL_MANIFEST_ROOT", None)
    monkeypatch.setattr(server, "_MODEL_MANIFEST_VOCABULARY", None)
    with _client() as client:
        inspected = client.post("/api/videos/inspect", json={"path": str(source)})
        assert inspected.status_code == 200
        created = client.post(
            "/api/jobs",
            json={
                "video_id": inspected.json()["video_id"],
                "selection_mode": "automatic",
                "references": [],
                "wanted": [],
                "avoided": [],
                "settings": {"coarse_step": 1.0, "output_root": str(export_root)},
            },
        )
        assert created.status_code == 200
        job_id = created.json()["job_id"]
        for _ in range(100):
            state = client.get(f"/api/jobs/{job_id}").json()
            if state["status"] in {"done", "error", "cancelled"}:
                break
            time.sleep(0.01)
        assert state["status"] == "done", state
        assert state["candidates"][0]["character_presence"] is True
        exported = client.post(f"/api/jobs/{job_id}/export", json={"candidate_ids": ["auto-pts-9"]})
    assert seen == {"references": [], "mode": "automatic"}
    assert exported.status_code == 200
    assert seen_pts == [9]
    public_images = Path(exported.json()["directory"])
    assert public_images == export_root / "episode" / "images"
    assert (public_images / "frame_pts_9.png").is_file()


def test_model_service_is_never_prepared_by_http_worker(tmp_path, monkeypatch):
    calls: list[str] = []

    class OfflineReadyService:
        @staticmethod
        def status(_root):
            return {"ready": True, "details": "prepared"}

        def __init__(self, **kwargs):
            calls.append(f"init:{kwargs.get('require_gpu')}")

        def prepare(self):
            calls.append("prepare")
            raise AssertionError("HTTP jobs must not prepare or download models")

    monkeypatch.setattr(server, "ModelService", OfflineReadyService)
    monkeypatch.setattr(server, "_MODEL_INSTANCE", None)
    service = server._load_model_service(tmp_path)
    assert isinstance(service, OfflineReadyService)
    assert calls == ["init:True"]


def test_logo_profiles_endpoint_exposes_validated_public_metadata(monkeypatch, tmp_path):
    class FakeProcessor:
        @staticmethod
        def status(root):
            return {"ready": True, "details": "fake-ready", "root": str(root)}

    class FakeStore:
        def __init__(self, root):
            assert Path(root) == tmp_path

        def list(self):
            return [{"id": "muse", "name": "MUSE", "reference_size": [3840, 2160]}]

    monkeypatch.setattr(server, "LogoProcessor", FakeProcessor)
    monkeypatch.setattr(server, "ProfileStore", FakeStore)
    monkeypatch.setenv("ANIME_PICKER_LOGO_PROFILE_ROOT", str(tmp_path))
    with _client() as client:
        response = client.get("/api/logo-profiles")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_enabled"] is True
    assert payload["profiles"] == [
        {"id": "muse", "name": "MUSE", "reference_size": [3840, 2160], "ready": True}
    ]
    assert payload["model"]["ready"] is True


def test_logo_off_export_is_exact_and_does_not_import_logo_runtime(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"video-source")
    output = tmp_path / "exports"
    expected = Image.new("RGB", (7, 5), "#102030")
    seen: list[int] = []

    class Reader:
        def __init__(self, _path):
            pass

        def export_png(self, pts, destination):
            seen.append(int(pts))
            expected.save(destination, format="PNG")

    def fail_import():
        raise AssertionError("logo runtime must not be imported for logo-off export")

    monkeypatch.setattr(server, "VideoReader", Reader)
    monkeypatch.setattr(server, "_import_logo_components", fail_import)
    record = server.VideoRecord(
        id="1" * 32,
        path=source,
        info={"width": 7, "height": 5, "duration": 1.0, "fps": 24.0, "time_base": "1/24"},
        fingerprint=server._fingerprint(source),
        created_at="now",
    )
    job = server.JobRecord(
        id="2" * 32,
        video_id=record.id,
        reference_items=[],
        wanted=[],
        avoided=[],
        settings={"output_root": str(output), "logo_removal": False, "logo_profiles": ["ignored"]},
        status="done",
        candidates=[{"id": "candidate-1", "pts": 17}],
    )
    server.VIDEO_REGISTRY[record.id] = record
    result = server._export_job(job, ["candidate-1"])
    assert seen == [17]
    assert result["originals"] == []
    with Image.open(result["files"][0]) as image:
        assert image.size == expected.size
        assert image.tobytes() == expected.tobytes()
    assert result["metadata"]["candidates"][0]["logo_cleanup"]["status"] == "disabled"


def test_logo_profile_ids_are_validated_by_profile_store(monkeypatch, tmp_path):
    class FakeProcessor:
        @staticmethod
        def status(_root):
            return {"ready": True}

    class FakeStore:
        def __init__(self, _root):
            pass

        def load(self, ids=None):
            if ids:
                raise ValueError("profile is unknown")
            return []

    monkeypatch.setattr(server, "LogoProcessor", FakeProcessor)
    monkeypatch.setattr(server, "ProfileStore", FakeStore)
    monkeypatch.setenv("ANIME_PICKER_LOGO_PROFILE_ROOT", str(tmp_path))
    with pytest.raises(server.HTTPException) as raised:
        server._validate_logo_profile_ids(["unknown"])
    assert raised.value.status_code == 422


def test_logo_on_export_retains_original_and_reports_no_hit(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"video-source")
    output = tmp_path / "exports"
    source_image = Image.new("RGB", (8, 6), "#304050")

    class Reader:
        def __init__(self, _path):
            pass

        def export_png(self, _pts, destination):
            source_image.save(destination, format="PNG")

    class FakeStore:
        def __init__(self, _root):
            pass

        def load(self, _ids=None):
            return [{"id": "muse", "name": "MUSE"}]

    class FakeProcessor:
        @staticmethod
        def status(_root):
            return {"ready": True}

        def __init__(self, **kwargs):
            assert kwargs.get("require_gpu") is True

        def process(self, image, _profiles, force=False):
            assert force is False
            return SimpleNamespace(
                image=image.copy(),
                metadata={"status": "skipped", "reason": "no registered logo matched"},
                edit_mask=Image.new("L", image.size, 0),
            )

    monkeypatch.setattr(server, "VideoReader", Reader)
    monkeypatch.setattr(server, "LogoProcessor", FakeProcessor)
    monkeypatch.setattr(server, "ProfileStore", FakeStore)
    monkeypatch.setattr(server, "_LOGO_PROCESSOR_INSTANCE", None)
    monkeypatch.setattr(server, "_LOGO_PROCESSOR_ROOT", None)
    record = server.VideoRecord(
        id="3" * 32,
        path=source,
        info={"width": 8, "height": 6, "duration": 1.0, "fps": 24.0, "time_base": "1/24"},
        fingerprint=server._fingerprint(source),
        created_at="now",
    )
    job = server.JobRecord(
        id="4" * 32,
        video_id=record.id,
        reference_items=[],
        wanted=[],
        avoided=[],
        settings={"output_root": str(output)},
        status="done",
        candidates=[{"id": "candidate-1", "pts": 19}],
    )
    server.VIDEO_REGISTRY[record.id] = record
    result = server._export_job(job, ["candidate-1"])
    assert server.DEFAULT_SETTINGS["logo_removal"] is True
    assert len(result["originals"]) == 1
    assert Path(result["originals"][0]).parent.name == "originals"
    with Image.open(result["files"][0]) as cleaned, Image.open(result["originals"][0]) as original:
        assert cleaned.size == original.size == source_image.size
        assert cleaned.tobytes() == original.tobytes()
    frame_meta = result["metadata"]["candidates"][0]["logo_cleanup"]
    assert frame_meta["requested"] is True
    assert frame_meta["applied"] is False
    assert frame_meta["status"] == "skipped"
    assert frame_meta["original_relative_path"] == "originals/frame_0001.png"
    assert result["warnings"]


def test_logo_preview_is_async_registered_and_applies_only_edit_mask(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"video-source")
    result_root = tmp_path / "job-result"
    result_root.mkdir()
    original_preview = result_root / "candidate-preview.jpg"
    Image.new("RGB", (8, 6), "#203040").save(original_preview, format="JPEG")
    source_image = Image.new("RGB", (8, 6), "#203040")
    profile_dir = tmp_path / "profiles" / "muse"
    profile_dir.mkdir(parents=True)
    profile_json = profile_dir / "profile.json"
    profile_json.write_text('{"version": 1, "id": "muse", "revision": 1}', encoding="utf-8")
    template_path = profile_dir / "template.png"
    Image.new("RGB", (2, 2), "#456789").save(template_path, format="PNG")

    class Reader:
        def __init__(self, _path):
            pass

        def export_png(self, _pts, destination):
            source_image.save(destination, format="PNG")

    class FakeStore:
        def __init__(self, _root):
            pass

        def load(self, _ids=None):
            return [
                {
                    "id": "muse",
                    "name": "MUSE",
                    "_profile_dir": str(profile_dir),
                    "_template_path": str(template_path),
                }
            ]

    class FakeProcessor:
        calls = 0

        @staticmethod
        def status(_root):
            return {"ready": True}

        def __init__(self, **_kwargs):
            pass

        def process(self, image, _profiles, force=False):
            assert force is False
            FakeProcessor.calls += 1
            cleaned = image.copy()
            cleaned.putpixel((0, 0), (255, 0, 0))
            mask = Image.new("L", image.size, 0)
            mask.putpixel((0, 0), 255)
            return SimpleNamespace(
                image=cleaned,
                metadata={
                    "status": "processed",
                    "profile_hits": [{"id": "muse"}],
                    "model_sha256": "fake-hash",
                    "dtype": "float32",
                    "bounds": [[0, 0, 1, 1]],
                },
                edit_mask=mask,
            )

    monkeypatch.setattr(server, "VideoReader", Reader)
    monkeypatch.setattr(server, "LogoProcessor", FakeProcessor)
    monkeypatch.setattr(server, "ProfileStore", FakeStore)
    monkeypatch.setattr(server, "_LOGO_PROCESSOR_INSTANCE", None)
    monkeypatch.setattr(server, "_LOGO_PROCESSOR_ROOT", None)
    record = server.VideoRecord(
        id="5" * 32,
        path=source,
        info={"width": 8, "height": 6, "duration": 1.0, "fps": 24.0, "time_base": "1/24"},
        fingerprint=server._fingerprint(source),
        created_at="now",
    )
    job = server.JobRecord(
        id="6" * 32,
        video_id=record.id,
        reference_items=[],
        wanted=[],
        avoided=[],
        settings={"logo_removal": True, "logo_profiles": ["muse"]},
        status="done",
        result_root=result_root,
        candidates=[{"id": "candidate-1", "pts": 23, "_preview_file": str(original_preview)}],
    )
    server.VIDEO_REGISTRY[record.id] = record
    server.JOB_REGISTRY[job.id] = job
    with _client() as client:
        response = client.post(
            f"/api/jobs/{job.id}/candidates/candidate-1/logo-preview",
            json={"logo_removal": True, "logo_profiles": ["muse"]},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["metadata"]["logo_cleanup"]["applied"] is True
        assert payload["metadata"]["logo_cleanup"]["model_hash"] == "fake-hash"
        assert payload["warnings"] == []
        assert payload["preview_url"].endswith("/logo-preview")
        preview = client.get(payload["preview_url"])
        first_signature = payload["metadata"]["profile_signature"]
        profile_json.write_text('{"version": 1, "id": "muse", "revision": 2}', encoding="utf-8")
        refreshed = client.post(
            f"/api/jobs/{job.id}/candidates/candidate-1/logo-preview",
            json={"logo_removal": True, "logo_profiles": ["muse"]},
        )
        assert refreshed.status_code == 200, refreshed.text
        assert refreshed.json()["metadata"]["profile_signature"] != first_signature
        assert FakeProcessor.calls == 2
    assert preview.status_code == 200
    with Image.open(io.BytesIO(preview.content)) as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)
        assert image.getpixel((7, 5)) == source_image.getpixel((7, 5))
