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
    assert Path(payload["directory"]).is_dir()
    metadata = json.loads(Path(payload["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["source_pts"] == [37]
    assert metadata["candidates"][0]["crop"] == [0.1, 0.2, 0.8, 0.9]


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
    assert directory.parent == default_root.resolve()
    assert (directory / "frame_0001.png").is_file()


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
