from __future__ import annotations

import array
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anime_picker import accepted_catalogue as ac, server
from scripts.build_accepted_catalogue import full_tags, join_records


def seal(root, release="fixture"):
    directory = root / "_accepted" / release
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["index_sha256"] = ac.digest(directory / "index.sqlite")
    (directory / "manifest.json").write_text(ac.canonical(manifest), encoding="utf-8")
    (root / "_accepted/current.json").write_text(ac.canonical({"schema": 1, "release": release,
        "manifest_sha256": ac.digest(directory / "manifest.json")}), encoding="utf-8")


@pytest.fixture
def catalogue(tmp_path):
    root = tmp_path / "outputs"
    release = root / "_accepted/fixture"
    release.mkdir(parents=True)
    image = root / "old/native.png"
    image.parent.mkdir()
    Image.new("RGB", (12, 8), "red").save(image)
    db = sqlite3.connect(release / "index.sqlite")
    ac.create_schema(db)
    values = [("01", 10, "raw:é"), ("01", 20, "raw:é"), ("02", 10, "raw:é"), ("02", 30, "last")]
    ids = []
    for seq, (row, pts, raw_id) in enumerate(values):
        identity = ac.stable_id(row, raw_id, pts, "fingerprint")
        ids.append(identity)
        metadata = {"source_name": '<script>悪意.png', "time": pts / 10, "native_origin": "existing_native",
            "tag_evidence": {"status": "complete" if seq != 3 else "missing"}, "accepted_record": {"cut_quality": .8}}
        db.execute("INSERT INTO frames VALUES(?,?,?,?,?,?,?,?,?,?,?)", (seq, identity, row, int(row), pts, raw_id,
            "old/native.png", ac.digest(image), 12, 8, ac.canonical(metadata)))
    tags = [(1, "character", "初音ミク", [.9, .08, .6, math.nan]),
            (2, "character", "初音ミク_(衣装)", [.8, .07, .5, math.nan]),
            (3, "general", "100%_literal!", [.5, .5, .5, math.nan]),
            (4, "character", "Straße", [.2, .2, .2, math.nan])]
    for tag_id, cat, name, scores in tags:
        db.execute("INSERT INTO vocabulary VALUES(?,?,?,?)", (tag_id, cat, name, ac.normalized(name)))
        for row, base in [("01", 0), ("02", 2)]:
            db.execute("INSERT INTO scores VALUES(?,?,?,?,?)", (tag_id, row, base, 2,
                ac.pack_scores(array.array("d", scores[base:base + 2]))))
    db.commit()
    db.close()
    (release / "manifest.json").write_text(ac.canonical({"schema": 1, "release": "fixture", "count": 4,
        "rows": [{"key": row, "order": int(row), "name": row, "count": 2} for row in ("01", "02")],
        "categories": ["character", "general"], "full_search_complete": False, "missing_tags": 1,
        "provenance_mode": "MIXED_SOURCE_ACCEPTED_REUSE", "suggestions": []}), encoding="utf-8")
    seal(root)
    return root, ids


def test_exact_join_and_raw_identity():
    records = [{"id": "é", "pts": 4, "image": "a", "detections": []},
               {"id": "é", "pts": 4, "image": "b", "detections": []}]
    mapping = [dict(r, new_kept=True) for r in records]
    assert len(list(join_records(records, {"kept": records[:1], "excluded": records[1:]}, mapping))) == 2
    assert ac.stable_id("01", "é", 4, "fp") != ac.stable_id("01", "é", 4, "fp")
    assert ac.stable_id("01", "é", 4, "fp") != ac.stable_id("02", "é", 4, "fp")
    with pytest.raises(ValueError, match="duplicate"):
        list(join_records(records, {"kept": records, "excluded": records[:1]}, mapping))
    with pytest.raises(ValueError, match="image/detections"):
        list(join_records([dict(records[0], image="changed")], {"kept": records, "excluded": []}, mapping))
    with pytest.raises(ValueError, match="noninteger"):
        list(join_records([dict(records[0], pts=4.0)], {"kept": records, "excluded": []}, mapping))


def test_full_tag_cache_exact_provenance_and_missing(tmp_path):
    fingerprint = "12:34:" + "a" * 64
    detector = tmp_path / ("a" * 12) / "detections/rev"
    detector.mkdir(parents=True)
    image = tmp_path / "input.png"
    Image.new("RGBA", (4, 3), (0, 40, 80, 255)).save(image)
    record = {"id": "pts:5", "pts": 5, "image": str(image), "detections": []}
    detector_record = {"source_fingerprint": fingerprint, "pts": 5, "image_sha256": ac.digest(image),
                       "detections": [], "detector_revision": "revision"}
    (detector / "pts_5.json").write_text(ac.canonical(detector_record))
    old = {"detector_backend": {"cache_directory": str(detector), "cache_revision": "revision"}}
    revision = "tagger:unchanged"
    import numpy as np
    with Image.open(image) as im:
        # Reproduce the existing batch_source cache recipe independently.
        pixels = np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
        key = hashlib.sha256(pixels.tobytes()).hexdigest()
    cache = detector.parent.parent / "tags" / hashlib.sha256(revision.encode()).hexdigest()[:16] / (key + ".json")
    cache.parent.mkdir(parents=True)
    tags = {"character": {"初音ミク": .00000001}, "general": {"smile": .123456789012345}}
    cache.write_text(ac.canonical(tags), encoding="utf-8")
    result, evidence = full_tags(record, old, fingerprint, revision)
    assert result == tags and evidence["status"] == "complete" and evidence["rgb_key"] == key
    assert evidence["cache_sha256"] == ac.digest(cache)
    cache.write_text("{}")
    assert full_tags(record, old, fingerprint, revision)[1]["status"] == "missing"
    cache.unlink()
    assert full_tags(record, old, fingerprint, revision)[1]["status"] == "missing"
    detector_record["source_fingerprint"] = "wrong"
    (detector / "pts_5.json").write_text(ac.canonical(detector_record))
    with pytest.raises(ValueError, match="provenance"):
        full_tags(record, old, fingerprint, revision)


def test_pages_unicode_literal_search_thresholds_and_cursors(catalogue):
    root, ids = catalogue
    store = ac.AcceptedCatalogue(root)
    assert store.summary()["count"] == 4
    with store._connect() as db:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("DELETE FROM frames")
    first = store.frames(limit=2)
    second = store.frames(limit=2, cursor=first["next_cursor"])
    assert [r["stable_id"] for r in first["items"] + second["items"]] == ids
    assert first["total"] == 4 and second["next_cursor"] is None
    assert store.frames(q="初音ミク")["total"] == 2  # two matching tags still one frame
    assert store.frames(q="初音ミク", min_score=.05)["total"] == 3
    assert store.frames(q="初音ミク", min_score=0)["total"] == 3
    assert store.frames(q="初音ミク", min_score=1)["total"] == 0
    assert store.frames(q="ＳＴＲＡＳＳＥ")["total"] == 3
    assert store.frames(q="100%_literal!")["total"] == 3
    assert store.frames(q="%_notwildcard")["total"] == 0
    assert store.frames(q="' OR 1=1 --")["total"] == 0
    assert store.frames(q="初音", row="02")["total"] == 1
    assert store.frames(q="初音", tag_category="general")["total"] == 0
    assert store.frames(row="02", min_score=1, tag_category="character")["total"] == 2
    for change in [dict(q="different"), dict(min_score=.01), dict(row="02"), dict(limit=3), dict(tag_category="general")]:
        with pytest.raises(ac.CatalogueError) as exc:
            store.frames(cursor=first["next_cursor"], **(dict(limit=2) | change))
        assert exc.value.status == 400
    for cursor in ["bad!", "a" * 2049, "e30=", "初音"]:
        with pytest.raises(ac.CatalogueError):
            store.frames(cursor=cursor)
    extra = json.loads(base64.urlsafe_b64decode(first["next_cursor"])) | {"unexpected": True}
    with pytest.raises(ac.CatalogueError) as malformed:
        store.frames(limit=2, cursor=base64.urlsafe_b64encode(ac.canonical(extra).encode()).decode())
    assert malformed.value.status == 400
    for params in [dict(limit=0), dict(limit=101), dict(limit=1.1), dict(q="x" * 257),
                   dict(min_score=float("nan")), dict(min_score=float("inf")), dict(min_score=-.1), dict(min_score=True)]:
        with pytest.raises(ac.CatalogueError) as exc:
            store.frames(**params)
        assert exc.value.status == 422
    with pytest.raises(ac.CatalogueError, match="unknown catalogue row"):
        store.frames(row="unknown")
    manifest_path = root / "_accepted/fixture/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["suggestions"] = [{"tag": "changed immutable release", "category": "character", "count": 1}]
    manifest_path.write_text(ac.canonical(manifest), encoding="utf-8")
    seal(root)
    with pytest.raises(ac.CatalogueError) as stale:
        store.frames(limit=2, cursor=first["next_cursor"])
    assert stale.value.status == 400


def test_missing_and_corruption_fail_closed(tmp_path, catalogue):
    assert ac.AcceptedCatalogue(tmp_path).summary()["count"] == 0
    assert ac.AcceptedCatalogue(tmp_path).frames()["total"] == 0
    root, ids = catalogue
    store = ac.AcceptedCatalogue(root)
    assert ac.png_dimensions(store.image(ids[0])) == (12, 8)
    index = root / "_accepted/fixture/index.sqlite"
    with index.open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ac.CatalogueError, match="drift"):
        store.frames()
    with pytest.raises(ac.CatalogueError, match="hash mismatch"):
        ac.AcceptedCatalogue(root).summary()
    seal(root)
    pointer = root / "_accepted/current.json"
    pointer.write_text('{"schema":1,"release":"../escape","manifest_sha256":"bad"}')
    with pytest.raises(ac.CatalogueError):
        store.summary()
    pointer.write_text("not json")
    with pytest.raises(ac.CatalogueError):
        store.summary()


@pytest.mark.parametrize("field,value", [("rows", None), ("count", "4"), ("categories", 12),
                                       ("suggestions", [{"tag": "invalid"}]), ("missing_tags", -1)])
def test_corrupt_manifest_schema_is_explicit_503(catalogue, monkeypatch, field, value):
    root, _ = catalogue
    path = root / "_accepted/fixture/manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(ac.canonical(manifest), encoding="utf-8")
    seal(root)
    monkeypatch.setattr(server, "DEFAULT_OUTPUT_ROOT", root)
    with TestClient(server.create_app()) as client:
        assert client.get("/api/accepted-catalogue").status_code == 503


@pytest.mark.parametrize("relative", ["../escape.png", "/absolute.png", "C:/drive.png", "C:relative.png",
    "//server/share.png", "old\\native.png", "old/%2fescape.png", "old/../old/native.png", "old//native.png",
    "old/.. /old/native.png", "old./native.png"])
def test_malicious_manifest_image_paths(catalogue, relative):
    root, _ = catalogue
    with sqlite3.connect(root / "_accepted/fixture/index.sqlite") as db:
        db.execute("UPDATE frames SET image=? WHERE seq=0", (relative,))
    seal(root)
    with pytest.raises(ac.CatalogueError):
        ac.AcceptedCatalogue(root).summary()


def test_symlink_and_reparse_rejected(catalogue, tmp_path, monkeypatch):
    root, _ = catalogue
    original = Path.lstat
    target = root / "old"
    from types import SimpleNamespace
    def reparse(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if path == target:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info
    monkeypatch.setattr(Path, "lstat", reparse)
    with pytest.raises(ac.CatalogueError, match="reparse"):
        ac.safe_file(root, "old/native.png", ".png")
    monkeypatch.setattr(Path, "lstat", original)
    link = root / "link"
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        return  # Reparse guard was exercised independently above on Windows.
    with pytest.raises(ac.CatalogueError, match="links/reparse"):
        ac.safe_file(root, "link/anything.png", ".png")


def test_bulk_validation_checks_each_directory_once_and_every_file(catalogue, monkeypatch):
    root, _ = catalogue
    (root / "old/second.png").write_bytes((root / "old/native.png").read_bytes())
    original = Path.lstat
    seen = []
    def spy(self, *args, **kwargs):
        seen.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", spy)
    verified = set()
    for name in ("native.png", "second.png", "native.png"):
        ac.safe_file(root, f"old/{name}", ".png", verified)
    assert seen.count(root) == 1 and seen.count(root / "old") == 1
    assert seen.count(root / "old/native.png") == 2 and seen.count(root / "old/second.png") == 1
    seen.clear()
    ac.safe_file(root, "old/native.png", ".png")
    ac.safe_file(root, "old/native.png", ".png")
    assert seen.count(root / "old") == 2  # per-request validation never reuses a prior check
    seen.clear()
    ac.AcceptedCatalogue(root).summary()  # four frames share one image path
    assert seen.count(root / "old") == 1 and seen.count(root / "old/native.png") == 4


def test_bulk_validation_still_rejects_reparse_file_and_directory(catalogue, monkeypatch):
    root, _ = catalogue
    (root / "old/second.png").write_bytes((root / "old/native.png").read_bytes())
    original = Path.lstat
    flagged = set()
    from types import SimpleNamespace
    def reparse(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if path in flagged:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info
    monkeypatch.setattr(Path, "lstat", reparse)
    verified = set()
    ac.safe_file(root, "old/native.png", ".png", verified)
    flagged.add(root / "old/second.png")  # sibling file in an already-verified directory
    with pytest.raises(ac.CatalogueError, match="reparse"):
        ac.safe_file(root, "old/second.png", ".png", verified)
    flagged.clear()
    flagged.add(root / "old")
    with pytest.raises(ac.CatalogueError, match="reparse"):
        ac.safe_file(root, "old/native.png", ".png", set())
    with pytest.raises(ac.CatalogueError, match="reparse"):
        ac.safe_file(root, "old/native.png", ".png")
    with pytest.raises(ac.CatalogueError, match="reparse"):
        ac.AcceptedCatalogue(root).summary()


def test_prewarm_verifies_release_once_and_defers_errors_to_requests(catalogue, monkeypatch):
    root, _ = catalogue
    calls = []
    real_digest = ac.digest
    monkeypatch.setattr(ac, "digest", lambda path: (calls.append(path.name), real_digest(path))[1])
    store = ac.AcceptedCatalogue(root)
    assert store.prewarm() is True and calls == ["index.sqlite"]
    assert store.summary()["count"] == 4 and store.frames()["total"] == 4
    assert calls == ["index.sqlite"]  # requests reuse the verified pin
    assert ac.AcceptedCatalogue(root.parent / "nowhere").prewarm() is False
    with (root / "_accepted/fixture/index.sqlite").open("ab") as stream:
        stream.write(b"drift")
    assert ac.AcceptedCatalogue(root).prewarm() is False  # no exception from the warm-up thread
    with pytest.raises(ac.CatalogueError, match="hash mismatch"):
        ac.AcceptedCatalogue(root).summary()


def test_serve_prewarms_catalogue_in_background(monkeypatch):
    import threading
    import uvicorn
    from anime_picker import __main__ as entry
    started = threading.Event()
    monkeypatch.setattr(ac.AcceptedCatalogue, "prewarm", lambda self: started.set())
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    assert entry.main(["serve", "--no-browser"]) == 0
    assert started.wait(5)
    assert isinstance(server.app.state.accepted_catalogue, ac.AcceptedCatalogue)


def forbid_inference(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("accepted search must never run inference/curation/video/export")
    for name in ("_import_model_service", "_load_model_service", "_suggest_character_frames", "_export_candidate_png", "_import_video_reader"):
        if hasattr(server, name):
            monkeypatch.setattr(server, name, fail)
    monkeypatch.setattr(server, "VideoReader", fail)
    monkeypatch.setattr(server, "ModelService", fail)
    import anime_picker.curation as curation
    import anime_picker.video as video
    for name in ("curate_character_frames", "curate_candidates", "curate", "calculate_curation"):
        if hasattr(curation, name):
            monkeypatch.setattr(curation, name, fail)
    monkeypatch.setattr(video.VideoReader, "export_pngs", fail)
    monkeypatch.setattr(video.VideoReader, "export_png", fail)


def test_api_read_only_images_validation_and_no_model(catalogue, monkeypatch):
    root, ids = catalogue
    forbid_inference(monkeypatch)
    monkeypatch.setattr(server, "DEFAULT_OUTPUT_ROOT", root)
    with TestClient(server.create_app()) as client:
        assert client.get("/api/accepted-catalogue").json()["count"] == 4
        assert client.get("/api/accepted-catalogue/frames", params={"q": "初音ミク"}).json()["total"] == 2
        image = client.get(f"/api/accepted-catalogue/frames/{ids[0]}/image")
        assert image.content == (root / "old/native.png").read_bytes()
        assert image.headers["content-type"] == "image/png" and image.headers["x-content-type-options"] == "nosniff"
        assert client.get("/api/accepted-catalogue/frames/" + "0" * 64 + "/image").status_code == 404
        assert client.get("/api/accepted-catalogue/frames/bad/image").status_code == 404
        assert client.get("/api/accepted-catalogue/frames", params={"min_score": "NaN"}).status_code == 422
        assert client.get("/api/accepted-catalogue/frames", params={"limit": 101}).status_code == 422
        assert client.get("/api/accepted-catalogue/frames", params={"row": "unknown"}).status_code == 404
        assert client.post("/api/accepted-catalogue/frames").status_code == 405
        (root / "old/native.png").write_bytes(b"bad")
        assert client.get(f"/api/accepted-catalogue/frames/{ids[0]}/image").status_code == 503
        (root / "_accepted/current.json").unlink()
        assert client.get("/api/accepted-catalogue").json()["available"] is False
        assert client.get("/api/accepted-catalogue/frames", params={"q": "初音"}).json()["total"] == 0


def test_module_has_no_inference_imports():
    code = "import sys; import anime_picker.accepted_catalogue; assert not any(k in sys.modules for k in ['torch','anime_picker.models','anime_picker.video','anime_picker.curation','scripts.batch_source'])"
    subprocess.run([sys.executable, "-B", "-c", code], check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def test_ui_contract_uses_text_and_separate_get_view():
    static = Path(__file__).parents[1] / "anime_picker/static"
    page = (static / "index.html").read_text(encoding="utf-8")
    js = (static / "app.js").read_text(encoding="utf-8").split("function bindAcceptedCatalogue()", 1)[1].split("function bindEvents()", 1)[0]
    assert 'id="acceptedEntry"' in page and 'value="0.1" required' in page
    assert "no identity inference" in page
    assert "innerHTML" not in js and "textContent" in js and "image.loading = 'lazy'" in js
    assert "character-suggestions" not in js and "/api/jobs" not in js
    assert "method: 'GET'" in js and "history.pop()" in js and "min_score:" in js
    assert "整合性を確認中" in js  # the first verification shows progress instead of a silent wait


@pytest.mark.skipif(not os.environ.get("ANIME_PICKER_CATALOGUE_EVIDENCE"), reason="explicit private release integration only")
def test_real_pinned_catalogue_integration(monkeypatch):
    evidence = Path(os.environ["ANIME_PICKER_CATALOGUE_EVIDENCE"])
    build = json.loads((evidence / "build-result.json").read_text())
    manifest_path = Path(build["manifest"])
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parents[2]
    store = ac.AcceptedCatalogue(root, trusted_release=manifest["release"], manifest_sha256=build["manifest_sha256"])
    forbid_inference(monkeypatch)
    monkeypatch.setattr(server, "DEFAULT_OUTPUT_ROOT", root)
    monkeypatch.setattr(server, "AcceptedCatalogue", lambda root: store)
    expected, excluded, additions, counts = set(), set(), set(), {}
    for row in manifest["rows"]:
        receipt = json.loads((evidence / f'row-{row["key"]}.json').read_text())
        assert ac.digest(Path(receipt["report"]["path"])) == receipt["report"]["sha256"]
        assert ac.digest(Path(receipt["batch"]["path"])) == receipt["batch"]["sha256"]
        wrapper = json.loads(Path(receipt["report"]["path"]).read_text())
        report = wrapper["report"]
        batch = json.loads(Path(receipt["batch"]["path"]).read_text())
        fp = batch["fingerprint"]
        assert fp == receipt["source_fingerprint"]
        counts[row["key"]] = len(report["kept"])
        row_accepted = {ac.stable_id(row["key"], f["id"], f["pts"], fp) for f in report["kept"]}
        expected.update(row_accepted)
        old = json.loads(Path(batch["analysis"]["curation"]["report"]).read_text())
        old_kept = {ac.stable_id(row["key"], f["id"], f["pts"], fp) for f in old["report"]["kept"]}
        row_excluded = old_kept - row_accepted
        assert row_excluded == {ac.stable_id(row["key"], i, pts, fp) for i, pts in receipt["excluded_old_kept"]}
        excluded.update(row_excluded)
        additions.update(ac.stable_id(r, i, pts, fp) for r, i, pts in receipt["overlay_identities"])
    overlay_path = next(Path(p) for p in manifest["accepted_pins"] if Path(p).name == "accepted-overlay-manifest.json")
    assert ac.digest(overlay_path) == manifest["accepted_pins"][str(overlay_path)]
    overlay = json.loads(overlay_path.read_text())
    expected_additions = {ac.stable_id(a["row_key"], a["id"], a["pts"], a["fingerprint"]) for a in overlay["additions"]}
    assert additions == expected_additions and len(additions) == 23
    with TestClient(server.create_app()) as client:
        summary = client.get("/api/accepted-catalogue").json()
        assert summary["count"] == 23792 and summary["full_search_complete"]
        actual, cursor, previous = [], "", None
        while True:
            response = client.get("/api/accepted-catalogue/frames", params={"limit": 100, "cursor": cursor})
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 23792
            for frame in data["items"]:
                order = (23 if frame["row_key"] == "accepted_ep1" else int(frame["row_key"]), frame["pts"], frame["raw_id"])
                assert previous is None or order > previous
                previous = order
                actual.append(frame["stable_id"])
            cursor = data["next_cursor"]
            if not cursor:
                break
        assert len(actual) == len(set(actual)) == len(expected) == 23792 and set(actual) == expected
        assert len(excluded) == 178 and not (set(actual) & excluded)
        assert len(additions) == 23 and additions <= set(actual)
        for row, count in counts.items():
            assert client.get("/api/accepted-catalogue/frames", params={"row": row}).json()["total"] == count
        tag = summary["suggestions"][0]["tag"]
        unicode_query = "".join(chr(ord(c) + 0xFEE0) if 'a' <= c <= 'z' else c for c in tag)
        search = client.get("/api/accepted-catalogue/frames", params={"q": unicode_query, "tag_category": "character"}).json()
        assert search["total"] > 0
        searched_id = search["items"][0]["stable_id"]
        with sqlite3.connect((manifest_path.parent / "index.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True) as db:
            for addition in overlay["additions"]:
                identity = ac.stable_id(addition["row_key"], addition["id"], addition["pts"], addition["fingerprint"])
                image_path, sha, width, height, record_json = db.execute(
                    "SELECT image,sha256,width,height,metadata FROM frames WHERE stable_id=?", (identity,)).fetchone()
                assert image_path == f"_accepted/{manifest['release']}/native/row-{addition['row_key']}/pts-{addition['pts']}.png"
                assert (sha, width, height) == (addition["sha256"], addition["width"], addition["height"])
                assert json.loads(record_json)["overlay_provenance"] == addition
                assert ac.digest(root / image_path) == sha and (root / image_path).stat().st_nlink == 1
            metadata_ = json.loads(db.execute("SELECT metadata FROM frames WHERE stable_id=?", (searched_id,)).fetchone()[0])
            tag_evidence = metadata_["tag_evidence"]
            cache_path = Path(tag_evidence["cache_path"])
            assert ac.digest(cache_path) == tag_evidence["cache_sha256"]
            cached_character = json.loads(cache_path.read_text(encoding="utf-8"))["character"]
            matched_character_tags = {name: score for name, score in cached_character.items()
                                      if ac.normalized(unicode_query) in ac.normalized(name) and score >= .1}
            assert matched_character_tags
            sample_ids = [next(i for i in actual if i not in additions), sorted(additions)[0]]
            image_evidence = []
            for identity in sample_ids:
                path, sha = db.execute("SELECT image,sha256 FROM frames WHERE stable_id=?", (identity,)).fetchone()
                image = client.get(f"/api/accepted-catalogue/frames/{identity}/image")
                assert image.status_code == 200 and hashlib.sha256(image.content).hexdigest() == sha
                assert image.content == (root / path).read_bytes()
                image_evidence.append({"id": identity, "sha256": sha, "bytes": len(image.content)})
        assert client.get("/api/accepted-catalogue/frames", params={"q": "__definitely_no_tag_7e05e28d__"}).json()["total"] == 0
    result = {"count": len(actual), "unique_ids": len(set(actual)), "all_ids_sha256": hashlib.sha256(ac.canonical(actual).encode()).hexdigest(),
        "per_row_counts": counts, "excluded_old_kept": len(excluded), "overlay_additions": len(additions),
        "unicode_query": unicode_query, "original_cache_tag": tag, "search_total": search["total"],
        "matching_cache_scores": matched_character_tags,
        "search_cache_evidence": tag_evidence, "image_responses": image_evidence, "model_video_curation_calls": 0}
    result_name = os.environ.get("ANIME_PICKER_CATALOGUE_RESULT_NAME", "integration-result.json")
    assert Path(result_name).name == result_name
    (evidence / result_name).write_text(ac.canonical(result), encoding="utf-8")
