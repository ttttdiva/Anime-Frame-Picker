"""Acceptance tests for the default "choose characters, then extract" flow.

GPU-free: detector/CCIP are deterministic CPU fakes (tests/char_select_fakes.py)
and every run manifest, cache, and output root is a per-test temp directory.
"""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anime_picker import pipeline, run_manifest, server
from anime_picker.character_candidates import CharacterCandidateIndex, detection_key
from char_select_fakes import BLUE, RED, RED_ALT_OUTFIT, FakeModels, FrameListReader, figure_frame


def _vector(colour):
    return np.asarray(colour, dtype=np.float32) / 255.0


def _distance(left, right):
    return np.linalg.norm(np.asarray(left)[:, None, :] - np.asarray(right)[None, :, :], axis=2)


# --------------------------------------------------------------------------
# Candidate characters: grouping, representatives, stable keys
# --------------------------------------------------------------------------

def test_candidate_groups_merge_outfit_changes_and_keep_different_characters_apart():
    index = CharacterCandidateIndex("fp", threshold=0.2)
    records = [
        {"pts": 10, "time": 0.4, "box": [0.1, 0.1, 0.3, 0.9], "score": 0.9, "kind": "person"},
        {"pts": 20, "time": 0.8, "box": [0.5, 0.1, 0.7, 0.9], "score": 0.9, "kind": "person"},
        {"pts": 30, "time": 1.2, "box": [0.2, 0.2, 0.25, 0.3], "score": 0.5, "kind": "head"},
    ]
    keys = index.add(records, [_vector(RED), _vector(BLUE), _vector(RED_ALT_OUTFIT)], _distance)
    assert keys[0] == keys[2] != keys[1]
    assert len(index) == 2
    public = index.public(lambda m: f"/p/{m['pts']}")
    red = next(item for item in public if item["character_key"] == keys[0])
    assert red["frame_count"] == 2
    # The clear, larger person crop represents the group, not the tiny head.
    assert red["representative"]["pts"] == 10


def test_character_keys_are_stable_when_display_order_and_representative_change():
    index = CharacterCandidateIndex("fp", threshold=0.2)
    first = index.add(
        [{"pts": 1, "time": 0.0, "box": [0.1, 0.1, 0.2, 0.3], "score": 0.4, "kind": "head"},
         {"pts": 2, "time": 1.0, "box": [0.5, 0.1, 0.9, 0.9], "score": 0.9, "kind": "person"},
         {"pts": 3, "time": 1.5, "box": [0.5, 0.1, 0.9, 0.9], "score": 0.9, "kind": "person"}],
        [_vector(RED), _vector(BLUE), _vector(BLUE)], _distance,
    )
    red_key, blue_key = first[0], first[1]
    before = [item["character_key"] for item in index.public(lambda m: "")]
    assert before == [blue_key, red_key]
    # "Show more": more red crops arrive, including a clearer representative.
    more = [{"pts": 10 + i, "time": 2.0 + i, "box": [0.05, 0.05, 0.5, 0.95], "score": 0.95, "kind": "person"} for i in range(3)]
    keys = index.add(more, [_vector(RED_ALT_OUTFIT)] * 3, _distance)
    after = index.public(lambda m: "")
    assert set(keys) == {red_key}
    assert [item["character_key"] for item in after] == [red_key, blue_key]  # display order changed
    red = next(item for item in after if item["character_key"] == red_key)
    assert red["representative"]["pts"] in {10, 11, 12}  # representative changed
    assert all(member["character_key"] == red_key for member in red["members"])
    assert detection_key("fp", 1, [0.1, 0.1, 0.2, 0.3]) == index.groups[0].anchor_detection_key


def test_without_ccip_contract_each_crop_is_its_own_candidate():
    index = CharacterCandidateIndex("fp")
    keys = index.add([{"pts": 1, "time": 0, "box": [0, 0, 0.5, 0.5]}, {"pts": 2, "time": 1, "box": [0, 0, 0.5, 0.5]}], None, None)
    assert keys[0] != keys[1]
    assert index.identity_grouping is False


# --------------------------------------------------------------------------
# Real reference pipeline with keyed multi-character selection
# --------------------------------------------------------------------------

def _two_character_video():
    return [
        figure_frame([(RED, 20, 20, 60, 100)], seed=1),
        figure_frame([(BLUE, 150, 20, 60, 100)], seed=2),
        figure_frame([(RED, 20, 20, 60, 100), (BLUE, 150, 20, 60, 100)], seed=3),
        figure_frame([(RED_ALT_OUTFIT, 90, 30, 55, 95)], seed=4),
        figure_frame([], seed=5),
    ]


def _crop(image, box):
    return image.crop(tuple(box))


def _run_pipeline(tmp_path, monkeypatch, references, keys):
    FrameListReader.frames = _two_character_video()
    monkeypatch.setattr(pipeline, "VideoReader", FrameListReader)
    settings = {
        "selection_mode": "reference", "start": 0.0, "end": 4.0, "coarse_step": 1.0, "dense_step": 1.0,
        "candidate_count": 0, "rescue": False, "quality_min": 0.0, "identity_threshold": 0.2,
    }
    if keys is not None:
        settings["_reference_character_keys"] = keys
    return pipeline.run_search("fixture.mp4", references, [], [], settings, FakeModels(), tmp_path)


def _labels(result):
    return {(item["pts"], item["character_key"]) for item in result["candidates"]}


def test_keyed_selection_reaches_matching_and_labels_each_character(tmp_path, monkeypatch):
    frames = _two_character_video()
    red_ref = _crop(frames[0], (18, 18, 82, 122))
    blue_ref = _crop(frames[1], (148, 18, 212, 122))
    result = _run_pipeline(tmp_path, monkeypatch, [red_ref, blue_ref], ["chr_red", "chr_blue"])
    labels = _labels(result)
    # The shared frame keeps one cut for each selected character.
    assert (1002, "chr_red") in labels and (1002, "chr_blue") in labels
    assert (1001, "chr_blue") in labels and (1000, "chr_red") in labels
    # The alternate outfit is still the red character.
    assert (1003, "chr_red") in labels
    assert not any(pts == 1004 for pts, _ in labels)  # empty background frame
    assert result["stats"]["selected_character_keys"] == ["chr_red", "chr_blue"]


def test_reordering_references_never_assigns_the_wrong_character(tmp_path, monkeypatch):
    frames = _two_character_video()
    red_ref = _crop(frames[0], (18, 18, 82, 122))
    blue_ref = _crop(frames[1], (148, 18, 212, 122))
    forward = _labels(_run_pipeline(tmp_path / "a", monkeypatch, [red_ref, blue_ref], ["chr_red", "chr_blue"]))
    reverse = _labels(_run_pipeline(tmp_path / "b", monkeypatch, [blue_ref, red_ref], ["chr_blue", "chr_red"]))
    assert forward == reverse


def test_single_character_selection_excludes_unselected_character(tmp_path, monkeypatch):
    frames = _two_character_video()
    red_refs = [_crop(frames[0], (18, 18, 82, 122)), _crop(frames[3], (88, 28, 147, 127))]
    result = _run_pipeline(tmp_path, monkeypatch, red_refs, ["chr_red", "chr_red"])
    assert {key for _pts, key in _labels(result)} == {"chr_red"}
    assert 1001 not in {pts for pts, _ in _labels(result)}  # blue-only frame
    assert result["stats"]["references_per_character"] == {"chr_red": 2}


# --------------------------------------------------------------------------
# HTTP flow: candidates -> select/deselect -> job -> run manifest
# --------------------------------------------------------------------------

class _Harness:
    def __init__(self, tmp_path, monkeypatch, *, frames=None, models=None):
        FrameListReader.frames = frames if frames is not None else _two_character_video()
        source = tmp_path / "Episode 01.mp4"
        source.write_bytes(b"fixture-source")
        self.video = server.VideoRecord(
            id="e" * 32, path=source,
            info={"width": 256, "height": 144, "duration": float(len(FrameListReader.frames)), "fps": 1.0, "path": str(source)},
            fingerprint="fake-fingerprint", created_at="now", display_name=source.name,
        )
        self.models = models or FakeModels()
        self.pipeline_calls: list[dict] = []
        monkeypatch.setattr(server, "VideoReader", FrameListReader)
        monkeypatch.setattr(server, "_load_model_service", lambda _root: self.models)
        monkeypatch.setattr(server, "_source_stat_unchanged", lambda _video: True)
        monkeypatch.setattr(server, "_archived_character_frames", lambda _video: [])
        monkeypatch.setattr(server, "_manifest_vocabulary", lambda: {"smile": "general"})
        monkeypatch.setitem(server.VIDEO_REGISTRY, self.video.id, self.video)
        monkeypatch.setattr(server, "RESULT_ROOT", tmp_path / "_results")
        monkeypatch.setattr(server, "REFERENCE_ROOT", tmp_path / "_refs")
        monkeypatch.setattr(server, "CACHE_ROOT", tmp_path)
        monkeypatch.setattr(server, "run_search", self._pipeline)
        self.fail_pipeline = False
        self.gate: threading.Event | None = None

    def _pipeline(self, **kwargs):
        if self.gate is not None:
            assert self.gate.wait(20), "test gate never opened"
        self.pipeline_calls.append({
            "references": len(kwargs["references"]),
            "keys": list(kwargs["settings"].get("_reference_character_keys") or []),
            "cut_quality_min": kwargs["settings"].get("cut_quality_min"),
        })
        if self.fail_pipeline:
            raise RuntimeError("fixture analysis failure")
        root = Path(kwargs["cache_dir"])
        root.mkdir(parents=True, exist_ok=True)
        preview = root / "c1-preview.jpg"
        Image.new("RGB", (16, 9), "#334455").save(preview)
        keys = kwargs["settings"].get("_reference_character_keys") or [None]
        return {"candidates": [
            {"id": f"cand-{index}", "pts": 1000 + index, "time": float(index), "bbox": [0, 0, 1, 1], "score": 0.9,
             "character_key": key, "preview_path": str(preview)}
            for index, key in enumerate(dict.fromkeys(keys))
        ], "warnings": [], "stats": {}}


def _wait(client, job_id):
    for _ in range(300):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in {"done", "error", "cancelled"}:
            return state
        time.sleep(0.01)
    raise AssertionError(state)


def _suggest(client, harness, page=0, times=None):
    times = times if times is not None else [0.0, 1.0, 2.0, 3.0, 4.0]
    response = client.post(f"/api/videos/{harness.video.id}/character-suggestions", json={"times": times, "page": page})
    return response


def _reference_for(client, harness, member):
    response = client.post(f"/api/videos/{harness.video.id}/reference", json={"pts": member["pts"], "box": member["box"]})
    assert response.status_code == 200, response.text
    data = response.json()
    return {"id": data["id"], "box": data["box"], "character_key": member["character_key"], "source_pts": member["pts"]}


def _job(client, harness, references, **settings):
    body = {"video_id": harness.video.id, "selection_mode": "reference", "references": references,
            "wanted": [], "avoided": [], "settings": {"candidate_count": 0, **settings}}
    response = client.post("/api/jobs", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_default_flow_presents_grouped_candidates_and_show_more_keeps_keys(tmp_path, monkeypatch):
    harness = _Harness(tmp_path, monkeypatch)
    with TestClient(server.create_app()) as client:
        first = _suggest(client, harness, times=[0.0, 1.0])
        assert first.status_code == 200, first.text
        data = first.json()
        assert data["identity_grouping"] is True
        assert len(data["characters"]) == 2
        keys = {group["character_key"] for group in data["characters"]}
        more = _suggest(client, harness, page=1, times=[2.0, 3.0, 4.0]).json()
        assert {group["character_key"] for group in more["characters"]} == keys  # outfit change merged
        red = max(more["characters"], key=lambda group: group["frame_count"])
        assert red["frame_count"] == 3
        preview = client.get(red["representative"]["preview_url"])
        assert preview.status_code == 200 and preview.headers["content-type"] == "image/jpeg"
        fresh = _suggest(client, harness, page=0, times=[1.0]).json()
        assert len(fresh["characters"]) == 1  # page 0 starts a new candidate screen


def test_zero_candidates_and_analysis_failure_are_reported_without_cpu_fallback(tmp_path, monkeypatch):
    harness = _Harness(tmp_path, monkeypatch, frames=[figure_frame([], seed=9)] * 3)
    with TestClient(server.create_app()) as client:
        empty = _suggest(client, harness, times=[0.0, 1.0, 2.0])
        assert empty.status_code == 200
        assert empty.json()["characters"] == [] and empty.json()["checked_frames"] == 3
        harness.models.fail_detect = True
        failed = _suggest(client, harness, page=1, times=[0.5])
        assert failed.status_code == 503
        assert "CPU fallback was not used" in failed.json()["detail"]


def _members(client, harness):
    data = _suggest(client, harness).json()
    groups = sorted(data["characters"], key=lambda group: -group["frame_count"])
    red, blue = groups[0], groups[1]
    return red, blue


def _manifest(run_id):
    return json.loads((server.RUN_ROOT / run_id / "run-manifest.json").read_text(encoding="utf-8"))


def test_single_multi_and_deselect_reach_extraction_and_manifest(tmp_path, monkeypatch):
    harness = _Harness(tmp_path, monkeypatch)
    with TestClient(server.create_app()) as client:
        red, blue = _members(client, harness)
        red_refs = [_reference_for(client, harness, member) for member in red["members"][:2]]
        blue_ref = _reference_for(client, harness, blue["representative"])

        single = _job(client, harness, red_refs[:1])
        state = _wait(client, single["job_id"])
        assert state["status"] == "done"
        assert harness.pipeline_calls[-1] == {"references": 1, "keys": [red["character_key"]], "cut_quality_min": 0.55}
        assert state["candidates"][0]["character_key"] == red["character_key"]

        multi = _job(client, harness, [*red_refs, blue_ref])
        state = _wait(client, multi["job_id"])
        call = harness.pipeline_calls[-1]
        assert call["references"] == 3
        assert call["keys"] == [red["character_key"], red["character_key"], blue["character_key"]]
        assert {item["character_key"] for item in state["candidates"]} == {red["character_key"], blue["character_key"]}
        manifest = _manifest(multi["run"]["run_id"])
        chars = {item["character_key"]: len(item["references"]) for item in manifest["selection"]["characters"]}
        assert chars == {red["character_key"]: 2, blue["character_key"]: 1}
        assert manifest["analysis"]["settings"]["cut_quality_min"] == 0.55
        assert manifest["status"] == "completed"
        assert Path(manifest["result_root"]).is_relative_to(server.RESULT_ROOT.resolve())
        assert not (tmp_path / "outputs").exists()  # manifests never land in public outputs

        deselected = _job(client, harness, red_refs)  # blue deselected
        _wait(client, deselected["job_id"])
        assert harness.pipeline_calls[-1]["keys"] == [red["character_key"]] * 2
        assert deselected["run"]["run_id"] not in {single["run"]["run_id"], multi["run"]["run_id"]}


def test_same_condition_rerun_reuses_but_threshold_or_selection_change_does_not(tmp_path, monkeypatch):
    harness = _Harness(tmp_path, monkeypatch)
    with TestClient(server.create_app()) as client:
        red, blue = _members(client, harness)
        refs = [_reference_for(client, harness, red["representative"])]
        first = _job(client, harness, refs)
        _wait(client, first["job_id"])
        assert len(harness.pipeline_calls) == 1

        # Order of keyed references and export-only logo options do not matter.
        again = _job(client, harness, refs, logo_removal=False)
        assert again["run"]["reused"] is True and again["run"]["run_id"] == first["run"]["run_id"]
        reused = _wait(client, again["job_id"])
        assert reused["status"] == "done" and reused["candidates"][0]["preview_url"]
        assert client.get(reused["candidates"][0]["preview_url"]).status_code == 200
        assert len(harness.pipeline_calls) == 1

        stricter = _job(client, harness, refs, cut_quality_min=0.7)
        assert stricter["run"]["reused"] is False and stricter["run"]["run_id"] != first["run"]["run_id"]
        _wait(client, stricter["job_id"])
        assert harness.pipeline_calls[-1]["cut_quality_min"] == 0.7

        identity = _job(client, harness, refs, identity_threshold=0.15)
        assert identity["run"]["reused"] is False
        _wait(client, identity["job_id"])

        other = _job(client, harness, [_reference_for(client, harness, blue["representative"])])
        assert other["run"]["reused"] is False
        _wait(client, other["job_id"])
        assert len(harness.pipeline_calls) == 4


def test_interrupted_run_resumes_after_restart_with_inherited_selection(tmp_path, monkeypatch):
    harness = _Harness(tmp_path, monkeypatch)
    with TestClient(server.create_app()) as client:
        red, blue = _members(client, harness)
        refs = [_reference_for(client, harness, red["representative"]), _reference_for(client, harness, blue["representative"])]
        first = _job(client, harness, refs, cut_quality_min=0.6, tag_threshold=0.3)
        _wait(client, first["job_id"])
        run_id = first["run"]["run_id"]
        # Simulate a crash mid-run: the manifest says running, the process is gone.
        manifest = _manifest(run_id)
        manifest["status"] = "running"
        run_manifest.write_manifest(server.RUN_ROOT, manifest)
        server.JOB_REGISTRY.clear()
        server.REFERENCE_REGISTRY.clear()

        listed = client.get(f"/api/videos/{harness.video.id}/runs").json()["runs"]
        assert [item["run_id"] for item in listed] == [run_id]
        assert listed[0]["status"] == "running"

        resumed = client.post(f"/api/runs/{run_id}/rerun", json={})
        assert resumed.status_code == 200, resumed.text
        info = resumed.json()["run"]
        assert info["run_id"] == run_id and info["resumed_from"] == "running" and info["reused"] is False
        state = _wait(client, resumed.json()["job_id"])
        assert state["status"] == "done"
        call = harness.pipeline_calls[-1]
        assert call["keys"] == [red["character_key"], blue["character_key"]] and call["cut_quality_min"] == 0.6
        assert _manifest(run_id)["attempts"] == 2

        rerun_same = client.post(f"/api/runs/{run_id}/rerun", json={}).json()
        assert rerun_same["run"]["reused"] is True
        changed = client.post(f"/api/runs/{run_id}/rerun", json={"settings": {"cut_quality_min": 0.8}}).json()
        assert changed["run"]["run_id"] != run_id and changed["run"]["reused"] is False
        _wait(client, changed["job_id"])
        assert harness.pipeline_calls[-1]["cut_quality_min"] == 0.8


def test_failed_analysis_is_not_reused_as_complete(tmp_path, monkeypatch):
    harness = _Harness(tmp_path, monkeypatch)
    with TestClient(server.create_app()) as client:
        red, _blue = _members(client, harness)
        refs = [_reference_for(client, harness, red["representative"])]
        harness.fail_pipeline = True
        failed = _job(client, harness, refs)
        state = _wait(client, failed["job_id"])
        assert state["status"] == "error" and "fixture analysis failure" in state["error"]
        assert _manifest(failed["run"]["run_id"])["status"] == "error"
        harness.fail_pipeline = False
        retried = _job(client, harness, refs)
        assert retried["run"]["resumed_from"] == "error" and retried["run"]["reused"] is False
        assert _wait(client, retried["job_id"])["status"] == "done"


def test_unknown_character_key_is_rejected():
    with pytest.raises(server.HTTPException):
        server._parse_reference_items([{"id": "x", "character_key": "../etc"}])


def test_unattended_modes_never_wait_for_a_selection(tmp_path, monkeypatch):
    harness = _Harness(tmp_path, monkeypatch)
    with TestClient(server.create_app()) as client:
        response = client.post("/api/jobs", json={"video_id": harness.video.id, "selection_mode": "automatic", "references": [], "settings": {}})
        assert response.status_code == 200, response.text
        assert _wait(client, response.json()["job_id"])["status"] == "done"
    # The nightly batch default (cute) is all-character and needs no stdin/selection.
    import builtins
    import sys

    import test_batch_source as batch_tests
    from scripts.batch_source import run

    def no_input(*_args, **_kwargs):
        raise AssertionError("unattended batch must never wait for input")

    monkeypatch.setattr(builtins, "input", no_input)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    batch_tests._install_cute_batch_pipeline(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "Episode.mp4").write_bytes(b"unattended fixture")
    assert run(["--source", str(source), "--output-root", str(tmp_path / "out"), "--state-root", str(tmp_path / "state"),
                "--cache-root", str(tmp_path / "cache"), "--no-logo-removal", "--no-archive", "--cut-quality-min", "0"]) == 0
    assert list((tmp_path / "out").rglob("images/*.png"))


def _concurrent_reruns(client, run_id, body, count=4):
    barrier = threading.Barrier(count)
    replies: list = [None] * count

    def submit(index):
        barrier.wait()
        replies[index] = client.post(f"/api/runs/{run_id}/rerun", json=body)

    threads = [threading.Thread(target=submit, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert all(reply is not None and reply.status_code == 200 for reply in replies), [getattr(r, "text", r) for r in replies]
    return [reply.json() for reply in replies]


def _jobs_for(run_id, *, include_reused=False):
    with server._STATE_LOCK:
        return [job for job in server.JOB_REGISTRY.values()
                if job.run.get("run_id") == run_id and (include_reused or not job.run.get("reused"))]


def test_concurrent_identical_submissions_start_exactly_one_job_per_run(tmp_path, monkeypatch):
    """Regression (D1): 4 simultaneous identical submissions, with and without
    force, never start more than one job for a run id."""

    harness = _Harness(tmp_path, monkeypatch)
    original_read = run_manifest.read_manifest

    def slow_read(*args, **kwargs):
        # Widen the check -> register window that the race needed.
        time.sleep(0.05)
        return original_read(*args, **kwargs)

    monkeypatch.setattr(run_manifest, "read_manifest", slow_read)
    with TestClient(server.create_app()) as client:
        red, _blue = _members(client, harness)
        first = _job(client, harness, [_reference_for(client, harness, red["representative"])])
        assert _wait(client, first["job_id"])["status"] == "done"
        source_run = first["run"]["run_id"]

        for trial in range(3):
            harness.gate = threading.Event()
            override = {"settings": {"cut_quality_min": round(0.6 + trial * 0.1, 2)}}
            normal = _concurrent_reruns(client, source_run, override)
            run_ids = {item["run"]["run_id"] for item in normal}
            assert len(run_ids) == 1 and source_run not in run_ids
            run_id = run_ids.pop()
            assert len({item["job_id"] for item in normal}) == 1
            assert sorted(item["run"]["attached"] for item in normal) == [False, True, True, True]
            # force while the job is live attaches as well (no parallel job).
            forced = _concurrent_reruns(client, source_run, {**override, "force": True})
            assert {item["job_id"] for item in forced} == {normal[0]["job_id"]}
            assert all(item["run"]["attached"] and item["run"]["force_attached"] for item in forced)
            assert len(_jobs_for(run_id)) == 1
            harness.gate.set()
            assert _wait(client, normal[0]["job_id"])["status"] == "done"
            manifest = _manifest(run_id)
            assert manifest["status"] == "completed" and manifest["attempts"] == 1
            assert manifest["job_id"] == normal[0]["job_id"] and manifest["rerun_of"] == source_run

            # force after completion: exactly one fresh job, the rest attach.
            harness.gate = threading.Event()
            again = _concurrent_reruns(client, source_run, {**override, "force": True})
            assert len({item["job_id"] for item in again}) == 1
            assert sorted(item["run"]["attached"] for item in again) == [False, True, True, True]
            assert len(_jobs_for(run_id)) == 2  # the finished one + exactly one new
            harness.gate.set()
            assert _wait(client, again[0]["job_id"])["status"] == "done"
            manifest = _manifest(run_id)
            assert manifest["status"] == "completed" and manifest["attempts"] == 2
            assert manifest["job_id"] == again[0]["job_id"]

            # Same condition, no force: everything reuses, still no new job.
            reused = _concurrent_reruns(client, source_run, override)
            assert all(item["run"]["reused"] for item in reused)
            assert len(_jobs_for(run_id)) == 2
        harness.gate = None


def test_cancelled_job_cannot_overwrite_the_manifest_of_its_replacement(tmp_path, monkeypatch):
    harness = _Harness(tmp_path, monkeypatch)
    with TestClient(server.create_app()) as client:
        red, _blue = _members(client, harness)
        refs = [_reference_for(client, harness, red["representative"])]
        harness.gate = threading.Event()
        first = _job(client, harness, refs)
        run_id = first["run"]["run_id"]
        for _ in range(500):
            if client.get(f"/api/jobs/{first['job_id']}").json()["status"] == "running":
                break
            time.sleep(0.01)
        client.post(f"/api/jobs/{first['job_id']}/cancel")
        # A cancel-requested job is not attached to; a replacement owns the run.
        second = _job(client, harness, refs)
        assert second["job_id"] != first["job_id"] and second["run"]["attached"] is False
        harness.gate.set()
        _wait(client, first["job_id"])
        assert _wait(client, second["job_id"])["status"] == "done"
        manifest = _manifest(run_id)
        assert manifest["status"] == "completed" and manifest["job_id"] == second["job_id"]
        harness.gate = None
