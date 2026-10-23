"""CPU-only coverage for the reusable, resumable batch analysis path."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from anime_picker.batch_analysis import ScannedFrame, analyze_identity, scan_video, select_candidates
from scripts.batch_source import _automatic_reference_groups, _deduplicate_group_candidates


@dataclass
class FakeFrame:
    pts: int
    time: float
    image: Image.Image


class FakeReader:
    def __init__(self, frames: list[FakeFrame], duration: float = 10.0) -> None:
        self.frames = frames
        self.info = SimpleNamespace(duration=duration, fps=24.0)
        self.iter_calls = 0

    def iter_frames(self, start=0.0, end=None, max_side=960):
        self.iter_calls += 1
        stop = self.info.duration if end is None else end
        for frame in self.frames:
            if start <= frame.time <= stop:
                yield frame

    def get_frame(self, pts: int, max_side=960):
        return next(frame for frame in self.frames if frame.pts == pts)


class TolerantFakeReader(FakeReader):
    """Emulate VideoReader returning a frame within its end-time tolerance."""

    def iter_frames(self, start=0.0, end=None, max_side=960):
        self.iter_calls += 1
        stop = self.info.duration if end is None else end
        tolerance = 1e-9
        for frame in self.frames:
            if start - tolerance <= frame.time <= stop + tolerance:
                yield frame


def _solid(value: int, size: tuple[int, int] = (96, 54)) -> Image.Image:
    return Image.new("RGB", size, (value, value, value))


def test_all_mode_candidate_selection_has_no_small_fixed_count_cap() -> None:
    thumbnail = np.zeros((27, 48), dtype=np.uint8)
    frames = [
        ScannedFrame(
            pts=index,
            time=float(index) * 0.5,
            sharpness=1.0,
            brightness=120.0,
            contrast=40.0,
            thumbnail=thumbnail,
        )
        for index in range(41)
    ]

    candidates, stats = select_candidates(frames, spacing_seconds=0.5)

    assert len(candidates) > 12
    assert stats["selected"] == len(candidates)


def _read_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def _write_npz_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _decode_metadata(array: np.ndarray) -> dict:
    return json.loads(array.tobytes().decode("utf-8"))


def _encode_metadata(metadata: dict) -> np.ndarray:
    return np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)


def test_scan_checkpoints_reuse_and_fingerprint_invalidation(tmp_path: Path) -> None:
    frames = [FakeFrame(i, float(i), _solid(20 + i * 15)) for i in range(9)]
    reader = FakeReader(frames, duration=10.0)
    first, first_stats, key = scan_video(reader, tmp_path, "fingerprint-a", chunk_seconds=5.0)
    assert len(first) == 9
    assert first_stats["chunks_rebuilt"] == 2
    calls = reader.iter_calls

    second, second_stats, same_key = scan_video(reader, tmp_path, "fingerprint-a", chunk_seconds=5.0)
    assert same_key == key
    assert len(second) == len(first)
    assert second_stats["chunks_cache_hit"] == 2
    assert reader.iter_calls == calls

    _, changed_stats, changed_key = scan_video(reader, tmp_path, "fingerprint-b", chunk_seconds=5.0)
    assert changed_key != key
    assert changed_stats["chunks_rebuilt"] == 2


def test_scan_interruption_resumes_completed_chunks(tmp_path: Path) -> None:
    frames = [FakeFrame(i, float(i), _solid(30 + i * 20)) for i in range(6)]
    reader = FakeReader(frames, duration=10.0)

    def interrupt_after_first_chunk(done: int, _total: int) -> None:
        if done == 1:
            raise InterruptedError("fixture interruption")

    with pytest.raises(InterruptedError):
        scan_video(reader, tmp_path, "resume-fixture", chunk_seconds=5.0, progress=interrupt_after_first_chunk)
    first_pass_calls = reader.iter_calls
    assert first_pass_calls == 1

    resumed, stats, _key = scan_video(reader, tmp_path, "resume-fixture", chunk_seconds=5.0)
    assert len(resumed) == 6
    assert stats["chunks_cache_hit"] == 1
    assert stats["chunks_rebuilt"] == 1
    assert reader.iter_calls == first_pass_calls + 1


def test_scan_half_open_chunks_survive_tolerant_reader_resume(tmp_path: Path) -> None:
    frames = [
        FakeFrame(1, 1.0, _solid(30)),
        FakeFrame(5, 5.0, _solid(120)),
        FakeFrame(6, 6.0, _solid(150)),
        FakeFrame(10, 10.0, _solid(220)),
    ]
    reader = TolerantFakeReader(frames, duration=10.0)

    def interrupt_after_first_chunk(done: int, _total: int) -> None:
        if done == 1:
            raise InterruptedError("fixture interruption")

    with pytest.raises(InterruptedError):
        scan_video(reader, tmp_path, "boundary-resume-fixture", chunk_seconds=5.0, progress=interrupt_after_first_chunk)

    # Simulate an older/tampered checkpoint that stored the exact boundary in
    # the preceding chunk. It remains parseable and otherwise internally valid.
    first_chunk = next((tmp_path / "scan").rglob("chunk_000000.npz"))
    arrays = _read_npz_arrays(first_chunk)
    metadata = _decode_metadata(arrays["metadata"])
    arrays["pts"] = np.append(arrays["pts"], np.int64(5))
    arrays["times"] = np.append(arrays["times"], np.float64(5.0))
    arrays["sharpness"] = np.append(arrays["sharpness"], np.float32(0.0))
    arrays["brightness"] = np.append(arrays["brightness"], np.float32(120.0))
    arrays["contrast"] = np.append(arrays["contrast"], np.float32(0.0))
    arrays["thumbnails"] = np.concatenate(
        [arrays["thumbnails"], np.zeros((1, 27, 48), dtype=np.uint8)], axis=0
    )
    metadata["frame_count"] += 1
    arrays["metadata"] = _encode_metadata(metadata)
    _write_npz_arrays(first_chunk, arrays)

    resumed, stats, _key = scan_video(reader, tmp_path, "boundary-resume-fixture", chunk_seconds=5.0)
    pts = [frame.pts for frame in resumed]
    assert sorted(pts) == [1, 5, 6, 10]
    assert len(pts) == len(set(pts))
    assert stats["chunks_cache_hit"] == 0
    assert stats["chunks_rebuilt"] == 2

    cached, cached_stats, _key = scan_video(reader, tmp_path, "boundary-resume-fixture", chunk_seconds=5.0)
    cached_pts = [frame.pts for frame in cached]
    assert sorted(cached_pts) == [1, 5, 6, 10]
    assert len(cached_pts) == len(set(cached_pts))
    assert cached_stats["chunks_cache_hit"] == 2
    assert cached_stats["chunks_rebuilt"] == 0


@pytest.mark.parametrize(
    "defect",
    ["frame_count", "thumbnail_shape", "out_of_chunk_time", "nonfinite_metric", "pts_time_order"],
)
def test_scan_rebuilds_malformed_but_parseable_checkpoint(tmp_path: Path, defect: str) -> None:
    frames = [FakeFrame(i + 1, float(i), _solid(40 + i * 20)) for i in range(3)]
    reader = FakeReader(frames, duration=3.0)
    _frames, _stats, _key = scan_video(reader, tmp_path, "corrupt-fixture", chunk_seconds=5.0)
    path = next((tmp_path / "scan").rglob("chunk_000000.npz"))
    arrays = _read_npz_arrays(path)
    metadata = _decode_metadata(arrays["metadata"])

    if defect == "frame_count":
        metadata["frame_count"] += 1
        arrays["metadata"] = _encode_metadata(metadata)
    elif defect == "thumbnail_shape":
        arrays["thumbnails"] = arrays["thumbnails"][:, :, :-1]
    elif defect == "out_of_chunk_time":
        arrays["times"][1] = 500.0
    elif defect == "nonfinite_metric":
        arrays["sharpness"][0] = np.nan
    else:
        arrays["pts"][1] = arrays["pts"][0] - 1
    _write_npz_arrays(path, arrays)
    calls_before_restore = reader.iter_calls

    restored, stats, _key = scan_video(reader, tmp_path, "corrupt-fixture", chunk_seconds=5.0)
    assert len(restored) == 3
    assert stats["chunks_cache_hit"] == 0
    assert stats["chunks_rebuilt"] == 1
    assert reader.iter_calls == calls_before_restore + 1


def test_cached_scan_selection_does_not_reopen_video(monkeypatch, tmp_path: Path) -> None:
    frames = [FakeFrame(i, float(i), _solid(20 + i * 30)) for i in range(4)]
    reader = FakeReader(frames, duration=4.0)
    _scanned, _stats, _key = scan_video(reader, tmp_path, "scene-cache-fixture", chunk_seconds=5.0)
    cached, cache_stats, _key = scan_video(reader, tmp_path, "scene-cache-fixture", chunk_seconds=5.0)
    assert cache_stats["chunks_cache_hit"] == 1

    def forbidden_open(_path):
        raise AssertionError("candidate selection must reuse scan summaries without reopening the source")

    scene_api = SimpleNamespace(
        open_video=forbidden_open,
        ContentDetector=object,
        SceneManager=object,
    )
    monkeypatch.setitem(sys.modules, "scenedetect", scene_api)
    candidates, stats = select_candidates(cached, spacing_seconds=0.5, video_path=tmp_path / "unused.mp4")
    assert candidates
    assert stats["scene_boundary_source"] == "already-decoded grayscale scan summaries"
    assert reader.iter_calls == 1


def test_candidate_selection_keeps_dark_frames_and_records_coverage() -> None:
    frames = [
        ScannedFrame(1, 0.0, 4.0, 3.0, 1.0, np.zeros((27, 48), dtype=np.uint8)),
        ScannedFrame(2, 0.5, 8.0, 4.0, 1.0, np.full((27, 48), 1, dtype=np.uint8)),
        ScannedFrame(3, 1.0, 7.0, 5.0, 1.0, np.full((27, 48), 2, dtype=np.uint8)),
    ]
    candidates, stats = select_candidates(frames, spacing_seconds=0.5)
    assert candidates
    assert stats["quality_gate_applied"] is False
    assert stats["selected"] == len(candidates)
    assert any(item["brightness"] <= 5.0 for item in candidates)


def test_identical_small_thumbnails_do_not_collapse_distinct_candidates() -> None:
    same_thumbnail = np.full((27, 48), 90, dtype=np.uint8)
    candidates, _stats = select_candidates(
        [
            ScannedFrame(1, 0.0, 1.0, 90.0, 0.0, same_thumbnail),
            ScannedFrame(2, 1.0, 1.0, 90.0, 0.0, same_thumbnail.copy()),
        ],
        spacing_seconds=0.5,
    )
    assert [item["pts"] for item in candidates] == [1, 2]


def test_near_duplicate_filter_preserves_target_expression_change() -> None:
    base = Image.new("L", (48, 27), 80)
    changed = base.copy()
    draw = ImageDraw.Draw(changed)
    draw.rectangle((19, 10, 28, 16), fill=240)
    candidates = [
        {"pts": 1, "time": 1.0, "score": 0.9, "bbox": [0.2, 0.1, 0.8, 0.9], "thumbnail": np.asarray(base)},
        {"pts": 2, "time": 1.2, "score": 0.8, "bbox": [0.2, 0.1, 0.8, 0.9], "thumbnail": np.asarray(changed)},
    ]
    kept = _deduplicate_group_candidates(candidates)
    assert [item["pts"] for item in kept] == [1, 2]


def test_identity_analysis_cache_is_shared_by_later_modes(tmp_path: Path) -> None:
    frames = [FakeFrame(i + 1, float(i + 1), _solid(20 + i * 20)) for i in range(3)]
    reader = FakeReader(frames, duration=5.0)

    class Model:
        def __init__(self):
            self.detect_calls = 0
            self.embed_calls = 0

        def detect(self, image):
            self.detect_calls += 1
            return [{"box": [4, 4, 90, 50], "score": 0.9, "kind": "person"}]

        def embed(self, images):
            self.embed_calls += 1
            return np.asarray([[float(np.asarray(image)[0, 0, 0])] for image in images], dtype=np.float32)

    def boxes(detections, size, limit):
        return [(list(detections[0]["box"]), detections[0]["score"], "person")]

    candidates = [{"pts": frame.pts, "time": frame.time} for frame in frames]
    model = Model()
    records, first = analyze_identity(
        reader, model, candidates, tmp_path, "shared-scan", "model-a",
        chunk_seconds=5.0, character_boxes=boxes,
        crop_image=lambda image, box: image.crop(tuple(map(int, box))),
    )
    assert len(records) == 3
    assert first["chunks_rebuilt"] == 1
    assert model.detect_calls == 3

    cached, second = analyze_identity(
        reader, model, candidates, tmp_path, "shared-scan", "model-a",
        chunk_seconds=5.0, character_boxes=boxes,
        crop_image=lambda image, box: image.crop(tuple(map(int, box))),
    )
    assert second["chunks_cache_hit"] == 1
    assert model.detect_calls == 3
    assert np.array_equal(records[0]["feature"], cached[0]["feature"])

    checkpoint = next((tmp_path / "identity").rglob("chunk_000000.npz"))
    arrays = _read_npz_arrays(checkpoint)
    metadata = _decode_metadata(arrays["metadata"])
    metadata["records"][1]["feature_index"] = 999
    arrays["metadata"] = _encode_metadata(metadata)
    _write_npz_arrays(checkpoint, arrays)

    repaired, repaired_stats = analyze_identity(
        reader, model, candidates, tmp_path, "shared-scan", "model-a",
        chunk_seconds=5.0, character_boxes=boxes,
        crop_image=lambda image, box: image.crop(tuple(map(int, box))),
    )
    assert len(repaired) == 3
    assert repaired_stats["chunks_cache_hit"] == 0
    assert repaired_stats["chunks_rebuilt"] == 1
    assert model.detect_calls == 6

    _, changed = analyze_identity(
        reader, model, candidates, tmp_path, "shared-scan", "model-b",
        chunk_seconds=5.0, character_boxes=boxes,
        crop_image=lambda image, box: image.crop(tuple(map(int, box))),
    )
    assert changed["chunks_rebuilt"] == 1
    assert model.detect_calls == 9


def test_identity_analysis_does_not_cap_crops_per_frame(tmp_path: Path) -> None:
    frame = FakeFrame(101, 0.5, _solid(90))
    reader = FakeReader([frame], duration=1.0)

    class Model:
        def detect(self, image):
            return [
                {"box": [index * 20, 4, index * 20 + 16, 40], "score": 0.9 - index * 0.1, "kind": "person"}
                for index in range(4)
            ]

        def embed(self, images):
            return np.asarray([[image.width, image.height] for image in images], dtype=np.float32)

    def boxes(detections, size, limit):
        assert limit is None
        selected = detections if limit is None else detections[:limit]
        return [(list(item["box"]), item["score"], item["kind"]) for item in selected]

    records, stats = analyze_identity(
        reader,
        Model(),
        [{"pts": frame.pts, "time": frame.time}],
        tmp_path,
        "uncapped-scan",
        "model-a",
        chunk_seconds=5.0,
        character_boxes=boxes,
        crop_image=lambda image, box: image.crop(tuple(map(int, box))),
    )

    assert len(records) == 4
    assert stats["embedded_crops"] == 4
    assert len({tuple(record["bbox"]) for record in records}) == 4


def test_frequency_support_counts_distinct_pts_not_boxes() -> None:
    class Model:
        threshold = 0.2

        def distance(self, refs, values):
            return np.abs(np.asarray(refs)[:, None, 0] - np.asarray(values)[None, :, 0])

    records = [
        {"pts": 100, "time": 1.0, "feature": np.asarray([0.01], dtype=np.float32), "detector_confidence": 0.9, "bbox": [0, 0, 1, 1], "detector_kind": "person"},
        {"pts": 100, "time": 1.0, "feature": np.asarray([0.02], dtype=np.float32), "detector_confidence": 0.8, "bbox": [0, 0, 1, 1], "detector_kind": "head"},
        {"pts": 200, "time": 2.0, "feature": np.asarray([0.03], dtype=np.float32), "detector_confidence": 0.9, "bbox": [0, 0, 1, 1], "detector_kind": "person"},
    ]
    groups, stats = _automatic_reference_groups(Model(), records, maximum=1)
    assert len(groups) == 1
    assert groups[0].support_frames == 2
    assert stats["candidate_frames"] == 2
