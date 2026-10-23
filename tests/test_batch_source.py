"""GPU-free coverage for the headless batch entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.batch_source import (
    _args,
    _automatic_reference_groups,
    _best_at_pts_candidates,
    _cute_candidate_for_curation,
    _curate_character_candidates,
    _dedup_frame_pts,
    _deduplicate_group_candidates,
    _deduplicate_with_reader,
    _export_character,
    _file_digest,
    _run_jobs,
    _safe_output_path,
    _verified_completed_outputs,
    auto_references,
    character_boxes,
    discover_videos,
    reference_files,
    run,
)


def test_batch_rejects_state_root_inside_public_outputs(tmp_path: Path) -> None:
    public_root = tmp_path / "outputs"
    private_root = public_root / ".private"
    args = _args([
        "--source", str(tmp_path / "source"),
        "--output-root", str(public_root),
        "--state-root", str(private_root),
    ])
    with pytest.raises(ValueError, match="separate, non-nested"):
        _run_jobs(args, [], [], [], [], None, "none", {}, {})
    assert not public_root.exists()


def test_discover_videos_only_direct_mp4_and_mkv(tmp_path: Path) -> None:
    for name in ("b.MKV", "A.mp4", "sub.vtt", "info.txt"):
        (tmp_path / name).write_bytes(b"sample")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.mp4").write_bytes(b"nested")
    assert [video.name for video in discover_videos(tmp_path)] == ["A.mp4", "b.MKV"]


def test_crop_selection_filters_nested_head_and_whole_frame() -> None:
    selected = character_boxes(
        [
            {"box": [0, 0, 100, 100], "score": 0.98, "kind": "person"},
            {"box": [10, 10, 60, 90], "score": 0.89, "kind": "person"},
            {"box": [21, 15, 42, 45], "score": 0.91, "kind": "head"},
            {"box": [69, 15, 99, 70], "score": 0.75, "kind": "person"},
        ],
        (100, 100),
    )
    assert len(selected) == 2
    assert [item[2] for item in selected] == ["person", "person"]
    assert all(box != [0, 0, 100, 100] for box, _, _ in selected)


def test_character_detection_has_no_default_three_box_cap() -> None:
    detections = [
        {"box": [index * 100 + 5, 10, index * 100 + 85, 90], "score": 0.9 - index * 0.01}
        for index in range(5)
    ]
    assert len(character_boxes(detections, (500, 100))) == 5


def test_auto_topn_accepts_more_than_twelve_and_preserves_identity_separation() -> None:
    class DistanceModel:
        threshold = 0.01

        def distance(self, references: np.ndarray, values: np.ndarray) -> np.ndarray:
            return np.abs(references[:, None, 0] - values[None, :, 0])

    canonical = _args(["--mode", "auto-topn", "--top-n", "13"])
    legacy_alias = _args(["--mode", "auto", "--characters-per-video", "13"])
    assert canonical.mode == legacy_alias.mode == "auto-topn"
    assert canonical.characters_per_video == legacy_alias.characters_per_video == 13

    records = []
    for identity in range(13):
        for sample in range(2):
            pts = identity * 2 + sample
            records.append({
                "pts": pts,
                "time": float(identity * 4 + sample * 2),
                "bbox": [0.1, 0.1, 0.9, 0.9],
                "detector_kind": "person",
                "detector_confidence": 0.9,
                "feature": np.asarray([float(identity)], dtype=np.float32),
            })
    groups, stats = _automatic_reference_groups(DistanceModel(), records, canonical.characters_per_video)
    assert len(groups) == 13
    assert stats["identity_groups"] == 13
    group_pts = [{int(item["source_pts"]) for item in group.evidence} for group in groups]
    assert all(len(points) == 2 for points in group_pts)
    assert all(left.isdisjoint(right) for index, left in enumerate(group_pts) for right in group_pts[index + 1:])


def test_batch_decode_threads_default_low_and_rejects_zero() -> None:
    assert _args([]).decode_threads == 1
    with pytest.raises(SystemExit):
        _args(["--decode-threads", "0"])


def test_duplicate_confirmation_uses_high_resolution_and_keeps_expression_changes() -> None:
    from PIL import ImageDraw

    def face(expression: str) -> Image.Image:
        image = Image.new("RGB", (640, 360), (40, 60, 90))
        draw = ImageDraw.Draw(image)
        draw.ellipse((245, 35, 395, 330), fill=(230, 190, 150))
        draw.ellipse((278, 130, 298, 150), fill=(15, 15, 15))
        draw.ellipse((342, 130, 362, 150), fill=(15, 15, 15))
        if expression == "smile":
            draw.arc((288, 205, 352, 255), start=10, end=170, fill=(70, 0, 0), width=9)
        else:
            draw.arc((288, 215, 352, 265), start=190, end=350, fill=(70, 0, 0), width=9)
        return image

    first, second = face("smile"), face("frown")
    same_thumb = np.full((27, 48), 120, dtype=np.uint8)
    candidates = [
        {"pts": 10, "time": 1.0, "score": 0.8, "bbox": [0.3, 0.05, 0.7, 0.95], "detector_kind": "person", "thumbnail": same_thumb},
        {"pts": 20, "time": 1.25, "score": 0.7, "bbox": [0.3, 0.05, 0.7, 0.95], "detector_kind": "person", "thumbnail": same_thumb},
    ]
    images = {10: first, 20: second}
    assert len(_deduplicate_group_candidates(candidates, frame_lookup=images.get)) == 2
    assert len(_deduplicate_group_candidates(candidates)) == 2  # Missing high-res evidence never rejects.

    exact = [dict(candidates[0], pts=30, time=1.5), dict(candidates[0], pts=40, time=1.75)]
    images.update({30: first, 40: first.copy()})
    assert len(_deduplicate_group_candidates(exact, frame_lookup=images.get)) == 1

    red = Image.new("RGB", (640, 360), (255, 0, 0))
    green = Image.new("RGB", (640, 360), (0, 130, 0))
    assert red.convert("L").tobytes() == green.convert("L").tobytes()
    hue_candidates = [
        {"pts": 50, "time": 2.0, "score": 0.8, "thumbnail": np.full((27, 48), 76, dtype=np.uint8)},
        {"pts": 60, "time": 2.25, "score": 0.7, "thumbnail": np.full((27, 48), 76, dtype=np.uint8)},
    ]
    assert len(_deduplicate_group_candidates(hue_candidates, frame_lookup={50: red, 60: green}.get)) == 2


def test_all_mode_preserves_local_change_and_scene_boundary_events() -> None:
    image = Image.new("RGB", (640, 360), (40, 60, 90))
    thumbnail = np.full((27, 48), 90, dtype=np.uint8)
    candidates = [
        {"pts": 10, "time": 1.0, "score": 1.0, "reasons": ["coverage_anchor"], "thumbnail": thumbnail},
        {"pts": 20, "time": 1.25, "score": 0.1, "reasons": ["local_visual_change"], "thumbnail": thumbnail},
        {"pts": 30, "time": 1.5, "score": 0.2, "reasons": ["scene_boundary"], "thumbnail": thumbnail},
    ]
    frame_lookup = {10: image, 20: image.copy(), 30: image.copy()}.get

    # These frames are exact pixel duplicates, but event reasons indicate a
    # deliberate scene/local-change candidate in comprehensive all mode.
    assert [item["pts"] for item in _deduplicate_group_candidates(candidates, frame_lookup=frame_lookup)] == [10]
    assert [item["pts"] for item in _deduplicate_group_candidates(
        candidates,
        frame_lookup=frame_lookup,
        preserve_scan_events=True,
    )] == [10, 20, 30]


def test_all_mode_still_collapses_duplicate_coverage_anchors() -> None:
    image = Image.new("RGB", (640, 360), (40, 60, 90))
    thumbnail = np.full((27, 48), 90, dtype=np.uint8)
    candidates = [
        {"pts": 10, "time": 1.0, "score": 0.8, "reasons": ["coverage_anchor"], "thumbnail": thumbnail},
        {"pts": 20, "time": 1.25, "score": 0.7, "reasons": ["coverage_anchor"], "thumbnail": thumbnail},
    ]
    frame_lookup = {10: image, 20: image.copy()}.get
    assert [item["pts"] for item in _deduplicate_group_candidates(
        candidates,
        frame_lookup=frame_lookup,
        preserve_scan_events=True,
        pixel_hash_distance=0,
        rgb_similarity_threshold=0.000001,
    )] == [10]


def test_all_mode_keeps_subtle_coverage_expression_change() -> None:
    from PIL import ImageDraw

    def face(mouth_shift: int) -> Image.Image:
        image = Image.new("RGB", (640, 360), (40, 60, 90))
        draw = ImageDraw.Draw(image)
        draw.ellipse((245, 35, 395, 330), fill=(230, 190, 150))
        draw.ellipse((278, 130, 298, 150), fill=(15, 15, 15))
        draw.ellipse((342, 130, 362, 150), fill=(15, 15, 15))
        draw.arc((288, 205 + mouth_shift, 352, 255 + mouth_shift), start=10, end=170, fill=(70, 0, 0), width=9)
        return image

    first, changed = face(0), face(5)
    thumbnail = np.full((27, 48), 90, dtype=np.uint8)
    candidates = [
        {"pts": 10, "time": 1.0, "score": 0.9, "reasons": ["coverage_anchor"], "thumbnail": thumbnail},
        {"pts": 20, "time": 1.25, "score": 0.8, "reasons": ["coverage_anchor"], "thumbnail": thumbnail},
    ]
    frame_lookup = {10: first, 20: changed}.get

    # The former all-mode tolerance collapsed this small but visible mouth
    # change because the rest of the frame remained static.
    assert [item["pts"] for item in _deduplicate_group_candidates(candidates, frame_lookup=frame_lookup)] == [10]
    assert [item["pts"] for item in _deduplicate_group_candidates(
        candidates,
        frame_lookup=frame_lookup,
        preserve_scan_events=True,
        pixel_hash_distance=0,
        rgb_similarity_threshold=0.000001,
    )] == [10, 20]


def test_prefetched_dedup_matches_legacy_boundary_ties_and_replacement_decisions() -> None:
    from PIL import ImageDraw

    red = Image.new("RGB", (160, 90), (180, 25, 30))

    def face(expression: str) -> Image.Image:
        result = Image.new("RGB", (640, 360), (40, 60, 90))
        draw = ImageDraw.Draw(result)
        draw.ellipse((245, 35, 395, 330), fill=(230, 190, 150))
        draw.ellipse((278, 130, 298, 150), fill=(15, 15, 15))
        draw.ellipse((342, 130, 362, 150), fill=(15, 15, 15))
        if expression == "smile":
            draw.arc((288, 205, 352, 255), start=10, end=170, fill=(70, 0, 0), width=9)
        else:
            draw.arc((288, 215, 352, 265), start=190, end=350, fill=(70, 0, 0), width=9)
        return result

    smile, frown = face("smile"), face("frown")
    green = Image.new("RGB", (160, 90), (15, 110, 30))
    thumbnail = np.full((27, 48), 96, dtype=np.uint8)
    candidates = [
        # Exactly 2 seconds remains inside the old active window. A score tie
        # retains the earlier candidate.
        {"pts": 10, "time": 0.0, "score": 0.7, "thumbnail": thumbnail},
        {"pts": 20, "time": 2.0, "score": 0.7, "thumbnail": thumbnail},
        {"pts": 30, "time": 2.25, "score": 0.9, "thumbnail": thumbnail},
        # Head/person pairs are deliberately kept separately; the stronger
        # third candidate then replaces both prior pixel matches.
        {"pts": 40, "time": 3.0, "score": 0.4, "detector_kind": "head", "bbox": [0.1, 0.1, 0.7, 0.9], "thumbnail": thumbnail},
        {"pts": 50, "time": 3.2, "score": 0.5, "detector_kind": "person", "bbox": [0.12, 0.12, 0.70, 0.88], "thumbnail": thumbnail},
        {"pts": 60, "time": 3.4, "score": 0.9, "detector_kind": "other", "bbox": [0.11, 0.11, 0.69, 0.89], "thumbnail": thumbnail},
        # Pixel-level expression and color changes must remain separate.
        {"pts": 70, "time": 6.0, "score": 0.8, "thumbnail": thumbnail},
        {"pts": 80, "time": 6.25, "score": 0.7, "thumbnail": thumbnail},
        {"pts": 90, "time": 7.0, "score": 0.8, "thumbnail": thumbnail},
        {"pts": 100, "time": 7.25, "score": 0.7, "thumbnail": thumbnail},
    ]
    images = {
        10: red, 20: red.copy(), 30: green.copy(),
        40: red.copy(), 50: red.copy(), 60: red.copy(),
        70: smile, 80: frown,
        90: Image.new("RGB", red.size, (255, 0, 0)),
        100: green,
    }

    legacy = _deduplicate_group_candidates(candidates, frame_lookup=images.get)

    ordered = _best_at_pts_candidates(candidates)
    requested = _dedup_frame_pts(ordered)
    lookups: list[int] = []

    def ordered_lookup(pts: int):
        lookups.append(pts)
        return images.get(pts)

    prefetched = _deduplicate_group_candidates(
        ordered,
        frame_lookup=ordered_lookup,
        frame_release=lambda _pts: None,
        preload_pts=set(requested),
    )
    assert lookups == requested

    fields = ("pts", "time", "score", "phash", "dhash", "crop_phash", "crop_dhash")
    assert [{key: item.get(key) for key in fields} for item in prefetched] == [
        {key: item.get(key) for key in fields} for item in legacy
    ]
    assert [item["pts"] for item in prefetched] == [10, 30, 60, 70, 80, 90, 100]


def test_reader_dedup_preserves_scan_events_without_leaving_sequential_fast_path() -> None:
    from types import SimpleNamespace

    image = Image.new("RGB", (160, 90), (50, 80, 120))
    thumbnail = np.full((27, 48), 100, dtype=np.uint8)

    class OrderedReader:
        def __init__(self) -> None:
            self.validated_pts: list[int] = []
            self.iterated_pts: list[int] = []

        def validate_sequential_pts(self, pts, *, metrics):
            self.validated_pts = list(pts)
            return True, "safe synthetic CFR stream"

        def iter_selected_frames(self, pts, max_side=960, *, metrics):
            del max_side, metrics
            self.iterated_pts = list(pts)
            for pts_value in self.iterated_pts:
                yield SimpleNamespace(pts=pts_value, image=image)

    candidates = [
        {"pts": 10, "time": 1.0, "score": 0.9, "reasons": ["coverage_anchor"], "thumbnail": thumbnail},
        {"pts": 20, "time": 1.25, "score": 0.1, "reasons": ["local_visual_change"], "thumbnail": thumbnail},
        {"pts": 30, "time": 1.5, "score": 0.2, "reasons": ["scene_boundary"], "thumbnail": thumbnail},
        {"pts": 40, "time": 1.75, "score": 0.8, "reasons": ["coverage_anchor"], "thumbnail": thumbnail},
    ]
    reader = OrderedReader()

    kept, stats = _deduplicate_with_reader(candidates, reader, preserve_scan_events=True)

    assert [item["pts"] for item in kept] == [10, 20, 30]
    assert reader.validated_pts == [10, 40]
    assert reader.iterated_pts == [10, 40]
    assert stats["sequential_fast_path"] is True
    assert stats["optimized_used"] is True
    assert stats["strict_fallback"] is False


def test_missing_sequential_image_evidence_retains_candidates() -> None:
    thumbnail = np.full((27, 48), 100, dtype=np.uint8)
    candidates = [
        {"pts": 11, "time": 1.0, "score": 0.9, "thumbnail": thumbnail},
        {"pts": 22, "time": 1.25, "score": 0.8, "thumbnail": thumbnail},
    ]
    assert [item["pts"] for item in _deduplicate_group_candidates(candidates, frame_lookup=lambda _pts: None)] == [11, 22]


def test_reader_dedup_reports_strict_fallback_and_uses_exact_get_frame() -> None:
    image = Image.new("RGB", (160, 90), (50, 80, 120))
    thumbnail = np.full((27, 48), 100, dtype=np.uint8)

    class FallbackReader:
        def __init__(self) -> None:
            self.lookups: list[int] = []

        def validate_sequential_pts(self, _pts, *, metrics):
            return False, "synthetic VFR stream"

        def iter_selected_frames(self, *_args, **_kwargs):
            raise AssertionError("unsafe streams must use exact lookup fallback")

        def get_frame(self, pts: int, max_side: int = 960, *, metrics):
            self.lookups.append(pts)
            metrics["reader_opens"] += 1
            metrics["decoded_frames"] += 5
            return image

    candidates = [
        {"pts": 1, "time": 1.0, "score": 0.9, "thumbnail": thumbnail},
        {"pts": 2, "time": 1.25, "score": 0.8, "thumbnail": thumbnail},
    ]
    reader = FallbackReader()
    kept, stats = _deduplicate_with_reader(candidates, reader)

    assert [item["pts"] for item in kept] == [1]
    assert sorted(reader.lookups) == [1, 2]
    assert stats["strict_fallback"] is True
    assert stats["sequential_fast_path"] is False
    assert stats["fallback_reason"] == "synthetic VFR stream"
    assert stats["reader_opens"] == 2
    assert stats["decoded_frames"] == 10


def test_failed_validation_keeps_fallback_lazy_and_orders_only_the_later_pair() -> None:
    from types import SimpleNamespace

    image = Image.new("RGB", (160, 90), (50, 80, 120))
    candidates = [
        {"pts": 11, "time": 1.0, "score": 0.9, "thumbnail": np.zeros((27, 48), dtype=np.uint8)},
        {"pts": 22, "time": 1.25, "score": 0.8, "thumbnail": np.full((27, 48), 255, dtype=np.uint8)},
        {"pts": 33, "time": 1.5, "score": 0.7, "thumbnail": np.full((27, 48), 255, dtype=np.uint8)},
    ]

    class FailedValidationReader:
        def __init__(self) -> None:
            self.get_frame_calls: list[int] = []

        def validate_sequential_pts(self, _pts, *, metrics):
            return False, "synthetic validation failure"

        def iter_selected_frames(self, *_args, **_kwargs):
            raise AssertionError("failed validation must not start an ordered pass")

        def get_frame(self, pts: int, max_side: int = 960, *, metrics):
            del max_side
            self.get_frame_calls.append(pts)
            metrics["reader_opens"] += 1
            metrics["decoded_frames"] += 1
            return SimpleNamespace(pts=pts, image=image)

    reader = FailedValidationReader()
    _kept, stats = _deduplicate_with_reader(candidates, reader)

    # A/B are thumbnail-dissimilar, so neither needs exact evidence.  C/B are
    # similar; lazy fallback must fetch C first, then B, with no preload of A.
    assert reader.get_frame_calls == [33, 22]
    assert stats["strict_fallback"] is True
    assert stats["sequential_fast_path"] is False
    assert stats["fallback_reason"] == "synthetic validation failure"


def test_partial_ordered_yield_restarts_clean_and_keeps_missing_frame_candidate() -> None:
    from types import SimpleNamespace

    from anime_picker.video import _UnsafeSequentialReadError

    first_pts, second_pts = 101, 202
    thumbnail = np.full((27, 48), 100, dtype=np.uint8)
    candidates = [
        {"pts": first_pts, "time": 1.0, "score": 0.9, "thumbnail": thumbnail},
        {"pts": second_pts, "time": 1.25, "score": 0.8, "thumbnail": thumbnail},
    ]
    image = Image.new("RGB", (160, 90), (50, 80, 120))

    class PartialReader:
        def __init__(self) -> None:
            self.get_frame_calls: list[int] = []

        def validate_sequential_pts(self, _pts, *, metrics):
            return True, "safe synthetic CFR stream"

        def iter_selected_frames(self, pts, **_kwargs):
            requested = list(pts)
            yield SimpleNamespace(pts=requested[0], image=image)
            raise _UnsafeSequentialReadError("synthetic failure after first selected frame")

        def get_frame(self, pts: int, max_side: int = 960, *, metrics):
            del max_side
            self.get_frame_calls.append(pts)
            metrics["reader_opens"] += 1
            metrics["decoded_frames"] += 1
            if pts == first_pts:
                return None
            return SimpleNamespace(pts=pts, image=image)

    reader = PartialReader()
    kept, stats = _deduplicate_with_reader(candidates, reader)

    # The first optimized yield is deliberately discarded before fallback.
    # Its exact lookup is then missing, so both candidates must survive rather
    # than reusing the stale image from the aborted pass.
    assert [item["pts"] for item in kept] == [first_pts, second_pts]
    assert sorted(reader.get_frame_calls) == [first_pts, second_pts]
    assert stats["optimized_attempted"] is True
    assert stats["optimized_used"] is False
    assert stats["strict_fallback"] is True
    assert stats["sequential_fast_path"] is False
    assert "synthetic failure" in str(stats["fallback_reason"])


@dataclass
class Frame:
    image: Image.Image
    time: float
    pts: int


class FakeReader:
    def __init__(self) -> None:
        self.info = type("Info", (), {"duration": 100.0})()
        self.frames = [
            Frame(Image.new("RGB", (100, 100), (255, 0, 0)), 10 + i * 20, 100 + i)
            for i in range(3)
        ] + [Frame(Image.new("RGB", (100, 100), (0, 0, 255)), 90, 103)]

    def sample(self, *_: object, **__: object):
        return iter(self.frames)


class FakeModel:
    threshold = 0.12

    def detect(self, _: Image.Image):
        return [{"box": [10, 10, 88, 88], "score": 0.93, "kind": "person"}]

    def embed(self, images: list[Image.Image]):
        return np.array([[float(np.asarray(im)[0, 0, 0]) / 255.0] for im in images], dtype=np.float32)

    def distance(self, refs: np.ndarray, values: np.ndarray):
        return abs(refs[:, None, 0] - values[None, :, 0])


def _install_synthetic_batch_reader(monkeypatch):
    import anime_picker.video as video_module
    from types import SimpleNamespace

    class Reader:
        export_calls: list[tuple[int, tuple[str, ...]]] = []
        get_frame_calls: list[int] = []

        def __init__(self, path: Path, *, decode_threads: int | None = None):
            self.path = Path(path)
            self.decode_threads = decode_threads
            self.info = SimpleNamespace(duration=2.0, time_base="1/1000", fps=1.0)
            self.frames = [Frame(Image.new("RGB", (96, 54), (110, 40, 25)), 1.0, 1001)]

        @property
        def fingerprint(self):
            from scripts.archive_completed import _fingerprint
            return _fingerprint(self.path)

        def iter_frames(self, start=0.0, end=None, max_side=960):
            stop = self.info.duration if end is None else end
            return (frame for frame in self.frames if start <= frame.time <= stop)

        def get_frame(self, pts: int, max_side: int = 960):
            type(self).get_frame_calls.append(int(pts))
            return next(frame for frame in self.frames if frame.pts == pts)

        def export_pngs(self, destinations):
            missing = set()
            for pts, paths in destinations.items():
                if not isinstance(paths, list):
                    paths = [paths]
                frame = next((frame for frame in self.frames if frame.pts == pts), None)
                if frame is None:
                    missing.add(pts)
                    continue
                for path in paths:
                    path = Path(path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.image.save(path, format="PNG")
                type(self).export_calls.append((pts, tuple(str(path) for path in paths)))
            return missing

    Reader.export_calls.clear()
    Reader.get_frame_calls.clear()
    monkeypatch.setattr(video_module, "VideoReader", Reader)
    return Reader


def test_auto_reference_prefers_character_present_in_multiple_frames() -> None:
    groups = auto_references(FakeReader(), FakeModel(), max_groups=2, sample_count=4)
    assert len(groups) == 1  # A non-recurring background character is ignored.
    assert groups[0].support_frames == 3
    assert len(groups[0].images) == 3
    assert all(group.mode == "automatic" for group in groups)


def test_auto_reference_default_retains_four_valid_crops_from_one_frame() -> None:
    class MultiDetectionModel:
        threshold = 0.12

        def __init__(self) -> None:
            self.embedded = 0

        def detect(self, _: Image.Image):
            return [
                {"box": [5, 5, 45, 45], "score": 0.9, "kind": "person"},
                {"box": [55, 5, 95, 45], "score": 0.89, "kind": "person"},
                {"box": [5, 55, 45, 95], "score": 0.88, "kind": "person"},
                {"box": [55, 55, 95, 95], "score": 0.87, "kind": "person"},
            ]

        def embed(self, images: list[Image.Image]):
            self.embedded += len(images)
            return np.zeros((len(images), 1), dtype=np.float32)

        def distance(self, references: np.ndarray, values: np.ndarray):
            return np.zeros((len(references), len(values)), dtype=np.float32)

    reader = FakeReader()
    reader.sample = lambda *_args, **_kwargs: iter([reader.frames[0]])
    model = MultiDetectionModel()
    groups = auto_references(reader, model, max_groups=2, sample_count=1)
    assert model.embedded == 4
    assert len(groups) == 1


class FakeExporter:
    def export_png(self, pts: int, path: Path) -> None:
        Image.new("RGB", (24, 12), (pts % 256, 7, 9)).save(path, format="PNG")


def test_native_export_is_exact_pts_and_unmodified_when_logo_off(tmp_path: Path) -> None:
    exported, warnings = _export_character(
        FakeExporter(),
        [{"pts": 1234, "time": 10.0, "bbox": [0, 0, 0.5, 1], "score": 0.85}],
        tmp_path,
        logo_removal=False,
    )
    assert warnings == []
    assert exported[0]["pts"] == 1234
    assert exported[0]["logo_cleanup"]["requested"] is False
    with Image.open(tmp_path / exported[0]["file"]) as image:
        assert image.size == (24, 12)
        assert image.convert("RGB").getpixel((0, 0)) == (1234 % 256, 7, 9)


def test_reference_file_grouping_is_exact_per_video(tmp_path: Path) -> None:
    video = tmp_path / "Episode 1.mp4"
    video.write_bytes(b"")
    refs = tmp_path / "refs"
    refs.mkdir()
    for name in ("Episode 1.png", "Episode 1__side.png", "Episode 10.png", "Episode 1 extra.png"):
        (refs / name).write_bytes(b"")
    names = [ref.name for ref in reference_files(video, refs, None)]
    assert names == ["Episode 1.png", "Episode 1__side.png"]


def test_dry_run_never_creates_output_or_reads_models(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "one.mp4").write_bytes(b"not an actual video")
    outputs = tmp_path / "outputs"
    code = run(["--source", str(tmp_path), "--output-root", str(outputs), "--dry-run"])
    assert code == 0
    assert not outputs.exists()


def test_runner_refreshes_gpu_diagnostics_after_inference(tmp_path: Path, monkeypatch) -> None:
    import json
    import sys
    from types import SimpleNamespace

    import anime_picker.models as models_module
    import scripts.batch_source as batch_source

    source = tmp_path / "source"
    source.mkdir()
    (source / "episode.mp4").write_bytes(b"fake video")
    reference = tmp_path / "validated.png"
    Image.new("RGB", (16, 16), (23, 99, 201)).save(reference)

    class FakeModelService:
        revision = "fixture-model-revision"

        def __init__(self, **_kwargs) -> None:
            self.diagnostics_calls = 0

        def diagnostics(self) -> dict[str, object]:
            self.diagnostics_calls += 1
            active = self.diagnostics_calls > 1
            providers = ["CUDAExecutionProvider"] if active else []
            return {
                "ready": True,
                "onnx_active_providers": providers,
                "cuda_execution_provider_active": active,
            }

        def close(self) -> dict[str, object]:
            return {"performed": True}

    monkeypatch.setattr(models_module, "ModelService", FakeModelService)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(batch_source, "_gpu_memory_snapshot", lambda: {"available": False})
    state_root = tmp_path / ".local" / "batch-state"

    def fake_run_jobs(args, _videos, _references, _wanted, _avoided, _model, _revision, diagnostics, _gpu_before):
        report_path = args.state_root / "batch_run_fixture.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        return {
            "status": "completed",
            "telemetry": {"model_diagnostics": dict(diagnostics)},
        }, report_path, True

    monkeypatch.setattr(batch_source, "_run_jobs", fake_run_jobs)
    output = tmp_path / "outputs"
    assert batch_source.run([
        "--source", str(source),
        "--output-root", str(output),
        "--state-root", str(state_root),
        "--cache-root", str(tmp_path / "cache"),
        "--reference", str(reference),
        "--mode", "character",
        "--no-archive",
    ]) == 0

    report = json.loads((state_root / "batch_run_fixture.json").read_text(encoding="utf-8"))
    telemetry = report["telemetry"]
    assert telemetry["model_diagnostics_before_inference"]["cuda_execution_provider_active"] is False
    assert telemetry["model_diagnostics_after_inference"]["cuda_execution_provider_active"] is True
    assert telemetry["model_diagnostics"]["onnx_active_providers"] == ["CUDAExecutionProvider"]
    assert telemetry["cleanup"]["performed"] is True


def test_headless_runner_uses_all_mode_resumes_and_archives(tmp_path: Path, monkeypatch) -> None:
    import json
    from types import SimpleNamespace

    import anime_picker.video as video_module

    source = tmp_path / "source"
    source.mkdir()
    ref_dir = source / "refs"
    ref_dir.mkdir()
    for stem in ("First", "Second"):
        (source / f"{stem}.mp4").write_bytes(b"fake video")
        (source / f"{stem}.vtt").write_text("WEBVTT\n", encoding="utf-8")
        Image.new("RGB", (16, 16), (23, 99, 201)).save(ref_dir / f"{stem}.png")

    class Reader:
        def __init__(self, file, *, decode_threads: int | None = None):
            self.path = Path(file)
            self.decode_threads = decode_threads
            self.info = SimpleNamespace(duration=10.0, time_base="1/1000", fps=24.0)
            self.frames = [
                Frame(Image.new("RGB", (96, 54), (120, 50, 30)), 7.001, 7001),
                Frame(Image.new("RGB", (96, 54), (120, 50, 30)), 7.501, 7501),
            ]

        @property
        def fingerprint(self):
            from scripts.archive_completed import _fingerprint
            return _fingerprint(self.path)

        def iter_frames(self, start=0.0, end=None, max_side=960):
            stop = self.info.duration if end is None else end
            return (frame for frame in self.frames if start <= frame.time <= stop)

        def export_pngs(self, destinations):
            missing = set()
            for pts, paths in destinations.items():
                if not isinstance(paths, list):
                    paths = [paths]
                if pts not in {frame.pts for frame in self.frames}:
                    missing.add(pts)
                    continue
                for path in paths:
                    Image.new("RGB", (40, 30), (pts % 255, 0, 0)).save(path, format="PNG")
            return missing

    monkeypatch.setattr(video_module, "VideoReader", Reader)
    out = tmp_path / "outputs"
    state_root = tmp_path / ".local" / "batch-state"
    argv = [
        "--source", str(source), "--reference-dir", str(ref_dir),
        "--mode", "all",
        "--output-root", str(out), "--state-root", str(state_root), "--cache-root", str(tmp_path / "cache"),
        "--no-logo-removal", "--no-archive",
    ]
    assert run(argv) == 0
    reports = sorted(state_root.glob("batch_run_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["output_root"] == str(out.resolve())
    assert report["state_root"] == str(state_root.resolve())
    assert report["params"]["mode"] == "all"
    assert report["params"]["selection_revision"]
    assert report["params"]["dedup_revision"]
    assert report["params"]["output_manifest_revision"]
    assert report["telemetry"]["model_loaded"] is False
    assert [job["status"] for job in report["jobs"]] == ["completed", "completed"]
    for job in report["jobs"]:
        images_directory = Path(job["images_directory"])
        assert images_directory == out / Path(job["video"]).stem / "images"
        assert len(list(images_directory.glob("*.png"))) == job["png_count"]
        assert all(path.suffix == ".png" for path in images_directory.iterdir())
        assert job["public_files"][0]["file"].startswith(Path(job["video"]).stem + "/images/")
        exported = Path(job["output"]) / job["saved_files"][0]
        assert exported.exists()
        with Image.open(exported) as image:
            assert image.size == (40, 30)
        assert len(job["outputs"]) == len(job["saved_files"])
        assert job["outputs"][0]["pts"] == 7001
        marker = json.loads((Path(job["output"]) / "batch-result.json").read_text(encoding="utf-8"))
        assert _verified_completed_outputs(Path(job["output"]), marker, job["fingerprint"], job["config_hash"])
        wrong_pts_marker = dict(marker)
        wrong_pts_marker["outputs"] = [dict(item) for item in marker["outputs"]]
        wrong_pts_marker["outputs"][0]["pts"] += 1
        assert not _verified_completed_outputs(Path(job["output"]), wrong_pts_marker, job["fingerprint"], job["config_hash"])
        metadata = json.loads((Path(job["output"]) / "frames" / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["exports"][0]["source_pts"] == 7001
        assert metadata["exports"][0]["sha256"] == job["outputs"][0]["sha256"]
        export = metadata["exports"][0]
        assert export["identity"] is None
        assert isinstance(export["quality"], (int, float)) and not isinstance(export["quality"], bool)
        assert 0.0 <= export["quality"] <= 1.0
        scan_diagnostics = export["quality_metrics"]
        assert set(scan_diagnostics) == {"scan_sharpness", "brightness", "contrast", "local_change", "scene_change"}
        assert all(isinstance(value, (int, float)) for value in scan_diagnostics.values())

    # A decodable but changed PNG invalidates the digest-bound PTS manifest and
    # is regenerated from that exact source PTS on resume.
    first_output = Path(report["jobs"][0]["output"]) / report["jobs"][0]["saved_files"][0]
    Image.new("RGB", (12, 8), (1, 2, 3)).save(first_output)
    tampered_marker = json.loads((first_output.parent.parent / "batch-result.json").read_text(encoding="utf-8"))
    assert not _verified_completed_outputs(first_output.parent.parent, tampered_marker, report["jobs"][0]["fingerprint"], report["jobs"][0]["config_hash"])

    assert run(argv) == 0
    latest = max(state_root.glob("batch_run_*.json"), key=lambda p: p.stat().st_mtime_ns)
    skipped_report = json.loads(latest.read_text(encoding="utf-8"))
    assert [job["status"] for job in skipped_report["jobs"]] == ["completed", "skipped"]
    with Image.open(first_output) as image:
        assert image.size == (40, 30)
        assert image.convert("RGB").getpixel((0, 0)) == (7001 % 255, 0, 0)

    assert run(argv) == 0
    latest = max(state_root.glob("batch_run_*.json"), key=lambda p: p.stat().st_mtime_ns)
    skipped_report = json.loads(latest.read_text(encoding="utf-8"))
    assert [job["status"] for job in skipped_report["jobs"]] == ["skipped", "skipped"]

    # Only verified complete pairs move, including exact-stem subtitles.
    assert run([option for option in argv if option != "--no-archive"]) == 0
    for stem in ("First", "Second"):
        assert not (source / f"{stem}.mp4").exists()
        assert not (source / f"{stem}.vtt").exists()
        assert (tmp_path / "completed" / f"{stem}.mp4").is_file()
        assert (tmp_path / "completed" / f"{stem}.vtt").is_file()
    archived_report = max(state_root.glob("batch_run_*.json"), key=lambda p: p.stat().st_mtime_ns)
    final = json.loads(archived_report.read_text(encoding="utf-8"))
    assert final["status"] == "completed"
    assert final["archive"]["videos_archived"] == 2
    assert all(item["public_pngs_verified"] == 1 for item in final["archive"]["jobs"])
    assert {item.name for item in out.iterdir()} == {"First", "Second"}


def test_interrupted_marker_regenerates_valid_but_unmanifested_png(tmp_path: Path, monkeypatch) -> None:
    import json

    Reader = _install_synthetic_batch_reader(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "Episode.mp4").write_bytes(b"bounded synthetic source")
    output_root = tmp_path / "outputs"
    state_root = tmp_path / ".local" / "batch-state"
    argv = [
        "--source", str(source), "--mode", "all", "--output-root", str(output_root),
        "--state-root", str(state_root),
        "--cache-root", str(tmp_path / "cache"), "--no-logo-removal", "--no-archive",
    ]
    assert run(argv) == 0
    report = json.loads(max(state_root.glob("batch_run_*.json"), key=lambda path: path.stat().st_mtime_ns).read_text(encoding="utf-8"))
    job = report["jobs"][0]
    folder = Path(job["output"])
    output = folder / job["saved_files"][0]
    marker_path = folder / "batch-result.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["status"] = "interrupted"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    (folder / "frames" / "metadata.json").unlink()
    Image.new("RGB", (7, 5), (1, 2, 3)).save(output)

    assert run(argv) == 0
    assert len(Reader.export_calls) == 2
    assert Reader.export_calls[-1][0] == 1001
    with Image.open(output) as image:
        assert image.size == (96, 54)
        assert image.convert("RGB").getpixel((0, 0)) == (110, 40, 25)
    final_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert final_marker["status"] == "completed"
    assert _verified_completed_outputs(folder, final_marker, job["fingerprint"], job["config_hash"])


def test_tampered_original_sidecar_is_regenerated_without_replacing_verified_output(tmp_path: Path, monkeypatch) -> None:
    import json

    import anime_picker.server as server_module

    Reader = _install_synthetic_batch_reader(monkeypatch)
    logo_calls = []

    def process_logo(image, _regions, *, original_relative_path):
        logo_calls.append(original_relative_path)
        return image.copy(), {"requested": True, "applied": False, "status": "no_profile_match"}, []

    monkeypatch.setattr(server_module, "_process_logo_frame", process_logo)
    source = tmp_path / "source"
    source.mkdir()
    (source / "Episode.mp4").write_bytes(b"bounded synthetic source with original")
    output_root = tmp_path / "outputs"
    state_root = tmp_path / ".local" / "batch-state"
    argv = ["--source", str(source), "--mode", "all", "--output-root", str(output_root), "--state-root", str(state_root), "--cache-root", str(tmp_path / "cache"), "--no-archive"]

    assert run(argv) == 0
    report = json.loads(max(state_root.glob("batch_run_*.json"), key=lambda path: path.stat().st_mtime_ns).read_text(encoding="utf-8"))
    job = report["jobs"][0]
    folder = Path(job["output"])
    output = folder / job["saved_files"][0]
    marker_path = folder / "batch-result.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    output_digest = _file_digest(output)
    original = folder / "frames" / "originals" / output.name
    assert len(Reader.export_calls) == 1
    assert len(logo_calls) == 1
    assert marker["outputs"][0]["original_sha256"] == _file_digest(original)

    Image.new("RGB", (8, 6), (200, 3, 4)).save(original)
    assert not _verified_completed_outputs(folder, marker, job["fingerprint"], job["config_hash"])
    assert run(argv) == 0

    assert len(Reader.export_calls) == 2
    assert len(logo_calls) == 1  # The verified processed PNG was preserved.
    assert _file_digest(output) == output_digest
    updated_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert _file_digest(original) == updated_marker["outputs"][0]["original_sha256"]
    assert updated_marker["outputs"][0]["original_file"] == "frames/originals/" + output.name
    assert _verified_completed_outputs(folder, updated_marker, job["fingerprint"], job["config_hash"])


def test_malformed_completed_exports_are_invalid_and_rebuilt(tmp_path: Path, monkeypatch) -> None:
    import json

    Reader = _install_synthetic_batch_reader(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "Episode.mp4").write_bytes(b"bounded synthetic malformed-metadata source")
    output_root = tmp_path / "outputs"
    state_root = tmp_path / ".local" / "batch-state"
    argv = [
        "--source", str(source), "--mode", "all", "--output-root", str(output_root),
        "--state-root", str(state_root),
        "--cache-root", str(tmp_path / "cache"), "--no-logo-removal", "--no-archive",
    ]
    assert run(argv) == 0
    report = json.loads(max(state_root.glob("batch_run_*.json"), key=lambda path: path.stat().st_mtime_ns).read_text(encoding="utf-8"))
    job = report["jobs"][0]
    folder = Path(job["output"])
    metadata_path = folder / "frames" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["exports"] = [None]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    marker = json.loads((folder / "batch-result.json").read_text(encoding="utf-8"))

    assert not _verified_completed_outputs(folder, marker, job["fingerprint"], job["config_hash"])
    assert run(argv) == 0
    assert len(Reader.export_calls) == 2
    repaired_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert isinstance(repaired_metadata["exports"][0], dict)
    repaired_marker = json.loads((folder / "batch-result.json").read_text(encoding="utf-8"))
    assert _verified_completed_outputs(folder, repaired_marker, job["fingerprint"], job["config_hash"])


def test_safe_output_path_rejects_symlink_escape(tmp_path: Path) -> None:
    folder = tmp_path / "output"
    folder.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = folder / "frames"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    assert _safe_output_path(folder, "frames/metadata.json") is None


def test_default_mode_is_cute_and_does_not_auto_select_refs(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "Episode.mp4").write_bytes(b"not a real video")
    refs = source / "refs"
    refs.mkdir()
    Image.new("RGB", (16, 16), (20, 20, 20)).save(refs / "Episode.png")
    outputs = tmp_path / "outputs"
    assert _args([]).mode == "cute"
    assert _args([]).cut_quality_min == pytest.approx(0.55)
    assert run(["--source", str(source), "--output-root", str(outputs), "--dry-run"]) == 0
    assert "mode=cute" in capsys.readouterr().out
    assert run(["--source", str(source), "--mode", "auto-topn", "--top-n", "13", "--dry-run"]) == 0
    message = capsys.readouterr().out
    assert "mode=auto-topn" in message
    assert "top-13 identity subset (not all identities)" in message
    assert not outputs.exists()


def _install_cute_batch_pipeline(monkeypatch, *, fail_tagging: bool = False):
    import sys
    from types import SimpleNamespace
    from PIL import ImageDraw

    import anime_picker.batch_analysis as analysis_module
    import anime_picker.models as models_module
    import scripts.batch_source as batch_module

    monkeypatch.setenv("ONNX_MODE", "gpu")

    Reader = _install_synthetic_batch_reader(monkeypatch)
    original_reader_init = Reader.__init__

    def reader_init(self, path: Path, *, decode_threads: int | None = None):
        original_reader_init(self, path, decode_threads=decode_threads)
        image = Image.new("RGB", (96, 54), (90, 130, 180))
        draw = ImageDraw.Draw(image)
        draw.ellipse((28, 5, 68, 45), fill=(245, 195, 170), outline=(30, 20, 30), width=2)
        draw.ellipse((38, 20, 42, 24), fill=(20, 20, 30))
        draw.ellipse((54, 20, 58, 24), fill=(20, 20, 30))
        draw.arc((43, 27, 54, 38), start=10, end=170, fill=(75, 25, 40), width=2)
        self.frames = [Frame(image, 1.0, 1001)]

    monkeypatch.setattr(Reader, "__init__", reader_init)
    monkeypatch.setattr(
        analysis_module,
        "scan_video",
        lambda *_args, **_kwargs: (
            [SimpleNamespace(time=1.0, sharpness=10.0)],
            {"frames_decoded": 1, "frames_total": 1, "cache_hits": 1},
            "fixture-scan-key",
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "select_candidates",
        lambda *_args, **_kwargs: (
            [{
                "pts": 1001,
                "time": 1.0,
                "scan_sharpness": 10.0,
                "score": 0.8,
                "reasons": ["coverage_anchor"],
                "thumbnail": np.zeros((27, 48), dtype=np.uint8),
            }],
            {"selected": 1},
        ),
    )

    class FakeModelService:
        require_gpu = True
        revision = "fixture-tag-model-v1"
        identity_revision = "fixture-detector-model-v1"
        tag_revision = "fixture-tag-model-v1:cuda-fp32"
        _tagger = SimpleNamespace(device="cuda:0")
        tag_many_calls = 0
        detector_calls = 0
        fallback_tag_calls = 0

        def __init__(self, **kwargs):
            assert kwargs["require_gpu"] is True

        def diagnostics(self):
            return {
                "ready": True,
                "require_gpu": True,
                "tag_revision": self.tag_revision,
                "tag_precision": "float32",
                "torch_cuda_available": True,
                "onnx_active_providers": [],
            }

        def tag_many(self, images, batch_size=8):
            type(self).tag_many_calls += 1
            assert 1 <= len(images) <= batch_size
            if fail_tagging:
                raise RuntimeError("GPU tag inference failed; CPU fallback forbidden")
            return [{"general": {"1girl": 0.91, "smile": 0.62}} for _ in images]

        def _prepare_runtime(self):
            return None

        def detect(self, _image):
            type(self).detector_calls += 1
            return [
                {"box": [16, 3, 78, 49], "score": 0.8, "kind": "person"},
                {"box": [28, 5, 68, 45], "score": 0.88, "kind": "head"},
            ]

        def tag(self, _image):
            type(self).fallback_tag_calls += 1
            raise AssertionError("cute mode must never retry tags one image at a time")

        def close(self):
            return {"performed": True}

    FakeModelService.tag_many_calls = 0
    FakeModelService.detector_calls = 0
    FakeModelService.fallback_tag_calls = 0
    monkeypatch.setattr(models_module, "ModelService", FakeModelService)
    monkeypatch.setattr(batch_module, "_gpu_memory_snapshot", lambda: {"available": True, "device": 0})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    return Reader, FakeModelService


def test_cute_mode_curates_before_export_and_skips_verified_resume(tmp_path: Path, monkeypatch) -> None:
    import json

    Reader, FakeModel = _install_cute_batch_pipeline(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "Episode.mp4").write_bytes(b"cute mode fixture")
    output_root = tmp_path / "outputs"
    state_root = tmp_path / ".local" / "batch-state"
    argv = [
        "--source", str(source), "--output-root", str(output_root),
        "--state-root", str(state_root), "--cache-root", str(tmp_path / "cache"),
        "--no-logo-removal", "--no-archive", "--cut-quality-min", "0",
    ]

    assert run(argv) == 0
    first_report_path = max(state_root.glob("batch_run_*.json"), key=lambda path: path.stat().st_mtime_ns)
    first_report = json.loads(first_report_path.read_text(encoding="utf-8"))
    job = first_report["jobs"][0]
    assert first_report["params"]["mode"] == "cute"
    assert job["status"] == "completed"
    assert job["analysis"]["curation"]["tagger_revision"] == FakeModel.tag_revision
    assert job["analysis"]["curation"]["ccip_used"] is False
    assert job["analysis"]["curation"]["detector_used"] is True
    assert job["analysis"]["curation"]["detector_calls"] == 1
    assert job["analysis"]["curation"]["detector_cache_misses"] == 1
    private = Path(job["output"])
    curation_path = Path(job["analysis"]["curation"]["report"])
    curation = json.loads(curation_path.read_text(encoding="utf-8"))
    assert curation["identity_id_emitted"] is False
    assert curation["report"]["summary"]["character_present"] == 1
    assert list((private / "curation-inputs").glob("*.png"))
    public_images = Path(job["images_directory"])
    assert len(list(public_images.glob("*.png"))) == 1
    metadata = json.loads((private / "character_01" / "metadata.json").read_text(encoding="utf-8"))
    exported = metadata["exports"][0]
    assert exported["character_presence"] is True
    assert exported["presence_score"] == pytest.approx(0.91)
    assert exported["cut_quality"] > 0
    assert "identity_id" not in exported and "identity" not in exported
    assert Reader.export_calls == [(1001, (str(private / "character_01" / "frame_pts_1001.png"),))]

    assert run(argv) == 0
    resumed_path = max(state_root.glob("batch_run_*.json"), key=lambda path: path.stat().st_mtime_ns)
    resumed = json.loads(resumed_path.read_text(encoding="utf-8"))
    assert resumed["jobs"][0]["status"] == "skipped"
    assert FakeModel.tag_many_calls == 1
    assert FakeModel.detector_calls == 1
    assert len(Reader.export_calls) == 1

    assert run([*argv, "--force"]) == 0
    forced_path = max(state_root.glob("batch_run_*.json"), key=lambda path: path.stat().st_mtime_ns)
    forced = json.loads(forced_path.read_text(encoding="utf-8"))
    forced_job = forced["jobs"][0]
    assert forced_job["status"] == "completed"
    assert forced_job["analysis"]["curation"]["detector_cache_hits"] == 1
    assert forced_job["analysis"]["curation"]["detector_cache_misses"] == 0
    assert FakeModel.tag_many_calls == 1
    assert FakeModel.detector_calls == 1
    assert len(Reader.export_calls) == 2


def test_cute_v1_completion_is_reprocessed_under_v9_revision(tmp_path: Path, monkeypatch, capsys) -> None:
    import json

    import scripts.batch_source as batch_module

    Reader, FakeModel = _install_cute_batch_pipeline(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "Episode.mp4").write_bytes(b"cute revision fixture")
    output_root = tmp_path / "outputs"
    state_root = tmp_path / ".local" / "batch-state"
    argv = [
        "--source", str(source), "--output-root", str(output_root),
        "--state-root", str(state_root), "--cache-root", str(tmp_path / "cache"),
        "--no-logo-removal", "--no-archive", "--cut-quality-min", "0",
    ]

    with monkeypatch.context() as legacy:
        legacy.setattr(batch_module, "CUTE_SELECTION_REVISION", batch_module.CUTE_SELECTION_REVISION_V1)
        legacy.setattr(batch_module, "CUTE_CURATION_REVISION", batch_module.CUTE_CURATION_REVISION_V1)
        assert run(argv) == 0
    old_report_path = max(state_root.glob("batch_run_*.json"), key=lambda path: path.stat().st_mtime_ns)
    old_job = json.loads(old_report_path.read_text(encoding="utf-8"))["jobs"][0]
    old_folder = Path(old_job["output"])
    old_marker = json.loads((old_folder / "batch-result.json").read_text(encoding="utf-8"))
    old_curation = json.loads((old_folder / "curation-report.json").read_text(encoding="utf-8"))
    assert old_job["status"] == "completed"
    assert Reader.get_frame_calls == [1001]
    assert old_job["config_hash"] == old_marker["config_hash"]
    assert _verified_completed_outputs(
        old_folder, old_marker, old_job["fingerprint"], old_job["config_hash"]
    )
    assert old_curation["revision"] == batch_module.CUTE_CURATION_REVISION_V1

    assert run(argv) == 0
    new_report_path = max(state_root.glob("batch_run_*.json"), key=lambda path: path.stat().st_mtime_ns)
    new_job = json.loads(new_report_path.read_text(encoding="utf-8"))["jobs"][0]
    new_folder = Path(new_job["output"])
    new_curation = json.loads((new_folder / "curation-report.json").read_text(encoding="utf-8"))
    assert new_job["status"] == "completed"
    assert new_job["analysis"]["curation"]["revision"] == batch_module.CUTE_CURATION_REVISION
    assert new_job["config_hash"] != old_job["config_hash"]
    assert new_curation["revision"] == batch_module.CUTE_CURATION_REVISION
    assert batch_module.CUTE_CURATION_REVISION.endswith("v9")
    assert not _verified_completed_outputs(
        old_folder, old_marker, old_job["fingerprint"], new_job["config_hash"]
    )
    assert FakeModel.tag_many_calls == 1  # The stable PixAI tag cache is still reused.
    assert Reader.get_frame_calls == [1001]  # Exact-PTS curation PNG was linked from the prior run.
    assert len(Reader.export_calls) == 2
    output = capsys.readouterr().out
    assert "[curation] candidates 0/1" in output
    assert "[curation] candidates 1/1" in output


def test_cute_content_quality_can_outrank_sharpness_without_double_weighting(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from PIL import ImageFilter
    from PIL import ImageDraw

    def face(background: tuple[int, int, int]) -> Image.Image:
        image = Image.new("RGB", (96, 72), background)
        draw = ImageDraw.Draw(image)
        draw.ellipse((28, 8, 68, 58), fill=(240, 195, 170), outline=(30, 20, 30), width=2)
        draw.ellipse((38, 25, 42, 29), fill=(20, 20, 30))
        draw.ellipse((54, 25, 58, 29), fill=(20, 20, 30))
        draw.arc((43, 34, 54, 45), start=10, end=170, fill=(75, 25, 40), width=2)
        return image

    high_sharpness_image = face((25, 25, 25))
    lower_sharpness_image = face((190, 125, 160)).filter(ImageFilter.GaussianBlur(2.0))
    frames = {
        1001: Frame(high_sharpness_image, 1.0, 1001),
        5001: Frame(lower_sharpness_image, 5.0, 5001),
    }

    class Reader:
        def get_frame(self, pts: int, max_side: int = 960):
            return frames[pts]

    class RankingModel:
        require_gpu = True
        tag_revision = "fixture-ranker-v1"
        _tagger = SimpleNamespace(device="cuda:0")

        def detect(self, _image):
            return [{"box": [5, 4, 90, 50], "score": 0.8, "kind": "person"}]

        def tag_many(self, images, batch_size=8):
            result = []
            for image in images:
                if image.getpixel((0, 0))[0] < 100:
                    result.append({"general": {"1girl": 0.9, "frown": 0.9, "from_behind": 0.9}})
                else:
                    result.append({
                        "general": {
                            "1girl": 0.9,
                            "smile": 0.9,
                            "portrait": 0.9,
                            "centered_composition": 0.9,
                            "looking_at_viewer": 0.9,
                        }
                    })
            return result

    candidates = [
        _cute_candidate_for_curation(
            {"pts": 1001, "time": 1.0, "reasons": ["coverage_anchor"]}, 1.0
        ),
        _cute_candidate_for_curation(
            {"pts": 5001, "time": 5.0, "reasons": ["coverage_anchor"]}, 0.0
        ),
    ]
    kept, audit, _stats = _curate_character_candidates(
        RankingModel(), Reader(), candidates,
        folder=tmp_path / "run",
        cache_root=tmp_path / "cache",
        source_fingerprint="fixture-source-fingerprint",
        tagger_revision=RankingModel.tag_revision,
        diagnostics={"tag_precision": "float32", "torch_cuda_available": True},
        minimum_cut_quality=0.0,
    )

    by_pts = {item["pts"]: item for item in kept}
    assert set(by_pts) == {1001, 5001}
    sharp_bad = by_pts[1001]["quality_components"]
    cute_less_sharp = by_pts[5001]["quality_components"]
    assert sharp_bad["technical"] == 1.0
    assert cute_less_sharp["technical"] == 0.0
    assert sharp_bad["source"] == cute_less_sharp["source"] == 0.5
    assert by_pts[5001]["cut_quality"] > by_pts[1001]["cut_quality"]
    assert audit["ccip_used"] is False


def test_cute_gpu_tagging_failure_does_not_retry_cpu_or_publish_partial_output(tmp_path: Path, monkeypatch) -> None:
    import json

    _Reader, FakeModel = _install_cute_batch_pipeline(monkeypatch, fail_tagging=True)
    source = tmp_path / "source"
    source.mkdir()
    (source / "Episode.mp4").write_bytes(b"cute mode failure fixture")
    output_root = tmp_path / "outputs"
    state_root = tmp_path / ".local" / "batch-state"
    assert run([
        "--source", str(source), "--output-root", str(output_root),
        "--state-root", str(state_root), "--cache-root", str(tmp_path / "cache"),
        "--no-logo-removal", "--no-archive",
    ]) == 1
    report_path = max(state_root.glob("batch_run_*.json"), key=lambda path: path.stat().st_mtime_ns)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    job = report["jobs"][0]
    assert job["status"] == "error"
    assert "CPU fallback forbidden" in job["error"]
    assert FakeModel.tag_many_calls == 1
    assert FakeModel.fallback_tag_calls == 0
    assert not list(output_root.rglob("*.png"))


def test_manual_mode_without_reference_fails_instead_of_falling_back_to_auto(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "Episode.mp4").write_bytes(b"not a real video")
    try:
        run(["--source", str(source), "--mode", "manual", "--dry-run"])
    except ValueError as error:
        assert "character mode requires" in str(error)
    else:
        raise AssertionError("manual mode silently accepted a missing reference")


def test_cute_cut_quality_floor_is_part_of_resume_hash(tmp_path: Path, monkeypatch) -> None:
    """A changed --cut-quality-min must never reuse another floor's completed run."""

    import json

    Reader, FakeModel = _install_cute_batch_pipeline(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "Episode.mp4").write_bytes(b"cute floor fixture")
    state_root = tmp_path / ".local" / "batch-state"

    def invoke(floor: str) -> dict:
        argv = [
            "--source", str(source), "--output-root", str(tmp_path / "outputs"),
            "--state-root", str(state_root), "--cache-root", str(tmp_path / "cache"),
            "--no-logo-removal", "--no-archive", "--cut-quality-min", floor,
        ]
        assert run(argv) == 0
        latest = max(state_root.glob("batch_run_*.json"), key=lambda path: path.stat().st_mtime_ns)
        return json.loads(latest.read_text(encoding="utf-8"))

    low = invoke("0")
    assert low["params"]["cut_quality_min"] == 0.0
    assert low["jobs"][0]["status"] == "completed"
    high = invoke("0.05")
    assert high["params"]["cut_quality_min"] == pytest.approx(0.05)
    assert high["jobs"][0]["status"] == "completed"
    assert high["jobs"][0]["config_hash"] != low["jobs"][0]["config_hash"]
    again = invoke("0.05")
    assert again["jobs"][0]["status"] == "skipped"
    assert again["jobs"][0]["config_hash"] == high["jobs"][0]["config_hash"]

