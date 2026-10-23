"""Focused behavioural tests for :mod:`anime_picker.video`."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
Image = pytest.importorskip("PIL.Image")

from anime_picker import video as video_module
from anime_picker.video import FrameNotFoundError, VideoReader
from scripts.batch_source import _deduplicate_group_candidates, _deduplicate_with_reader


TIME_BASE = Fraction(1, 1000)
FRAME_SIZE = (72, 48)


def _make_vfr_bframe_video(path: Path) -> list[int]:
    """Encode a tiny VFR file whose pictures are easy to identify by colour."""

    try:
        container = av.open(str(path), mode="w")
        stream = container.add_stream("libx264", rate=30)
    except Exception as exc:  # pragma: no cover - codec availability varies
        pytest.skip(f"libx264 unavailable in this PyAV build: {exc}")

    stream.width, stream.height = FRAME_SIZE
    stream.pix_fmt = "yuv420p"
    stream.time_base = TIME_BASE
    stream.codec_context.options = {
        "bf": "2",
        "g": "3",
        "keyint_min": "3",
        "sc_threshold": "0",
    }
    # Deliberately irregular presentation timestamps in milliseconds.
    timestamps = [0, 100, 260, 430, 700, 910]
    try:
        for index, pts in enumerate(timestamps):
            red = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 0, dtype=np.uint8)
            red[..., 0] = 20 + index * 30
            red[..., 1] = 220 - index * 20
            red[..., 2] = 25 + index * 17
            frame = av.VideoFrame.from_ndarray(red, format="rgb24")
            frame.pts = pts
            frame.time_base = TIME_BASE
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    except Exception as exc:  # pragma: no cover - codec availability varies
        pytest.skip(f"libx264 cannot encode the synthetic fixture: {exc}")
    finally:
        container.close()

    # The muxer is allowed to choose a stream time base, so return the exact
    # presentation PTS values that the decoder exposes instead of the input
    # millisecond hints.  Tests must never infer PTS by rounding a nominal FPS.
    try:
        encoded = av.open(str(path), mode="r")
    except Exception as exc:  # pragma: no cover - codec availability varies
        pytest.skip(f"PyAV cannot reopen the synthetic fixture: {exc}")
    try:
        stream = next(stream for stream in encoded.streams if stream.type == "video")
        return [int(frame.pts) for frame in encoded.decode(stream) if frame.pts is not None]
    finally:
        encoded.close()


def _make_cfr_no_bframes_video(path: Path) -> list[int]:
    """Encode a small constant-cadence H.264 source with reordering disabled."""

    try:
        container = av.open(str(path), mode="w")
        stream = container.add_stream("libx264", rate=10)
    except Exception as exc:  # pragma: no cover - codec availability varies
        pytest.skip(f"libx264 unavailable in this PyAV build: {exc}")

    stream.width, stream.height = FRAME_SIZE
    stream.pix_fmt = "yuv420p"
    stream.time_base = TIME_BASE
    stream.codec_context.options = {"bf": "0", "g": "30", "sc_threshold": "0"}
    timestamps = [0, 100, 200, 300, 400, 500]
    try:
        for index, pts in enumerate(timestamps):
            rgb = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            group = index // 2
            rgb[..., 0] = 25 + group * 55
            rgb[..., 1] = 210 - group * 37
            rgb[..., 2] = 15 + group * 41
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = pts
            frame.time_base = TIME_BASE
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    except Exception as exc:  # pragma: no cover - codec availability varies
        pytest.skip(f"libx264 cannot encode the synthetic fixture: {exc}")
    finally:
        container.close()
    return timestamps


def _make_cfr_bframes_video(path: Path, *, start_pts: int = 0) -> list[int]:
    """Encode constant-cadence H.264 with real B pictures in the decode path."""

    try:
        container = av.open(str(path), mode="w")
        stream = container.add_stream("libx264", rate=10)
    except Exception as exc:  # pragma: no cover - codec availability varies
        pytest.skip(f"libx264 unavailable in this PyAV build: {exc}")

    stream.width, stream.height = FRAME_SIZE
    stream.pix_fmt = "yuv420p"
    stream.time_base = TIME_BASE
    stream.codec_context.options = {
        "bf": "2",
        "g": "30",
        "keyint_min": "30",
        "sc_threshold": "0",
    }
    timestamps = [start_pts + index * 100 for index in range(6)]
    try:
        for index, pts in enumerate(timestamps):
            rgb = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            group = index // 2
            rgb[..., 0] = 25 + group * 55
            rgb[..., 1] = 210 - group * 37
            rgb[..., 2] = 15 + group * 41
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = pts
            frame.time_base = TIME_BASE
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    except Exception as exc:  # pragma: no cover - codec availability varies
        pytest.skip(f"libx264 cannot encode the synthetic fixture: {exc}")
    finally:
        container.close()

    try:
        encoded = av.open(str(path), mode="r")
    except Exception as exc:  # pragma: no cover - codec availability varies
        pytest.skip(f"PyAV cannot reopen the synthetic fixture: {exc}")
    try:
        stream = next(stream for stream in encoded.streams if stream.type == "video")
        return [int(frame.pts) for frame in encoded.decode(stream) if frame.pts is not None]
    finally:
        encoded.close()


def _make_nonzero_start_cfr_video(path: Path) -> list[int]:
    """Encode CFR H.264 whose first presentation timestamp is nonzero."""

    try:
        container = av.open(str(path), mode="w")
        stream = container.add_stream("libx264", rate=10)
    except Exception as exc:  # pragma: no cover - codec availability varies
        pytest.skip(f"libx264 unavailable in this PyAV build: {exc}")

    stream.width, stream.height = FRAME_SIZE
    stream.pix_fmt = "yuv420p"
    stream.time_base = TIME_BASE
    stream.codec_context.options = {"bf": "0", "g": "30", "sc_threshold": "0"}
    timestamps = [1000 + index * 100 for index in range(6)]
    try:
        for index, pts in enumerate(timestamps):
            rgb = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            group = index // 2
            rgb[..., 0] = 35 + group * 45
            rgb[..., 1] = 190 - group * 31
            rgb[..., 2] = 20 + group * 37
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = pts
            frame.time_base = TIME_BASE
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    except Exception as exc:  # pragma: no cover - codec availability varies
        pytest.skip(f"libx264 cannot encode the synthetic fixture: {exc}")
    finally:
        container.close()

    try:
        encoded = av.open(str(path), mode="r")
    except Exception as exc:  # pragma: no cover - codec availability varies
        pytest.skip(f"PyAV cannot reopen the synthetic fixture: {exc}")
    try:
        stream = next(stream for stream in encoded.streams if stream.type == "video")
        return [int(frame.pts) for frame in encoded.decode(stream) if frame.pts is not None]
    finally:
        encoded.close()


@pytest.fixture(params=("mp4", "mkv"))
def vfr_video(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, list[int]]:
    path = tmp_path / f"vfr-bframes.{request.param}"
    return path, _make_vfr_bframe_video(path)


@pytest.fixture
def cfr_video(tmp_path: Path) -> tuple[Path, list[int]]:
    path = tmp_path / "cfr-no-bframes.mp4"
    return path, _make_cfr_no_bframes_video(path)


@pytest.fixture(params=("mp4", "mkv"))
def cfr_bframe_video(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, list[int]]:
    path = tmp_path / f"cfr-bframes.{request.param}"
    return path, _make_cfr_bframes_video(path)


@pytest.fixture
def nonzero_start_cfr_video(tmp_path: Path) -> tuple[Path, list[int]]:
    path = tmp_path / "cfr-nonzero-start.mp4"
    return path, _make_nonzero_start_cfr_video(path)


@pytest.fixture
def combined_bframe_nonzero_cfr_video(tmp_path: Path) -> tuple[Path, list[int]]:
    path = tmp_path / "parity-cfr-bframes-nonzero-start.mp4"
    return path, _make_cfr_bframes_video(path, start_pts=1000)


@pytest.fixture(params=("cfr_bframe", "nonzero_start", "combined_bframe_nonzero"))
def newly_eligible_cfr_video(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> tuple[Path, list[int]]:
    if request.param == "cfr_bframe":
        path = tmp_path / "parity-cfr-bframes.mp4"
        return path, _make_cfr_bframes_video(path)
    if request.param == "nonzero_start":
        path = tmp_path / "parity-cfr-nonzero-start.mp4"
        return path, _make_nonzero_start_cfr_video(path)
    path = tmp_path / "parity-cfr-bframes-nonzero-start.mp4"
    return path, _make_cfr_bframes_video(path, start_pts=1000)


def test_vfr_fixture_proves_h264_b_picture_capability_and_picture_exists(
    vfr_video: tuple[Path, list[int]],
) -> None:
    path, _encoded_pts = vfr_video
    container = av.open(str(path), mode="r")
    try:
        stream = next(stream for stream in container.streams if stream.type == "video")
        assert bool(stream.codec_context.has_b_frames), "fixture must advertise H.264 B-picture capability"
        picture_types = [getattr(frame, "pict_type", None) for frame in container.decode(stream)]
    finally:
        container.close()

    # FFmpeg's AVPictureType.B value is 3.  Checking decoded pictures as well
    # as codec capability prevents a nominal encoder option from faking this
    # fixture's intended coverage.
    def is_b_picture(value: object) -> bool:
        if isinstance(value, str):
            return value.rsplit(".", 1)[-1].upper() == "B"
        try:
            return int(value) == 3
        except (TypeError, ValueError, OverflowError):
            return False

    assert any(is_b_picture(value) for value in picture_types), (
        f"fixture decoded picture types {picture_types!r}, expected a B picture"
    )


def test_cfr_bframe_fixture_is_constant_and_contains_decoded_b_picture(
    cfr_bframe_video: tuple[Path, list[int]],
) -> None:
    path, encoded_pts = cfr_bframe_video
    container = av.open(str(path), mode="r")
    try:
        stream = next(stream for stream in container.streams if stream.type == "video")
        assert bool(stream.codec_context.has_b_frames), "fixture must advertise H.264 B-picture capability"
        decoded = list(container.decode(stream))
        picture_types = [getattr(frame, "pict_type", None) for frame in decoded]
        decoded_pts = [int(frame.pts) for frame in decoded if frame.pts is not None]
    finally:
        container.close()

    assert decoded_pts == encoded_pts
    assert len(decoded_pts) >= 2
    assert len({right - left for left, right in zip(decoded_pts, decoded_pts[1:])}) == 1

    def is_b_picture(value: object) -> bool:
        if isinstance(value, str):
            return value.rsplit(".", 1)[-1].upper() == "B"
        try:
            return int(value) == 3
        except (TypeError, ValueError, OverflowError):
            return False

    assert any(is_b_picture(value) for value in picture_types), (
        f"fixture decoded picture types {picture_types!r}, expected a B picture"
    )


def test_nonzero_start_cfr_with_bframes_is_eligible_for_fast_path(
    combined_bframe_nonzero_cfr_video: tuple[Path, list[int]],
) -> None:
    path, encoded_pts = combined_bframe_nonzero_cfr_video
    container = av.open(str(path), mode="r")
    try:
        stream = next(stream for stream in container.streams if stream.type == "video")
        has_b_frames = bool(stream.codec_context.has_b_frames)
        decoded = list(container.decode(stream))
        picture_types = [getattr(frame, "pict_type", None) for frame in decoded]
    finally:
        container.close()

    def is_b_picture(value: object) -> bool:
        if isinstance(value, str):
            return value.rsplit(".", 1)[-1].upper() == "B"
        try:
            return int(value) == 3
        except (TypeError, ValueError, OverflowError):
            return False

    assert has_b_frames
    assert any(is_b_picture(value) for value in picture_types)
    assert encoded_pts[0] > 0

    reader = VideoReader(path, decode_threads=1)
    assert reader.info.start_time > 0.0
    frames = list(reader.iter_frames(0.0, None, max_side=960))
    assert [frame.pts for frame in frames] == encoded_pts
    safe, reason = reader.validate_sequential_pts([frame.pts for frame in frames])
    assert safe, reason
    selected = list(reader.iter_selected_frames([frame.pts for frame in frames], max_side=960))
    assert [frame.pts for frame in selected] == encoded_pts
    for sequential, exact in zip(selected, frames):
        assert np.array_equal(np.asarray(sequential.image), np.asarray(exact.image))


def test_ordered_selected_pts_matches_exact_lookup_and_converts_selected_only(
    cfr_video: tuple[Path, list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _source_pts = cfr_video
    reader = VideoReader(path)
    decoded_pts = [frame.pts for frame in reader.iter_frames(0.0, None, max_side=None)]
    selected = [decoded_pts[1], decoded_pts[3], decoded_pts[5]]
    stats: dict[str, float | int] = {}
    safe, reason = reader.validate_sequential_pts(selected, metrics=stats)
    assert safe, reason
    cached_stats: dict[str, float | int] = {}
    cached_safe, cached_reason = reader.validate_sequential_pts([decoded_pts[0]], metrics=cached_stats)
    assert cached_safe, cached_reason
    assert cached_stats.get("reader_opens", 0) == 0

    converted: list[int] = []
    original_frame = reader._frame

    def counted_frame(decoded: object, max_side: int | None = None):
        converted.append(int(getattr(decoded, "pts")))
        return original_frame(decoded, max_side)

    monkeypatch.setattr(reader, "_frame", counted_frame)
    sequential = list(reader.iter_selected_frames(selected, max_side=960, metrics=stats))
    assert [frame.pts for frame in sequential] == selected
    assert converted == selected
    assert stats["reader_opens"] == 2
    assert stats["decoded_frames"] == 12  # Six safety frames plus the ordered pass through the last selection.

    for frame in sequential:
        exact = reader.get_frame(frame.pts, max_side=960)
        assert frame.image.size == exact.image.size
        assert np.array_equal(np.asarray(frame.image), np.asarray(exact.image))


def test_reader_dedup_fast_path_matches_legacy_pixels_and_decisions(
    newly_eligible_cfr_video: tuple[Path, list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _source_pts = newly_eligible_cfr_video
    reader = VideoReader(path)
    frames = list(reader.iter_frames(0.0, None, max_side=960))
    thumbnail = np.full((27, 48), 105, dtype=np.uint8)
    candidates = [
        {"pts": frame.pts, "time": frame.time, "score": 0.9 - index * 0.01, "thumbnail": thumbnail}
        for index, frame in enumerate(frames)
    ]

    legacy_pixels: dict[int, np.ndarray] = {}

    def legacy_lookup(pts: int):
        frame = reader.get_frame(pts, max_side=960)
        legacy_pixels[int(pts)] = np.asarray(frame.image).copy()
        return frame.image

    legacy = _deduplicate_group_candidates(
        candidates,
        frame_lookup=legacy_lookup,
    )

    sequential_pixels: dict[int, np.ndarray] = {}
    original_iter_selected = reader.iter_selected_frames

    def capture_sequential(*args: object, **kwargs: object):
        for frame in original_iter_selected(*args, **kwargs):
            sequential_pixels[int(frame.pts)] = np.asarray(frame.image).copy()
            yield frame

    monkeypatch.setattr(reader, "iter_selected_frames", capture_sequential)
    ordered, stats = _deduplicate_with_reader(candidates, reader)

    fields = ("pts", "time", "score", "phash", "dhash", "crop_phash", "crop_dhash")
    assert [{key: item.get(key) for key in fields} for item in ordered] == [
        {key: item.get(key) for key in fields} for item in legacy
    ]
    assert [item["pts"] for item in ordered] == [frames[index].pts for index in (0, 2, 4)]
    assert stats["sequential_fast_path"] is True
    assert stats["strict_fallback"] is False
    assert stats["reader_opens"] == 2
    assert stats["seek_fallbacks"] == 0
    assert stats["pixel_comparisons"] == 3
    assert 0 < stats["max_cached_images"] <= len(candidates)
    assert stats["optimized_used"] is True
    assert stats["dedup_wall_seconds"] == stats["dedup_seconds"]
    assert stats["validation_seconds"] >= 0.0
    assert stats["prefilter_seconds"] >= 0.0
    assert set(sequential_pixels) == set(legacy_pixels)
    for pts in legacy_pixels:
        assert np.array_equal(sequential_pixels[pts], legacy_pixels[pts])


def test_validation_open_oserror_falls_back_to_exact_pts_lookup(
    cfr_video: tuple[Path, list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, source_pts = cfr_video
    reader = VideoReader(path, decode_threads=1)
    original_open = reader._open_video
    original_get_frame = reader.get_frame
    open_calls = 0
    lookup_pts: list[int] = []

    def fail_first_open():
        nonlocal open_calls
        open_calls += 1
        if open_calls == 1:
            raise OSError("synthetic validation open failure")
        return original_open()

    def counted_get_frame(pts: int, *args: object, **kwargs: object):
        lookup_pts.append(int(pts))
        return original_get_frame(pts, *args, **kwargs)

    monkeypatch.setattr(reader, "_open_video", fail_first_open)
    monkeypatch.setattr(reader, "get_frame", counted_get_frame)
    thumbnail = np.full((27, 48), 100, dtype=np.uint8)
    candidates = [
        {"pts": source_pts[0], "time": 0.0, "score": 0.9, "thumbnail": thumbnail},
        {"pts": source_pts[1], "time": 0.1, "score": 0.8, "thumbnail": thumbnail},
    ]

    kept, stats = _deduplicate_with_reader(candidates, reader)

    assert [item["pts"] for item in kept] == source_pts[:2]
    assert open_calls >= 2  # The failed proof is followed by an exact-PTS read.
    assert lookup_pts == [source_pts[1]]
    assert stats["optimized_attempted"] is False
    assert stats["optimized_used"] is False
    assert stats["strict_fallback"] is True
    assert stats["sequential_fast_path"] is False
    assert "OSError" in str(stats["fallback_reason"])


def test_cfr_validation_rejects_a_selected_pts_absent_from_the_source(
    cfr_video: tuple[Path, list[int]],
) -> None:
    path, _source_pts = cfr_video
    reader = VideoReader(path)
    decoded_pts = [frame.pts for frame in reader.iter_frames(0.0, None, max_side=None)]
    assert decoded_pts

    safe, reason = reader.validate_sequential_pts(
        [decoded_pts[0], max(decoded_pts) + 1],
    )

    assert not safe
    assert "missing" in reason


def test_nonzero_start_cfr_preserves_exact_pts_with_timing_proof(
    nonzero_start_cfr_video: tuple[Path, list[int]],
) -> None:
    path, encoded_pts = nonzero_start_cfr_video
    reader = VideoReader(path)
    assert reader.info.start_time > 0.0
    assert encoded_pts

    frames = list(reader.iter_frames(0.0, None, max_side=None))
    assert [frame.pts for frame in frames] == encoded_pts
    assert all(left.pts < right.pts for left, right in zip(frames, frames[1:]))

    selected = [frames[1], frames[-1]]
    for expected in selected:
        exact = reader.get_frame(expected.pts, max_side=None)
        assert exact.pts == expected.pts
        assert np.array_equal(np.asarray(exact.image), np.asarray(expected.image))

    safe, reason = reader.validate_sequential_pts([frame.pts for frame in selected])
    assert safe, reason
    ordered = list(reader.iter_selected_frames([frame.pts for frame in selected], max_side=None))
    assert [frame.pts for frame in ordered] == [frame.pts for frame in selected]


@pytest.mark.parametrize(
    ("pts_values", "reason"),
    [
        ([0, None, 20], "missing"),
        ([0, 10, 10], "duplicate PTS"),
        ([0, 10, 5], "nonmonotonic"),
        ([0, 10, 25], "variable frame cadence"),
    ],
)
def test_ordered_pts_validation_rejects_unproven_streams(
    pts_values: list[int | None], reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    class Decoded:
        def __init__(self, pts: int | None) -> None:
            self.pts = pts
            self.time_base = TIME_BASE

        def to_image(self):
            raise AssertionError("validation must not convert nonselected frames")

    class Container:
        def close(self) -> None:
            pass

    reader = object.__new__(VideoReader)
    reader._info = SimpleNamespace(start_time=0.0, stream_index=0)
    reader._time_base = TIME_BASE
    container = Container()
    stream = SimpleNamespace(
        time_base=TIME_BASE,
        start_time=0,
        codec_context=SimpleNamespace(has_b_frames=0),
    )
    reader._open_video = lambda: (container, stream)
    monkeypatch.setattr(
        VideoReader,
        "_demux_frames",
        staticmethod(lambda _container, _stream: iter(Decoded(value) for value in pts_values)),
    )
    metrics: dict[str, float | int] = {}

    safe, fallback_reason = reader.validate_sequential_pts([0, 10], metrics=metrics)

    assert not safe
    assert reason in fallback_reason
    assert metrics.get("reader_opens", 0) == 1


def test_nonintegral_stream_start_is_rejected_without_int_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    class Container:
        def close(self) -> None:
            pass

    reader = object.__new__(VideoReader)
    reader._info = SimpleNamespace(start_time=1.5)
    reader._time_base = TIME_BASE
    container = Container()
    stream = SimpleNamespace(time_base=TIME_BASE, start_time=1.5, codec_context=SimpleNamespace())
    reader._open_video = lambda: (container, stream)
    monkeypatch.setattr(
        VideoReader,
        "_demux_frames",
        staticmethod(lambda _container, _stream: iter(())),
    )

    # A fractional stream start must not become start PTS 1 through int().
    assert reader._stream_timing(stream) is None
    safe, reason = reader.validate_sequential_pts([0, 100])

    assert not safe
    assert "start" in reason.lower() or "integer" in reason.lower()


@pytest.mark.parametrize("decoded_time_base", [None, Fraction(1, 2000)])
def test_validation_requires_each_decoded_frame_time_base(
    decoded_time_base: Fraction | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    class Decoded:
        def __init__(self, pts: int) -> None:
            self.pts = pts
            self.time_base = decoded_time_base

    class Container:
        def close(self) -> None:
            pass

    reader = object.__new__(VideoReader)
    reader._info = SimpleNamespace(start_time=0.0, stream_index=0)
    reader._time_base = TIME_BASE
    container = Container()
    stream = SimpleNamespace(time_base=TIME_BASE, start_time=0, codec_context=SimpleNamespace())
    reader._open_video = lambda: (container, stream)
    monkeypatch.setattr(
        VideoReader,
        "_demux_frames",
        staticmethod(lambda _container, _stream: iter(Decoded(value) for value in (0, 100, 200))),
    )

    safe, reason = reader.validate_sequential_pts([0, 100])

    assert not safe
    assert "time_base" in reason.lower() or "time base" in reason.lower()


@pytest.mark.parametrize("decoded_time_base", [None, Fraction(1, 2000)])
def test_ordered_pass_rejects_missing_or_changed_decoded_frame_time_base(
    decoded_time_base: Fraction | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    class Decoded:
        def __init__(self, pts: int, time_base: Fraction | None) -> None:
            self.pts = pts
            self.time_base = time_base

        def to_image(self):
            return Image.new("RGB", (2, 2), (10, 20, 30))

    class Container:
        def close(self) -> None:
            pass

    reader = object.__new__(VideoReader)
    reader._info = SimpleNamespace(start_time=0.0, stream_index=0)
    reader._time_base = TIME_BASE
    container = Container()
    stream = SimpleNamespace(time_base=TIME_BASE, start_time=0, codec_context=SimpleNamespace())
    reader._open_video = lambda: (container, stream)
    passes = [
        [Decoded(0, TIME_BASE), Decoded(100, TIME_BASE), Decoded(200, TIME_BASE)],
        [Decoded(0, TIME_BASE), Decoded(100, decoded_time_base), Decoded(200, decoded_time_base)],
    ]
    monkeypatch.setattr(
        VideoReader,
        "_demux_frames",
        staticmethod(lambda _container, _stream: iter(passes.pop(0))),
    )

    safe, reason = reader.validate_sequential_pts([0, 100])
    assert safe, reason

    from anime_picker.video import _UnsafeSequentialReadError

    with pytest.raises(_UnsafeSequentialReadError, match="time[_ ]base"):
        list(reader.iter_selected_frames([0, 100], max_side=None))


def test_vfr_bframe_video_uses_strict_fallback(vfr_video: tuple[Path, list[int]]) -> None:
    path, pts = vfr_video
    reader = VideoReader(path)
    safe, reason = reader.validate_sequential_pts(pts[:2])
    assert not safe
    assert "B-frames" in reason or "variable frame cadence" in reason

    frames = list(reader.iter_frames(0.0, 0.92, max_side=960))
    thumbnail = np.full((27, 48), 105, dtype=np.uint8)
    candidates = [
        {"pts": frame.pts, "time": frame.time, "score": 0.9, "thumbnail": thumbnail}
        for frame in frames[:3]
    ]
    _kept, stats = _deduplicate_with_reader(candidates, reader)
    assert stats["strict_fallback"] is True
    assert stats["sequential_fast_path"] is False
    assert "B-frames" in str(stats["fallback_reason"]) or "variable frame cadence" in str(stats["fallback_reason"])
    assert stats["frame_lookups"] > 0


def test_sample_pts_export_pixels_and_native_dimensions(vfr_video: tuple[Path, list[int]], tmp_path: Path) -> None:
    path, _ = vfr_video
    reader = VideoReader(path)

    sampled = list(reader.sample(0.0, 0.92, 0.19, max_side=None))
    assert sampled
    assert [frame.pts for frame in sampled] == sorted({frame.pts for frame in sampled})
    assert [frame.time for frame in sampled] == sorted({frame.time for frame in sampled})
    assert all(frame.image.size == FRAME_SIZE for frame in sampled)

    for frame in sampled:
        destination = tmp_path / f"{frame.pts}.png"
        exported = reader.export_png(frame.pts, destination)
        with Image.open(destination) as png:
            png.load()
            assert png.size == FRAME_SIZE
            assert np.array_equal(np.asarray(frame.image), np.asarray(png.convert("RGB")))
        assert exported.pts == frame.pts
        assert exported.time == pytest.approx(frame.time, abs=1e-9)


def test_sample_respects_max_side_and_get_frame_is_exact(vfr_video: tuple[Path, list[int]]) -> None:
    path, _ = vfr_video
    reader = VideoReader(path)

    small = list(reader.sample(0.0, 0.92, 0.25, max_side=32))
    assert small
    assert all(max(frame.image.size) <= 32 for frame in small)

    exact = reader.get_frame(small[0].pts)
    assert exact.pts == small[0].pts
    assert exact.image.size == FRAME_SIZE
    assert reader.get_frame(exact.pts, max_side=24).image.size[0] <= 24


def test_open_video_applies_requested_decoder_thread_count(vfr_video: tuple[Path, list[int]]) -> None:
    path, _ = vfr_video
    reader = VideoReader(path, decode_threads=1)
    container, stream = reader._open_video()
    try:
        assert stream.codec_context.thread_count == 1
    finally:
        reader._close_video(container)


def test_iter_frames_covers_every_vfr_pts_and_bulk_export_is_atomic(
    vfr_video: tuple[Path, list[int]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, source_pts = vfr_video
    reader = VideoReader(path)
    frames = list(reader.iter_frames(0.0, 0.92, max_side=None))
    decoded_pts = [frame.pts for frame in frames]
    assert len(decoded_pts) == len(source_pts)
    assert decoded_pts == sorted(set(decoded_pts))
    destinations = {
        frames[index].pts: tmp_path / f"bulk-{frames[index].pts}.png"
        for index in (1, 4)
    }
    opens = 0
    original_open = reader._open_video

    def counted_open() -> tuple[object, object]:
        nonlocal opens
        opens += 1
        return original_open()

    monkeypatch.setattr(reader, "_open_video", counted_open)
    missing = reader.export_pngs(destinations)
    assert missing == set()
    assert opens == 1
    for pts, destination in destinations.items():
        assert destination.is_file()
        assert not destination.with_name(destination.name + ".tmp").exists()
        with Image.open(destination) as image:
            assert image.size == FRAME_SIZE
            expected = next(frame.image for frame in frames if frame.pts == pts)
            assert np.array_equal(np.asarray(image.convert("RGB")), np.asarray(expected))


def test_bulk_export_handles_out_of_order_and_repeated_pts(
    vfr_video: tuple[Path, list[int]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = vfr_video
    reader = VideoReader(path)

    class DecodedFrame:
        def __init__(self, pts: int, color: tuple[int, int, int]) -> None:
            self.pts = pts
            self._image = Image.new("RGB", (2, 2), color)

        def to_image(self) -> Image.Image:
            return self._image.copy()

    class Container:
        def seek(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("synthetic sequential fallback")

        def close(self) -> None:
            pass

    decoded = [
        DecodedFrame(100, (10, 20, 30)),
        DecodedFrame(500, (40, 50, 60)),
        DecodedFrame(100, (70, 80, 90)),
        DecodedFrame(300, (100, 110, 120)),
    ]
    container = Container()
    monkeypatch.setattr(reader, "_open_video", lambda: (container, object()))
    monkeypatch.setattr(
        reader,
        "_demux_frames",
        lambda _container, _stream: iter(decoded),
    )
    first = tmp_path / "first-100.png"
    later = tmp_path / "later-300.png"
    absent = tmp_path / "absent-200.png"

    missing = reader.export_pngs({100: first, 200: absent, 300: later})

    assert missing == {200}
    with Image.open(first) as image:
        assert image.getpixel((0, 0)) == (10, 20, 30)
    with Image.open(later) as image:
        assert image.getpixel((0, 0)) == (100, 110, 120)
    assert not absent.exists()


def test_bulk_export_keeps_existing_file_if_atomic_replace_fails(
    vfr_video: tuple[Path, list[int]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = vfr_video
    reader = VideoReader(path)
    pts = next(reader.iter_frames(0.0, 0.92, max_side=None)).pts
    destination = tmp_path / "existing.png"
    original_content = b"previous output"
    destination.write_bytes(original_content)

    def fail_replace(_source: str | Path, _target: str | Path) -> None:
        raise OSError("synthetic atomic-replace failure")

    monkeypatch.setattr(video_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="atomic-replace"):
        reader.export_pngs({pts: destination})

    assert destination.read_bytes() == original_content
    assert not destination.with_name(destination.name + ".tmp").exists()


def test_invalid_ranges_and_absent_pts(vfr_video: tuple[Path, list[int]]) -> None:
    path, _ = vfr_video
    reader = VideoReader(path)

    with pytest.raises(ValueError, match="end"):
        reader.sample(1.0, 0.0, 0.2)
    with pytest.raises(ValueError, match="step"):
        reader.sample(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        reader.sample(float("nan"), 1.0, 0.2)
    with pytest.raises(ValueError, match="max_side"):
        reader.sample(0.0, 1.0, 0.2, max_side=0)
    with pytest.raises(FrameNotFoundError, match="exact PTS"):
        reader.get_frame(123456789)


def test_fingerprint_is_stable_for_unchanged_file(vfr_video: tuple[Path, list[int]]) -> None:
    path, _ = vfr_video
    reader = VideoReader(path)
    first = reader.fingerprint
    assert first == reader.fingerprint
    assert len(first.split(":")) == 3
    assert reader.info.width == FRAME_SIZE[0]
    assert reader.info.height == FRAME_SIZE[1]
    assert reader.info.duration >= 0
