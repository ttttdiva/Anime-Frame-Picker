"""Focused behavioural tests for :mod:`anime_picker.video`."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
Image = pytest.importorskip("PIL.Image")

from anime_picker.video import FrameNotFoundError, VideoReader


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
    return timestamps


@pytest.fixture(params=("mp4", "mkv"))
def vfr_video(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, list[int]]:
    path = tmp_path / f"vfr-bframes.{request.param}"
    return path, _make_vfr_bframe_video(path)


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
