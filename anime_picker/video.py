"""Frame accurate, thread-safe video reading for local MP4 and MKV files.

``VideoReader`` deliberately treats presentation timestamps (PTS) as the
source of truth.  Metadata is read with ``ffprobe`` when it is available and
falls back to PyAV.  A frame's ``time`` is measured in seconds from the
selected video's stream start, while ``Frame.pts`` remains the original
integer PTS.  This matters for variable frame-rate files and codecs that use
B-frames: no synthetic frame number or nominal FPS is used to identify a
frame.

Each operation opens its own PyAV container and closes it before returning,
so readers can be used concurrently and do not retain decoder state between
calls.  ``sample`` seeks to the first requested time and decodes forward,
selecting the first presentation frame at or after each sampling target.  It
returns each PTS and timestamp at most once.  ``get_frame`` and
``export_png`` require an exact PTS and raise :class:`FrameNotFoundError` if
that PTS is absent.  PNG export always uses the native decoded dimensions;
only ``sample`` and an explicitly resized ``get_frame`` result can be
downscaled.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

from PIL import Image


@dataclass(frozen=True)
class VideoInfo:
    """Metadata for the first video stream in a local file."""

    path: str
    width: int
    height: int
    duration: float
    fps: float
    stream_index: int
    time_base: str
    start_time: float


@dataclass
class Frame:
    """A decoded video frame identified by its original presentation PTS."""

    pts: int
    time: float
    image: Image.Image
    stream_index: int = 0


class FrameNotFoundError(ValueError, LookupError):
    """Raised when an exact requested PTS is not present in the video."""


def _import_av() -> Any:
    """Import PyAV lazily so metadata-only use can still use ffprobe."""

    try:
        import av  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyAV is required for frame decoding; install the 'av' package"
        ) from exc
    return av


def _fraction(value: Any, default: Fraction | None = None) -> Fraction | None:
    """Parse a rational metadata value without letting ``0/0`` escape."""

    if value is None:
        return default
    if isinstance(value, Fraction):
        return value if value.denominator else default
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "NONE", "0/0"}:
        return default
    try:
        result = Fraction(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return default
    return result if result.denominator else default


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _rate(stream: dict[str, Any]) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        parsed = _fraction(stream.get(key))
        if parsed is not None and parsed > 0:
            return float(parsed)
    return 0.0


def _subprocess_kwargs() -> dict[str, Any]:
    """Return subprocess options that keep helper consoles hidden on Windows."""

    if os.name == "nt":
        # CREATE_NO_WINDOW is available on supported Windows Python builds;
        # the literal keeps this module compatible with test doubles and
        # older Python distributions as well.
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def _find_ffprobe() -> str | None:
    """Choose ffprobe in a deterministic order on Windows installations.

    The project may run in an environment whose ``PATH`` still points at an
    older ffmpeg build.  An explicit ``ANIME_PICKER_FFPROBE`` path wins,
    followed by the current Scoop ffmpeg installation, and only then the
    ordinary ``PATH`` lookup.  Invalid preferred paths are skipped so a
    machine with no ffprobe still reaches the PyAV metadata fallback.
    """

    preferred = os.environ.get("ANIME_PICKER_FFPROBE", "").strip()
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)

    scoop_roots = []
    configured_scoop = os.environ.get("SCOOP", "").strip()
    if configured_scoop:
        scoop_roots.append(Path(configured_scoop))
    scoop_roots.append(Path.home() / "scoop")
    user_profile = os.environ.get("USERPROFILE", "").strip()
    if user_profile:
        scoop_roots.append(Path(user_profile) / "scoop")

    seen_roots: set[str] = set()
    for root in scoop_roots:
        root_key = os.path.normcase(os.fspath(root))
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        candidates.append(os.fspath(root / "apps" / "ffmpeg" / "current" / "bin" / "ffprobe.exe"))

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            return os.fspath(candidate_path)

    return shutil.which("ffprobe")


def _metadata_from_ffprobe(path: Path) -> dict[str, Any] | None:
    """Read the first video stream from ffprobe's JSON output."""

    executable = _find_ffprobe()
    if not executable:
        return None

    command = [
        executable,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        os.fspath(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            **_subprocess_kwargs(),
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError):
        return None

    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list):
        return None
    stream = next(
        (item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"),
        None,
    )
    if stream is None:
        return None

    width = _integer(stream.get("width"))
    height = _integer(stream.get("height"))
    index = _integer(stream.get("index"))
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    if index is None:
        index = 0

    time_base = _fraction(stream.get("time_base"), Fraction(1, 1)) or Fraction(1, 1)
    start_time = _number(stream.get("start_time"))
    if start_time is None:
        start_pts = _integer(stream.get("start_pts"))
        start_time = float(start_pts * time_base) if start_pts is not None else 0.0

    duration = _number(stream.get("duration"))
    if duration is None:
        duration_ts = _integer(stream.get("duration_ts"))
        if duration_ts is not None:
            duration = float(duration_ts * time_base)
    if duration is None:
        format_data = payload.get("format")
        format_duration = (
            _number(format_data.get("duration"))
            if isinstance(format_data, dict)
            else None
        )
        duration = max(0.0, format_duration - start_time) if format_duration is not None else 0.0

    return {
        "width": width,
        "height": height,
        "duration": max(0.0, duration),
        "fps": _rate(stream),
        "stream_index": index,
        "time_base": f"{time_base.numerator}/{time_base.denominator}",
        "start_time": start_time,
    }


def _metadata_from_pyav(path: Path) -> dict[str, Any]:
    """Read metadata using PyAV when ffprobe is unavailable or incomplete."""

    av = _import_av()
    with av.open(os.fspath(path)) as container:
        streams = [stream for stream in container.streams if getattr(stream, "type", None) == "video"]
        if not streams:
            raise ValueError(f"No video stream found in {path}")
        stream = streams[0]

        time_base = _fraction(getattr(stream, "time_base", None), Fraction(1, 1)) or Fraction(1, 1)
        start_pts = _integer(getattr(stream, "start_time", None))
        start_time = float(start_pts * time_base) if start_pts is not None else 0.0

        duration = None
        stream_duration = _integer(getattr(stream, "duration", None))
        if stream_duration is not None:
            duration = float(stream_duration * time_base)
        if duration is None or duration <= 0:
            container_duration = _number(getattr(container, "duration", None))
            if container_duration is not None:
                # AV_TIME_BASE is the unit used by container.duration.
                av_time_base = getattr(av, "time_base", 1_000_000)
                av_time_base_number = _number(av_time_base) or 1_000_000.0
                duration = max(0.0, container_duration / av_time_base_number - start_time)
        if duration is None:
            duration = 0.0

        rate = getattr(stream, "average_rate", None)
        if rate is None or _fraction(rate) is None or _fraction(rate) <= 0:
            rate = getattr(stream, "base_rate", None)
        fps_fraction = _fraction(rate)
        fps = float(fps_fraction) if fps_fraction is not None and fps_fraction > 0 else 0.0

        width = _integer(getattr(stream, "width", None))
        height = _integer(getattr(stream, "height", None))
        if width is None or height is None or width <= 0 or height <= 0:
            codec_context = getattr(stream, "codec_context", None)
            width = _integer(getattr(codec_context, "width", None))
            height = _integer(getattr(codec_context, "height", None))
        if width is None or height is None or width <= 0 or height <= 0:
            raise ValueError(f"Video stream in {path} has no valid dimensions")

        stream_index = _integer(getattr(stream, "index", None)) or 0
        return {
            "width": width,
            "height": height,
            "duration": max(0.0, duration),
            "fps": fps,
            "stream_index": stream_index,
            "time_base": f"{time_base.numerator}/{time_base.denominator}",
            "start_time": start_time,
        }


def _validate_max_side(max_side: int | None) -> int | None:
    if max_side is None:
        return None
    try:
        value = operator.index(max_side)
    except TypeError as exc:
        raise TypeError("max_side must be a positive integer or None") from exc
    if value <= 0:
        raise ValueError("max_side must be a positive integer or None")
    return int(value)


def _resize_image(image: Any, max_side: int | None) -> Any:
    if max_side is None:
        return image
    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image
    scale = max_side / largest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resampling = getattr(Image, "Resampling", Image)
    return image.resize(size, resampling.LANCZOS)


class VideoReader:
    """Read exact-PTS frames from the first video stream in a local file."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        try:
            candidate = Path(os.fspath(path))
        except TypeError as exc:
            raise TypeError("path must be a string or path-like object") from exc
        if not candidate.exists():
            raise FileNotFoundError(os.fspath(candidate))
        if not candidate.is_file():
            raise ValueError(f"Video path is not a file: {candidate}")

        self._path = candidate.resolve()
        self._fingerprint_lock = threading.Lock()
        self._fingerprint_cache: tuple[int, int, str] | None = None

        metadata = _metadata_from_ffprobe(self._path)
        if metadata is None:
            metadata = _metadata_from_pyav(self._path)
        self._info = VideoInfo(path=os.fspath(self._path), **metadata)
        parsed_time_base = _fraction(self._info.time_base, Fraction(1, 1))
        self._time_base = parsed_time_base or Fraction(1, 1)

    @property
    def info(self) -> VideoInfo:
        """Metadata discovered when the reader was created."""

        return self._info

    @property
    def fingerprint(self) -> str:
        """Return ``size:mtime_ns:sha256`` for the current file contents.

        The digest is cached while size and modification time remain unchanged,
        which keeps repeated cache lookups cheap while still recognizing a
        replaced or edited file.
        """

        with self._fingerprint_lock:
            stat = self._path.stat()
            signature = (int(stat.st_size), int(stat.st_mtime_ns))
            cached = self._fingerprint_cache
            if cached is not None and cached[:2] == signature:
                return cached[2]

            digest = hashlib.sha256()
            with self._path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            value = f"{signature[0]}:{signature[1]}:{digest.hexdigest()}"
            self._fingerprint_cache = (signature[0], signature[1], value)
            return value

    def _open_video(self) -> tuple[Any, Any]:
        av = _import_av()
        container = av.open(os.fspath(self._path))
        try:
            stream = next(
                stream
                for stream in container.streams
                if getattr(stream, "type", None) == "video"
                and _integer(getattr(stream, "index", None)) == self._info.stream_index
            )
        except StopIteration:
            container.close()
            raise ValueError(
                f"Video stream index {self._info.stream_index} is unavailable in {self._path}"
            )
        return container, stream

    @staticmethod
    def _close_video(container: Any) -> None:
        try:
            container.close()
        except Exception:
            # A decoder can report a secondary close error after the useful
            # frames have already been returned; do not mask the caller's
            # result with that cleanup failure.
            pass

    @staticmethod
    def _demux_frames(container: Any, stream: Any) -> Iterator[Any]:
        for packet in container.demux(stream):
            for frame in packet.decode():
                yield frame

    @staticmethod
    def _seek(container: Any, stream: Any, pts: int) -> bool:
        try:
            container.seek(pts, stream=stream, backward=True, any_frame=False)
            return True
        except TypeError:
            # Older PyAV releases do not expose ``any_frame``.
            try:
                container.seek(pts, stream=stream, backward=True)
                return True
            except Exception:
                return False
        except Exception:
            return False

    @staticmethod
    def _frame_pts(frame: Any) -> int | None:
        pts = getattr(frame, "pts", None)
        if pts is None:
            return None
        try:
            return int(pts)
        except (TypeError, ValueError, OverflowError):
            return None

    def _frame(self, decoded: Any, max_side: int | None = None) -> Frame | None:
        pts = self._frame_pts(decoded)
        if pts is None:
            return None
        try:
            image = decoded.to_image()
        except AttributeError as exc:
            raise RuntimeError("PyAV returned a frame without image conversion support") from exc
        # RGB makes sampled images and PNG exports deterministic across PyAV
        # versions whose ``to_image`` mode can otherwise vary by pixel format.
        if getattr(image, "mode", None) != "RGB":
            image = image.convert("RGB")
        image = _resize_image(image, max_side)
        return Frame(
            pts=pts,
            time=float(pts * self._time_base) - self._info.start_time,
            image=image,
            stream_index=self._info.stream_index,
        )

    def _seek_pts_for_time(self, relative_time: float) -> int:
        absolute_time = self._info.start_time + relative_time
        return int(round(absolute_time / float(self._time_base)))

    @staticmethod
    def _coerce_pts(pts: int) -> int:
        try:
            value = operator.index(pts)
        except TypeError as exc:
            raise TypeError("pts must be an integer PTS") from exc
        return int(value)

    @staticmethod
    def _coerce_time(value: float, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be a finite number") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} must be a finite number")
        return result

    def sample(
        self,
        start: float,
        end: float,
        step: float,
        max_side: int | None = 960,
    ) -> Iterator[Frame]:
        """Yield sparse presentation frames between ``start`` and ``end``.

        Times are relative to the video stream start.  For each target in
        ``start, start + step, ...`` the first decoded frame at or after that
        target is selected, with ``end`` inclusive.  The returned PTS and
        times are strictly increasing and never repeated.
        """

        start_value = self._coerce_time(start, "start")
        end_value = self._coerce_time(end, "end")
        step_value = self._coerce_time(step, "step")
        if start_value < 0:
            raise ValueError("start must be >= 0")
        if end_value < start_value:
            raise ValueError("end must be >= start")
        if step_value <= 0:
            raise ValueError("step must be > 0")
        max_side_value = _validate_max_side(max_side)

        def iterator() -> Iterator[Frame]:
            container, stream = self._open_video()
            try:
                seek_succeeded = self._seek(
                    container,
                    stream,
                    self._seek_pts_for_time(start_value),
                )
                # A failed seek simply leaves the decoder at its initial
                # position, which is a correct sequential fallback.
                del seek_succeeded

                next_target = start_value
                seen_pts: set[int] = set()
                last_time = -math.inf
                epsilon = 1e-9
                for decoded in self._demux_frames(container, stream):
                    raw_pts = self._frame_pts(decoded)
                    if raw_pts is None or raw_pts in seen_pts:
                        continue
                    seen_pts.add(raw_pts)
                    relative_time = float(raw_pts * self._time_base) - self._info.start_time
                    if relative_time < start_value - epsilon:
                        continue
                    if relative_time > end_value + epsilon:
                        break
                    if relative_time + epsilon < next_target:
                        continue
                    if relative_time <= last_time + epsilon:
                        continue

                    frame = self._frame(decoded, max_side_value)
                    if frame is None:
                        continue
                    yield frame
                    last_time = frame.time
                    while next_target <= relative_time + epsilon:
                        next_target += step_value
            finally:
                self._close_video(container)

        return iterator()

    def get_frame(self, pts: int, max_side: int | None = None) -> Frame:
        """Decode and return the frame with exactly ``pts``.

        Seeking is attempted first.  If a demuxer cannot seek accurately, a
        fresh sequential pass is used so that an existing PTS is never
        incorrectly reported as missing.
        """

        target_pts = self._coerce_pts(pts)
        max_side_value = _validate_max_side(max_side)
        found: Frame | None = None

        container, stream = self._open_video()
        try:
            seek_succeeded = self._seek(container, stream, target_pts)
            previous_pts: int | None = None
            non_monotonic = False
            for decoded in self._demux_frames(container, stream):
                decoded_pts = self._frame_pts(decoded)
                if decoded_pts is None:
                    continue
                if previous_pts is not None and decoded_pts < previous_pts:
                    non_monotonic = True
                previous_pts = decoded_pts
                if decoded_pts == target_pts:
                    found = self._frame(decoded, max_side_value)
                    break
                # Decoded presentation timestamps are normally monotonic.  A
                # strictly later frame proves this seek pass cannot contain
                # the target; a non-monotonic stream is allowed to continue.
                if seek_succeeded and not non_monotonic and decoded_pts > target_pts:
                    break
        finally:
            self._close_video(container)

        if found is not None:
            return found

        # Reopen from the beginning for exact correctness when seeking was
        # unsupported or a demuxer skipped the target around a B-frame.
        container, stream = self._open_video()
        try:
            for decoded in self._demux_frames(container, stream):
                if self._frame_pts(decoded) == target_pts:
                    found = self._frame(decoded, max_side_value)
                    break
        finally:
            self._close_video(container)

        if found is None:
            raise FrameNotFoundError(
                f"No video frame with exact PTS {target_pts} in {self._path}"
            )
        return found

    def export_png(self, pts: int, destination: str | os.PathLike[str]) -> Frame:
        """Export exact-PTS frame ``pts`` as a native-resolution PNG."""

        try:
            destination_path = Path(os.fspath(destination))
        except TypeError as exc:
            raise TypeError("destination must be a string or path-like object") from exc
        if destination_path.exists() and destination_path.is_dir():
            raise IsADirectoryError(os.fspath(destination_path))

        frame = self.get_frame(pts, max_side=None)
        frame.image.save(os.fspath(destination_path), format="PNG")
        return frame


__all__ = ["Frame", "FrameNotFoundError", "VideoInfo", "VideoReader"]
