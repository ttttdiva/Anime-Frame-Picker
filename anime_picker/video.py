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
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, MutableMapping

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
    start_pts: int | None = None


@dataclass
class Frame:
    """A decoded video frame identified by its original presentation PTS."""

    pts: int
    time: float
    image: Image.Image
    stream_index: int = 0


@dataclass(frozen=True)
class _SequentialPtsProof:
    """Evidence collected from one raw decoder pass for ordered reads."""

    safe: bool
    reason: str
    first_pts: int | None
    last_pts: int | None
    cadence: int | None
    frame_count: int
    time_base: Fraction | None
    start_pts: int | None


class FrameNotFoundError(ValueError, LookupError):
    """Raised when an exact requested PTS is not present in the video."""


class _UnsafeSequentialReadError(RuntimeError):
    """The source failed a condition required by ordered selected-PTS reading."""


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
    start_pts = _integer(stream.get("start_pts"))
    start_time = _number(stream.get("start_time"))
    if start_time is None:
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
        "start_pts": start_pts,
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
            "start_pts": start_pts,
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

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        decode_threads: int | None = None,
    ) -> None:
        try:
            candidate = Path(os.fspath(path))
        except TypeError as exc:
            raise TypeError("path must be a string or path-like object") from exc
        if not candidate.exists():
            raise FileNotFoundError(os.fspath(candidate))
        if not candidate.is_file():
            raise ValueError(f"Video path is not a file: {candidate}")

        if decode_threads is None:
            self._decode_threads = None
        else:
            try:
                self._decode_threads = operator.index(decode_threads)
            except TypeError as exc:
                raise TypeError("decode_threads must be a positive integer or None") from exc
            if isinstance(decode_threads, bool) or self._decode_threads <= 0:
                raise ValueError("decode_threads must be a positive integer or None")

        self._path = candidate.resolve()
        self._fingerprint_lock = threading.Lock()
        self._fingerprint_cache: tuple[int, int, str] | None = None
        self._ordered_pts_safety: _SequentialPtsProof | None = None

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
        if self._decode_threads is not None:
            try:
                stream.codec_context.thread_count = self._decode_threads
            except (AttributeError, TypeError, ValueError) as exc:
                container.close()
                raise RuntimeError(
                    f"Unable to set decoder thread count to {self._decode_threads} for {self._path}"
                ) from exc
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
            # ``int(1.5)`` silently changes the source timestamp.  Ordered
            # reading is only safe when the decoder exposes an integer PTS.
            value = operator.index(pts)
        except (TypeError, ValueError, OverflowError):
            return None
        if isinstance(pts, bool):
            return None
        return int(value)

    @staticmethod
    def _frame_time_base(frame: Any) -> Fraction | None:
        """Return a decoded frame's exact time base when it is exposed."""

        return _fraction(getattr(frame, "time_base", None))

    def _stream_timing(self, stream: Any) -> tuple[Fraction, int] | None:
        """Return exact stream time base and integer start PTS when exposed."""

        time_base = _fraction(getattr(stream, "time_base", None))
        if time_base is None:
            return None
        raw_start = getattr(stream, "start_time", None)
        if raw_start is None:
            start_pts = 0
        else:
            if isinstance(raw_start, bool):
                return None
            try:
                start_pts = operator.index(raw_start)
            except (TypeError, ValueError, OverflowError):
                return None
        return time_base, start_pts

    def _timing_matches(self, timing: tuple[Fraction, int] | None) -> bool:
        """Check that a reopened stream retains the reader's exact timing."""

        if timing is None:
            return False
        time_base, start_pts = timing
        expected_time_base = getattr(self, "_time_base", None)
        if expected_time_base is None or time_base != expected_time_base:
            return False
        expected_start_pts = getattr(self._info, "start_pts", None)
        if expected_start_pts is not None:
            return start_pts == expected_start_pts
        # Older metadata providers only retained the float start time.  Use
        # the same PTS*time_base conversion as metadata discovery, with a
        # small representation tolerance for values such as 1/3 second.
        actual_start = float(time_base * start_pts)
        expected_start = float(getattr(self._info, "start_time", 0.0))
        return math.isclose(actual_start, expected_start, rel_tol=0.0, abs_tol=1e-9)

    @staticmethod
    def _proof_contains(proof: _SequentialPtsProof, wanted: set[int]) -> bool:
        """Check selected PTS membership from a compact CFR proof."""

        if (
            proof.first_pts is None
            or proof.last_pts is None
            or proof.cadence is None
            or proof.cadence <= 0
            or proof.frame_count < 2
        ):
            return False
        if proof.last_pts < proof.first_pts:
            return False
        return all(
            proof.first_pts <= value <= proof.last_pts
            and (value - proof.first_pts) % proof.cadence == 0
            for value in wanted
        )

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
            stream_index=getattr(self._info, "stream_index", 0),
        )

    def _seek_pts_for_time(self, relative_time: float) -> int:
        absolute_time = self._info.start_time + relative_time
        return int(round(absolute_time / float(self._time_base)))

    @staticmethod
    def _coerce_pts(pts: int) -> int:
        if isinstance(pts, bool):
            raise TypeError("pts must be an integer PTS")
        try:
            value = operator.index(pts)
        except TypeError as exc:
            raise TypeError("pts must be an integer PTS") from exc
        return int(value)

    @staticmethod
    def _add_metric(metrics: MutableMapping[str, Any] | None, name: str, value: int | float) -> None:
        if metrics is not None:
            metrics[name] = metrics.get(name, 0) + value

    @staticmethod
    def _counted_frames(
        container: Any,
        stream: Any,
        metrics: MutableMapping[str, Any] | None,
    ) -> Iterator[Any]:
        frames = iter(VideoReader._demux_frames(container, stream))
        while True:
            started = time.perf_counter()
            try:
                decoded = next(frames)
            except StopIteration:
                return
            VideoReader._add_metric(metrics, "decode_seconds", time.perf_counter() - started)
            VideoReader._add_metric(metrics, "decoded_frames", 1)
            yield decoded

    def validate_sequential_pts(
        self,
        pts: Iterator[int] | list[int] | tuple[int, ...],
        *,
        metrics: MutableMapping[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Prove whether ordered selected-PTS reads are safe for this source.

        Validation consumes the raw ``packet.decode()`` stream directly.  It
        deliberately does not use :meth:`iter_frames`, which filters duplicate
        and out-of-order timestamps before a caller can reject the source.
        B-frame *capability* is not evidence of failure; the observed decoded
        PTS sequence is what determines eligibility.  Constant cadence is
        required because the optimized pass is only equivalent to the legacy
        exact lookup for a proven CFR source.
        """

        started = time.perf_counter()
        try:
            try:
                wanted = [self._coerce_pts(value) for value in pts]
            except (TypeError, ValueError):
                return False, "invalid selected PTS"
            if not wanted or wanted != sorted(set(wanted)):
                return False, "selected PTS are not strictly increasing"

            cached = getattr(self, "_ordered_pts_safety", None)
            if cached is not None:
                if not isinstance(cached, _SequentialPtsProof):
                    # Be tolerant of readers constructed by older callers or
                    # test doubles that retained the former tuple shape.
                    try:
                        legacy_values = sorted(set(cached[2])) if cached[2] is not None else []
                        legacy_cadence = (
                            legacy_values[1] - legacy_values[0]
                            if len(legacy_values) >= 2
                            else None
                        )
                        if legacy_cadence is not None and any(
                            right - left != legacy_cadence
                            for left, right in zip(legacy_values, legacy_values[1:])
                        ):
                            legacy_cadence = None
                        cached = _SequentialPtsProof(
                            safe=bool(cached[0]),
                            reason=str(cached[1]),
                            first_pts=legacy_values[0] if legacy_values else None,
                            last_pts=legacy_values[-1] if legacy_values else None,
                            cadence=legacy_cadence,
                            frame_count=len(legacy_values),
                            time_base=getattr(self, "_time_base", None),
                            start_pts=None,
                        )
                    except (IndexError, TypeError, ValueError):
                        cached = None
                if cached is not None:
                    if not cached.safe:
                        return False, cached.reason
                    if self._proof_contains(cached, set(wanted)):
                        return True, cached.reason
                    return False, "one or more selected PTS are missing"

            container, stream = self._open_video()
            self._add_metric(metrics, "reader_opens", 1)
            try:
                timing = self._stream_timing(stream)
                if timing is None:
                    reason = "stream time_base/start_time is unavailable"
                    self._ordered_pts_safety = _SequentialPtsProof(
                        False, reason, None, None, None, 0, None, None
                    )
                    return False, reason
                if not self._timing_matches(timing):
                    reason = "stream time_base/start_time changed"
                    self._ordered_pts_safety = _SequentialPtsProof(
                        False, reason, None, None, None, 0, timing[0], timing[1]
                    )
                    return False, reason

                wanted_set = set(wanted)
                wanted_remaining = set(wanted_set)
                previous: int | None = None
                first_pts: int | None = None
                last_pts: int | None = None
                cadence: int | None = None
                frame_count = 0
                for decoded in self._counted_frames(container, stream, metrics):
                    frame_count += 1
                    value = self._frame_pts(decoded)
                    if value is None:
                        reason = "source frame has missing PTS or a non-integer PTS"
                        self._ordered_pts_safety = _SequentialPtsProof(
                            False, reason, first_pts, last_pts, cadence, frame_count, timing[0], timing[1]
                        )
                        return False, reason
                    frame_time_base = self._frame_time_base(decoded)
                    if frame_time_base is None:
                        reason = "decoded frame time_base is unavailable"
                        self._ordered_pts_safety = _SequentialPtsProof(
                            False, reason, first_pts, last_pts, cadence, frame_count, timing[0], timing[1]
                        )
                        return False, reason
                    if frame_time_base != timing[0]:
                        reason = "decoded frame time_base changed"
                        self._ordered_pts_safety = _SequentialPtsProof(
                            False, reason, first_pts, last_pts, cadence, frame_count, timing[0], timing[1]
                        )
                        return False, reason
                    if first_pts is None:
                        first_pts = value
                    else:
                        delta = value - previous  # type: ignore[operator]
                        if delta <= 0:
                            if delta == 0:
                                reason = "source has duplicate PTS"
                            else:
                                reason = "source PTS are nonmonotonic"
                            self._ordered_pts_safety = _SequentialPtsProof(
                                False, reason, first_pts, last_pts, cadence, frame_count, timing[0], timing[1]
                            )
                            return False, reason
                        if cadence is None:
                            cadence = delta
                        elif delta != cadence:
                            reason = "source has variable frame cadence"
                            self._ordered_pts_safety = _SequentialPtsProof(
                                False, reason, first_pts, last_pts, cadence, frame_count, timing[0], timing[1]
                            )
                            return False, reason
                    previous = value
                    last_pts = value
                    wanted_remaining.discard(value)

                if frame_count < 2 or cadence is None:
                    reason = "source cadence cannot be established"
                    self._ordered_pts_safety = _SequentialPtsProof(
                        False, reason, first_pts, last_pts, cadence, frame_count, timing[0], timing[1]
                    )
                    return False, reason
                proof = _SequentialPtsProof(
                    True,
                    "safe constant-cadence stream",
                    first_pts,
                    last_pts,
                    cadence,
                    frame_count,
                    timing[0],
                    timing[1],
                )
                self._ordered_pts_safety = proof
                if wanted_remaining:
                    return False, "one or more selected PTS are missing"
                return True, proof.reason
            except Exception as exc:
                reason = f"sequential PTS validation failed ({type(exc).__name__})"
                self._ordered_pts_safety = _SequentialPtsProof(
                    False, reason, None, None, None, 0, None, None
                )
                return False, reason
            finally:
                self._close_video(container)
        finally:
            self._add_metric(metrics, "validation_seconds", time.perf_counter() - started)

    def iter_selected_frames(
        self,
        pts: Iterator[int] | list[int] | tuple[int, ...],
        *,
        max_side: int | None = 960,
        metrics: MutableMapping[str, Any] | None = None,
    ) -> Iterator[Frame]:
        """Yield exact selected frames in a single ordered decode from start.

        Call :meth:`validate_sequential_pts` first. Only requested frames are
        converted to RGB or resized; other decoded frames remain PyAV frames.
        """

        wanted = [self._coerce_pts(value) for value in pts]
        if not wanted or wanted != sorted(set(wanted)):
            raise ValueError("selected PTS must be strictly increasing")
        max_side_value = _validate_max_side(max_side)

        def iterator() -> Iterator[Frame]:
            container, stream = self._open_video()
            self._add_metric(metrics, "reader_opens", 1)
            try:
                proof = getattr(self, "_ordered_pts_safety", None)
                if not isinstance(proof, _SequentialPtsProof) or not proof.safe:
                    raise _UnsafeSequentialReadError(
                        "ordered PTS read was not proven safe before iteration"
                    )
                timing = self._stream_timing(stream)
                if timing is None:
                    raise _UnsafeSequentialReadError(
                        "stream time_base/start_time is unavailable"
                    )
                if not self._timing_matches(timing):
                    raise _UnsafeSequentialReadError(
                        "stream time_base/start_time changed after validation"
                    )
                if proof.time_base is not None and timing[0] != proof.time_base:
                    raise _UnsafeSequentialReadError(
                        "stream time_base changed after validation"
                    )
                if proof.start_pts is not None and timing[1] != proof.start_pts:
                    raise _UnsafeSequentialReadError(
                        "stream start_time changed after validation"
                    )
                target_index = 0
                previous: int | None = None
                for decoded in self._counted_frames(container, stream, metrics):
                    value = self._frame_pts(decoded)
                    if value is None:
                        raise _UnsafeSequentialReadError(
                            "source frame has missing PTS or a non-integer PTS"
                        )
                    frame_time_base = self._frame_time_base(decoded)
                    if frame_time_base is None:
                        raise _UnsafeSequentialReadError(
                            "decoded frame time_base is unavailable"
                        )
                    if frame_time_base != timing[0]:
                        raise _UnsafeSequentialReadError(
                            "decoded frame time_base changed after validation"
                        )
                    if previous is None:
                        if proof.first_pts is not None and value != proof.first_pts:
                            raise _UnsafeSequentialReadError(
                                "first PTS changed after validation"
                            )
                    else:
                        delta = value - previous
                        if delta <= 0:
                            raise _UnsafeSequentialReadError(
                                "source PTS order changed after validation"
                            )
                        if proof.cadence is not None and delta != proof.cadence:
                            raise _UnsafeSequentialReadError(
                                "source PTS cadence changed after validation"
                            )
                    previous = value
                    if target_index < len(wanted) and value == wanted[target_index]:
                        started = time.perf_counter()
                        frame = self._frame(decoded, max_side_value)
                        self._add_metric(metrics, "rgb_seconds", time.perf_counter() - started)
                        if frame is None:
                            raise _UnsafeSequentialReadError("selected source frame has no PTS")
                        target_index += 1
                        yield frame
                        if target_index == len(wanted):
                            return
                    elif target_index < len(wanted) and value > wanted[target_index]:
                        raise _UnsafeSequentialReadError("selected PTS was skipped during ordered decode")
                if target_index != len(wanted):
                    raise _UnsafeSequentialReadError("one or more selected PTS disappeared during ordered decode")
            finally:
                self._close_video(container)

        return iterator()

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

    def iter_frames(
        self,
        start: float = 0.0,
        end: float | None = None,
        max_side: int | None = 960,
    ) -> Iterator[Frame]:
        """Yield every decoded presentation frame in a time range.

        Unlike :meth:`sample`, this method does not synthesize a sampling
        cadence: every frame exposed by the decoder is yielded in PTS order.
        It is intended for low-cost sequential analysis and keeps only one
        decoded frame alive at a time.  ``start`` and ``end`` are relative to
        the stream start; ``end`` is inclusive.
        """

        start_value = self._coerce_time(start, "start")
        if start_value < 0:
            raise ValueError("start must be >= 0")
        end_value = self._info.duration if end is None else self._coerce_time(end, "end")
        if end_value < start_value:
            raise ValueError("end must be >= start")
        max_side_value = _validate_max_side(max_side)

        def iterator() -> Iterator[Frame]:
            container, stream = self._open_video()
            try:
                self._seek(container, stream, self._seek_pts_for_time(start_value))
                seen_pts: set[int] = set()
                last_time = -math.inf
                epsilon = 1e-9
                for decoded in self._demux_frames(container, stream):
                    pts = self._frame_pts(decoded)
                    if pts is None or pts in seen_pts:
                        continue
                    seen_pts.add(pts)
                    relative_time = float(pts * self._time_base) - self._info.start_time
                    if relative_time < start_value - epsilon:
                        continue
                    if relative_time > end_value + epsilon:
                        break
                    if relative_time <= last_time + epsilon:
                        continue
                    frame = self._frame(decoded, max_side_value)
                    if frame is None:
                        continue
                    yield frame
                    last_time = frame.time
            finally:
                self._close_video(container)

        return iterator()

    def get_frame(
        self,
        pts: int,
        max_side: int | None = None,
        *,
        metrics: MutableMapping[str, Any] | None = None,
    ) -> Frame:
        """Decode and return the frame with exactly ``pts``.

        Seeking is attempted first.  If a demuxer cannot seek accurately, a
        fresh sequential pass is used so that an existing PTS is never
        incorrectly reported as missing.
        """

        target_pts = self._coerce_pts(pts)
        max_side_value = _validate_max_side(max_side)
        found: Frame | None = None

        container, stream = self._open_video()
        self._add_metric(metrics, "reader_opens", 1)
        try:
            seek_succeeded = self._seek(container, stream, target_pts)
            previous_pts: int | None = None
            non_monotonic = False
            for decoded in self._counted_frames(container, stream, metrics):
                decoded_pts = self._frame_pts(decoded)
                if decoded_pts is None:
                    continue
                if previous_pts is not None and decoded_pts < previous_pts:
                    non_monotonic = True
                previous_pts = decoded_pts
                if decoded_pts == target_pts:
                    started = time.perf_counter()
                    found = self._frame(decoded, max_side_value)
                    self._add_metric(metrics, "rgb_seconds", time.perf_counter() - started)
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
        self._add_metric(metrics, "seek_fallbacks", 1)
        container, stream = self._open_video()
        self._add_metric(metrics, "reader_opens", 1)
        try:
            for decoded in self._counted_frames(container, stream, metrics):
                if self._frame_pts(decoded) == target_pts:
                    started = time.perf_counter()
                    found = self._frame(decoded, max_side_value)
                    self._add_metric(metrics, "rgb_seconds", time.perf_counter() - started)
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

    def export_pngs(
        self,
        destinations: dict[int, str | os.PathLike[str] | list[str | os.PathLike[str]]],
    ) -> set[int]:
        """Export selected exact PTS frames in one ordered decode pass.

        Each output is first written beside its final path and atomically
        renamed into place.  Missing PTS values are returned so callers can
        mark only those candidates as failed without accepting partial files.
        """

        normalized: dict[int, list[Path]] = {}
        for raw_pts, raw_destinations in destinations.items():
            pts = self._coerce_pts(raw_pts)
            values = raw_destinations if isinstance(raw_destinations, list) else [raw_destinations]
            normalized[pts] = [Path(os.fspath(value)) for value in values]
        if not normalized:
            return set()
        wanted = sorted(normalized)
        wanted_set = set(wanted)
        found: set[int] = set()
        start_time = max(0.0, float(wanted[0] * self._time_base) - self._info.start_time)
        container, stream = self._open_video()
        try:
            self._seek(container, stream, self._seek_pts_for_time(start_time))
            for decoded in self._demux_frames(container, stream):
                pts = self._frame_pts(decoded)
                if pts is None:
                    continue
                if pts in wanted_set:
                    # Match get_frame's first-match semantics if a damaged or
                    # unusual source repeats a presentation timestamp.
                    if pts in found:
                        continue
                    frame = self._frame(decoded, max_side=None)
                    if frame is None:
                        continue
                    for destination in normalized[pts]:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        temp = destination.with_name(destination.name + ".tmp")
                        try:
                            frame.image.save(os.fspath(temp), format="PNG")
                            os.replace(temp, destination)
                        finally:
                            temp.unlink(missing_ok=True)
                    found.add(pts)
                    if len(found) == len(wanted_set):
                        break
        finally:
            self._close_video(container)
        return wanted_set - found


__all__ = ["Frame", "FrameNotFoundError", "VideoInfo", "VideoReader"]
