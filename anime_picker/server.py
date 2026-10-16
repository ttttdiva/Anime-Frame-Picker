"""Local HTTP API for Anime Frame Picker.

The server is deliberately small and boring at its boundary.  It keeps all
state in process, accepts only loopback requests, and exposes only files that
have been registered as references or job results.  Heavy model and video
work is deferred to the single worker used for jobs; importing this module is
therefore safe for the CLI, tests, and the offline browser UI.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import mimetypes
import os
import re
import shutil
import threading
import uuid
import webbrowser
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:  # Pillow is present in the supported local runtime.
    from PIL import Image
except Exception:  # pragma: no cover - import errors are reported by endpoints
    Image = None  # type: ignore[assignment]


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
# Packaged installs keep the UI beside the Python package.  During local
# development the repository-level static directory is also accepted while
# that package directory is being assembled.
STATIC_ROOT = PACKAGE_ROOT / "anime_picker" / "static"
if not STATIC_ROOT.is_dir():
    STATIC_ROOT = PACKAGE_ROOT / "static"
CACHE_ROOT = Path(os.environ.get("ANIME_PICKER_CACHE", str(PACKAGE_ROOT / ".cache")))
REFERENCE_ROOT = CACHE_ROOT / "references"
RESULT_ROOT = CACHE_ROOT / "results"
UPLOAD_ROOT = CACHE_ROOT / "uploads"

# Keep the public defaults stable.  The export root is resolved from the
# repository location so direct launches do not depend on their current
# working directory.  An explicit environment or per-job path remains valid.
DEFAULT_MODEL_ROOT = Path(os.environ.get("ANIME_PICKER_MODEL_ROOT", r"D:\AI\models\Cold_F\image"))


def _resolve_output_root(value: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an export root against the repository when it is relative."""

    raw = os.fspath(value) if value is not None else ""
    if not isinstance(raw, str) or "\x00" in raw:
        raise ValueError("invalid output root")
    if not raw.strip():
        return (PACKAGE_ROOT / "outputs").resolve(strict=False)
    if raw.startswith(("\\\\", "//")):
        raise ValueError("network paths are not allowed")
    path = Path(raw).expanduser()
    if not path.is_absolute() and (path.drive or path.root):
        raise ValueError("output root must be repository-relative or absolute")
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return path.resolve(strict=False)


DEFAULT_OUTPUT_ROOT = _resolve_output_root(os.environ.get("ANIME_PICKER_OUTPUT_ROOT"))
# Logo removal shares the prepared AnimeLaMa bundle with the rest of the
# local models.  The model path is a fixed local setup value; HTTP requests
# may select profiles but may not select arbitrary model files.
DEFAULT_LOGO_MODEL_ROOT = Path(
    os.environ.get(
        "ANIME_PICKER_LOGO_MODEL_ROOT",
        str(DEFAULT_MODEL_ROOT / "anime-lama" / "anime-manga-big-lama.pt"),
    )
)
DEFAULT_LOGO_PROFILE_ROOT = Path(
    os.environ.get("ANIME_PICKER_LOGO_PROFILE_ROOT", str(PACKAGE_ROOT / ".local" / "logo-profiles"))
)

VIDEO_SUFFIXES = {".mp4", ".mkv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
LOGO_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_UPLOAD_BYTES = int(os.environ.get("ANIME_PICKER_MAX_UPLOAD_BYTES", str(8 * 1024**3)))
MAX_REFERENCE_BYTES = int(os.environ.get("ANIME_PICKER_MAX_REFERENCE_BYTES", str(100 * 1024**2)))
MAX_STATIC_BYTES = int(os.environ.get("ANIME_PICKER_MAX_STATIC_BYTES", str(20 * 1024**2)))

DEFAULT_SETTINGS: dict[str, Any] = {
    "start": 0.0,
    "end": 0.0,
    "coarse_step": 2.0,
    "dense_step": 0.25,
    "candidate_count": 30,
    "identity_threshold": 0.213231,
    "tag_threshold": 0.17,
    "quality_min": 0.15,
    "rescue": True,
    "identity_weight": 0.35,
    "condition_weight": 0.30,
    "quality_weight": 0.35,
    # Logo cleanup is an export-time operation.  Analysis and candidate
    # previews always remain the original pixels so scores and PTS are stable.
    "logo_removal": True,
    "logo_profiles": [],
}

# Conditions are accepted only when they appear in the prepared model's
# vocabulary manifest.  Keeping this empty avoids presenting guessed PixAI
# labels (for example ``person`` or ``smile``) as supported tags before the
# local model has been prepared.  The alias table remains useful for resolving
# Japanese labels after a manifest is available.
BUILTIN_VOCABULARY: dict[str, str] = {}

BUILTIN_ALIASES: dict[str, str] = {
    "人物": "person",
    "キャラクター": "character",
    "顔": "face",
    "一人": "solo",
    "笑顔": "smile",
    "笑い": "smile",
    "嬉しい": "happy",
    "悲しい": "sad",
    "怒り": "angry",
    "驚き": "surprised",
    "無表情": "neutral",
    "横顔": "profile",
    "顔アップ": "close-up",
    "アップ": "close-up",
    "全身": "full_body",
    "座っている": "sitting",
    "上半身": "upper_body",
    "正面": "facing_viewer",
    "こちらを見る": "looking_at_viewer",
    "ポートレート": "portrait",
    "屋外": "outdoors",
    "室内": "indoors",
    "夜": "night",
    "昼": "day",
    "アクション": "action",
}


def _import_video_reader() -> Any:
    try:
        from .video import VideoReader

        return VideoReader
    except Exception:
        return None


def _import_model_service() -> Any:
    try:
        from .models import ModelService

        return ModelService
    except Exception:
        return None


def _import_pipeline() -> Any:
    try:
        from .pipeline import run_search

        return run_search
    except Exception:
        return None


def _import_logo_components() -> tuple[Any, Any]:
    """Resolve logo helpers only when a logo operation is requested.

    ``server`` is also used by logo-off jobs and by the CLI's inspection
    commands.  Keeping this import lazy is what prevents an explicit logo-off
    path from importing AnimeLaMa, torch, or any other logo runtime.
    """

    global LogoProcessor, ProfileStore
    if LogoProcessor is not None and ProfileStore is not None:
        return LogoProcessor, ProfileStore
    try:
        from .logo_removal import LogoProcessor as processor_class
        from .logo_removal import ProfileStore as profile_store_class
    except Exception:
        return LogoProcessor, ProfileStore
    if LogoProcessor is None:
        LogoProcessor = processor_class
    if ProfileStore is None:
        ProfileStore = profile_store_class
    return LogoProcessor, ProfileStore


# These names are intentionally module globals.  Tests and embedding callers
# can provide small fakes without importing heavyweight model dependencies.
VideoReader: Any = _import_video_reader()
ModelService: Any = None
run_search: Any = None
# These are deliberately kept optional and imported lazily.  An explicit
# logo-off export must not import AnimeLaMa/runtime libraries or initialize a
# logo model.  Tests and embedding callers can install small doubles here.
LogoProcessor: Any = None
ProfileStore: Any = None


@dataclass
class VideoRecord:
    id: str
    path: Path
    info: dict[str, Any]
    fingerprint: str
    created_at: str


@dataclass
class ReferenceRecord:
    id: str
    path: Path
    name: str
    width: int
    height: int
    created_at: str


@dataclass
class JobRecord:
    id: str
    video_id: str
    reference_items: list[dict[str, Any]]
    wanted: list[dict[str, str]]
    avoided: list[dict[str, str]]
    settings: dict[str, Any]
    status: str = "queued"
    progress: float = 0.0
    message: str = "待機中"
    error: str | None = None
    warnings: list[Any] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    future: Future[Any] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    result_root: Path | None = None
    model_revision: Any = None
    created_at: str = field(default_factory=lambda: _utc_now())


VIDEO_REGISTRY: dict[str, VideoRecord] = {}
REFERENCE_REGISTRY: dict[str, ReferenceRecord] = {}
JOB_REGISTRY: dict[str, JobRecord] = {}
_STATE_LOCK = threading.RLock()
_GPU_LOCK = threading.Lock()
_EXECUTOR: ThreadPoolExecutor | None = None
_MODEL_INSTANCE: Any = None
_MODEL_LOCK = threading.Lock()
_MODEL_MANIFEST_INSTANCE: Any = None
_MODEL_MANIFEST_ROOT: Path | None = None
_MODEL_MANIFEST_VOCABULARY: dict[str, str] | None = None
_LOGO_PROCESSOR_INSTANCE: Any = None
_LOGO_PROCESSOR_ROOT: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_dirs() -> None:
    for directory in (CACHE_ROOT, REFERENCE_ROOT, RESULT_ROOT, UPLOAD_ROOT):
        directory.mkdir(parents=True, exist_ok=True)


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        # The single worker is an explicit GPU resource boundary.  It also
        # makes cancellation and result bookkeeping deterministic.
        _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anime-picker-job")
    return _EXECUTOR


def _safe_resolve(path: str | os.PathLike[str]) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("invalid path")
    # Network shares and UNC paths would let a browser request reach outside
    # the local machine.  The app is intentionally local-drive only.
    if raw.startswith(("\\\\", "//")):
        raise ValueError("network paths are not allowed")
    return Path(raw).expanduser().resolve(strict=False)


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _safe_filename(value: str, fallback: str = "export") -> str:
    """Return a portable Windows-safe filename component."""

    value = str(value or "").strip()
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    if not value:
        value = fallback
    if value.upper() in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"COM[1-9]|LPT[1-9]", value.upper()):
        value = f"_{value}"
    return value[:120]


def _fingerprint(path: Path) -> str:
    """Hash a source in bounded chunks and include size/mtime for diagnostics."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return f"sha256:{digest.hexdigest()}:size:{stat.st_size}:mtime_ns:{stat.st_mtime_ns}"


def _check_video_path(path_value: str | os.PathLike[str]) -> Path:
    try:
        path = _safe_resolve(path_value)
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail="invalid video path") from exc
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=422, detail="only MP4 and MKV videos are supported")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="video file not found")
    return path


def _video_roots() -> list[Path]:
    roots = [PACKAGE_ROOT / "source", Path.cwd() / "source"]
    configured = os.environ.get("ANIME_PICKER_VIDEO_ROOT")
    if configured:
        roots.extend(Path(item) for item in configured.split(os.pathsep) if item and not item.startswith(("\\\\", "//")))
    return [root.resolve(strict=False) for root in roots]


def _discover_videos() -> list[Path]:
    seen: set[str] = set()
    discovered: list[Path] = []
    for root in _video_roots():
        if not root.is_dir():
            continue
        try:
            files = sorted(root.iterdir(), key=lambda item: item.name.casefold())
        except OSError:
            continue
        for path in files:
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                key = os.path.normcase(os.fspath(path.resolve()))
                if key not in seen:
                    seen.add(key)
                    discovered.append(path.resolve())
    return discovered


def _coerce_number(value: Any, *, name: str, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise HTTPException(status_code=422, detail=f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise HTTPException(status_code=422, detail=f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise HTTPException(status_code=422, detail=f"{name} must be <= {maximum}")
    return result


def _coerce_int(value: Any, *, name: str, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise HTTPException(status_code=422, detail=f"{name} must be an integer")
    if minimum is not None and number < minimum:
        raise HTTPException(status_code=422, detail=f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise HTTPException(status_code=422, detail=f"{name} must be <= {maximum}")
    return number


def _coerce_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise HTTPException(status_code=422, detail=f"{name} must be a boolean")
    return value


def _extract_video_info(path: Path) -> dict[str, Any]:
    """Use the project's PTS-aware reader, with OpenCV metadata fallback."""

    reader_class = VideoReader or _import_video_reader()
    info: Any = None
    if reader_class is not None:
        try:
            reader = reader_class(path)
            info_value = getattr(reader, "info", None)
            info = info_value() if callable(info_value) else info_value
            if info is None:
                info_value = getattr(reader, "metadata", None)
                info = info_value() if callable(info_value) else info_value
        except Exception:
            info = None

    def value(name: str, default: Any = None) -> Any:
        if isinstance(info, Mapping):
            return info.get(name, default)
        return getattr(info, name, default) if info is not None else default

    width = value("width")
    height = value("height")
    duration = value("duration", 0.0)
    fps = value("fps", 0.0)
    stream_index = value("stream_index", 0)
    time_base = value("time_base", "1/1")
    start_time = value("start_time", 0.0)

    if width is None or height is None:
        try:
            import cv2

            capture = cv2.VideoCapture(os.fspath(path))
            try:
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
            finally:
                capture.release()
        except Exception:
            width = height = 0

    try:
        width_i, height_i = int(width), int(height)
    except (TypeError, ValueError):
        width_i = height_i = 0
    if width_i <= 0 or height_i <= 0:
        raise HTTPException(status_code=422, detail="video has no readable video stream")
    try:
        duration_f = max(0.0, float(duration or 0.0))
    except (TypeError, ValueError):
        duration_f = 0.0
    try:
        fps_f = max(0.0, float(fps or 0.0))
    except (TypeError, ValueError):
        fps_f = 0.0
    return {
        "width": width_i,
        "height": height_i,
        "duration": duration_f,
        "fps": fps_f,
        "path": os.fspath(path),
        "stream_index": int(stream_index or 0),
        "time_base": str(time_base or "1/1"),
        "start_time": float(start_time or 0.0),
    }


def _reader_fingerprint(path: Path) -> str:
    reader_class = VideoReader or _import_video_reader()
    if reader_class is not None:
        try:
            reader = reader_class(path)
            method = getattr(reader, "fingerprint", None)
            if callable(method):
                value = method()
                if value:
                    return str(value)
            elif method:
                return str(method)
        except Exception:
            pass
    return _fingerprint(path)


def _register_video(path: Path, info: dict[str, Any] | None = None) -> VideoRecord:
    resolved = path.resolve(strict=True)
    info = info or _extract_video_info(resolved)
    fingerprint = _reader_fingerprint(resolved)
    with _STATE_LOCK:
        for existing in VIDEO_REGISTRY.values():
            try:
                if existing.path.resolve() == resolved:
                    existing.info.update(info)
                    existing.fingerprint = fingerprint
                    return existing
            except OSError:
                continue
        record = VideoRecord(
            id=uuid.uuid4().hex,
            path=resolved,
            info=info,
            fingerprint=fingerprint,
            created_at=_utc_now(),
        )
        VIDEO_REGISTRY[record.id] = record
        return record


def _info_response(record: VideoRecord) -> dict[str, Any]:
    return {
        "video_id": record.id,
        "info": {
            "width": record.info.get("width"),
            "height": record.info.get("height"),
            "duration": record.info.get("duration", 0.0),
            "fps": record.info.get("fps", 0.0),
            "path": os.fspath(record.path),
        },
    }


def _get_video(video_id: str) -> VideoRecord:
    if not re.fullmatch(r"[0-9a-fA-F-]{16,64}", video_id or ""):
        raise HTTPException(status_code=404, detail="video not found")
    with _STATE_LOCK:
        record = VIDEO_REGISTRY.get(video_id)
    if record is None:
        raise HTTPException(status_code=404, detail="video not found")
    try:
        if not record.path.is_file():
            raise HTTPException(status_code=404, detail="video file not found")
    except OSError as exc:
        raise HTTPException(status_code=404, detail="video file unavailable") from exc
    return record


def _manifest_vocabulary() -> dict[str, str] | None:
    """Read the prepared tag manifest without initializing model runtimes."""

    global _MODEL_MANIFEST_INSTANCE, _MODEL_MANIFEST_ROOT, _MODEL_MANIFEST_VOCABULARY
    root = DEFAULT_MODEL_ROOT.resolve(strict=False)
    if _MODEL_MANIFEST_ROOT == root and _MODEL_MANIFEST_VOCABULARY is not None:
        return dict(_MODEL_MANIFEST_VOCABULARY)
    service_class = ModelService or _import_model_service()
    if service_class is None:
        return None
    try:
        # ModelService.__init__ only reads manifest/config files and configures
        # offline cache paths; it does not load CUDA weights.
        service = service_class(model_root=DEFAULT_MODEL_ROOT, require_gpu=False)
        value = service.vocabulary()
    except TypeError:
        try:
            value = service_class.vocabulary()
            service = None
        except Exception:
            return None
    except Exception:
        return None
    if not isinstance(value, Mapping):
        return None
    result = {
        str(key).strip(): str(category or "general")
        for key, category in value.items()
        if isinstance(key, str) and key.strip()
    }
    if not result:
        return None
    _MODEL_MANIFEST_INSTANCE = service
    _MODEL_MANIFEST_ROOT = root
    _MODEL_MANIFEST_VOCABULARY = result
    return dict(result)


def _aliases_and_vocabulary() -> tuple[dict[str, str], dict[str, str]]:
    aliases = dict(BUILTIN_ALIASES)
    vocab = dict(BUILTIN_VOCABULARY)
    # A loaded instance is preferred.  Before the first job, a local manifest
    # reader supplies real PixAI tags without touching GPU/runtime libraries.
    manifest_vocab = _manifest_vocabulary()
    if manifest_vocab:
        vocab.update(manifest_vocab)
    if _MODEL_INSTANCE is not None:
        try:
            value = _MODEL_INSTANCE.vocabulary()
            if isinstance(value, Mapping):
                vocab.update({str(k).strip(): str(v or "general") for k, v in value.items() if isinstance(k, str) and k.strip()})
        except Exception:
            pass
        for attr in ("aliases", "alias_map"):
            value = getattr(_MODEL_INSTANCE, attr, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            if isinstance(value, Mapping):
                aliases.update({str(k): str(v) for k, v in value.items() if isinstance(k, str) and isinstance(v, str)})
    return aliases, vocab


def _prepared_vocabulary() -> dict[str, str] | None:
    """Return the model manifest, or ``None`` until one is prepared.

    The public vocabulary endpoint can still describe an empty/unready state,
    but job submission must never treat a hand-written suggestion as a model
    condition.  Static ``vocabulary`` methods are supported for manifest-only
    test and doctor implementations; a normal instance method is used after
    the worker has loaded ModelService.
    """

    if _MODEL_INSTANCE is not None:
        try:
            value = _MODEL_INSTANCE.vocabulary()
        except Exception:
            value = None
        if isinstance(value, Mapping):
            result = {
                str(key).strip(): str(category or "general")
                for key, category in value.items()
                if isinstance(key, str) and key.strip()
            }
            if result:
                return result
    return _manifest_vocabulary()


def _scope_for_category(category: str) -> str:
    lower = str(category or "frame").strip().casefold()
    return "subject" if lower in {"subject", "person", "character", "identity", "face", "target"} else "frame"


def _normalize_tags(items: Any, *, field: str) -> list[dict[str, str]]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise HTTPException(status_code=422, detail=f"{field} must be an array")
    aliases, vocab = _aliases_and_vocabulary()
    prepared_vocab = _prepared_vocabulary()
    if prepared_vocab is not None:
        # Job validation is always against the prepared manifest.  The merged
        # public vocabulary may contain future UI suggestions, but those must
        # not become executable conditions until the model provides them.
        vocab = prepared_vocab
    elif items:
        raise HTTPException(status_code=422, detail="model vocabulary is not ready")
    normalized: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            raw_tag, requested_scope = item, None
        elif isinstance(item, Mapping):
            raw_tag = item.get("tag", item.get("name"))
            requested_scope = item.get("scope")
        else:
            raise HTTPException(status_code=422, detail=f"{field} contains an invalid tag")
        if not isinstance(raw_tag, str) or not raw_tag.strip():
            raise HTTPException(status_code=422, detail=f"{field} contains an invalid tag")
        raw_tag = raw_tag.strip()
        canonical = aliases.get(raw_tag, raw_tag)
        if canonical not in vocab:
            # Case-insensitive matching is helpful for labels from the model
            # manifest while preserving its canonical spelling in the result.
            match = next((key for key in vocab if key.casefold() == canonical.casefold()), None)
            if match is not None:
                canonical = match
        if canonical not in vocab:
            raise HTTPException(status_code=422, detail=f"unknown vocabulary tag: {raw_tag}")
        category = str(vocab[canonical]).strip().casefold()
        if category != "general":
            raise HTTPException(
                status_code=422,
                detail=f"tag {raw_tag} belongs to unsupported vocabulary category {vocab[canonical]}; choose a general tag",
            )
        # PixAI's general labels are scene/expression conditions.  They may be
        # evaluated on the subject crop or full frame according to the UI
        # scope, while character/copyright/style/meta/rating labels are never
        # silently treated as expressions.
        expected_scope = "subject" if requested_scope == "subject" else "frame"
        if requested_scope is not None:
            requested_scope = str(requested_scope)
            if requested_scope not in {"subject", "frame"}:
                raise HTTPException(status_code=422, detail=f"{field} tag scope must be subject or frame")
            # Character/person tags are supplemental identity conditions.  Do
            # not let them masquerade as expression/frame conditions.
            if requested_scope != expected_scope:
                raise HTTPException(
                    status_code=422,
                    detail=f"tag {raw_tag} belongs to the {expected_scope} scope",
                )
        normalized.append({"tag": canonical, "scope": requested_scope or expected_scope})
    return normalized


def _model_status() -> dict[str, Any]:
    service_class = ModelService or _import_model_service()
    details: Any = None
    ready = False
    if service_class is not None:
        status_method = getattr(service_class, "status", None)
        if callable(status_method):
            try:
                value = status_method(DEFAULT_MODEL_ROOT)
                if isinstance(value, Mapping):
                    details = dict(value)
                    ready = bool(details.get("ready", False))
                else:
                    ready = bool(value)
                    details = {"model_root": os.fspath(DEFAULT_MODEL_ROOT)}
            except Exception as exc:
                details = {"model_root": os.fspath(DEFAULT_MODEL_ROOT), "error": str(exc)}
        else:
            details = {"model_root": os.fspath(DEFAULT_MODEL_ROOT), "ready": False}
    else:
        details = {"model_root": os.fspath(DEFAULT_MODEL_ROOT), "ready": False, "reason": "models unavailable"}
    if details is None:
        details = {"model_root": os.fspath(DEFAULT_MODEL_ROOT)}
    details.setdefault("model_root", os.fspath(DEFAULT_MODEL_ROOT))
    details.setdefault("ready", ready)
    return {"ready": bool(ready), "details": details}


class _LogoUnavailable(RuntimeError):
    """Raised when a local logo model/profile bundle cannot be used."""


def _logo_profile_root() -> Path:
    value = os.environ.get("ANIME_PICKER_LOGO_PROFILE_ROOT")
    if value:
        try:
            return _safe_resolve(value)
        except (TypeError, ValueError, OSError):
            # The endpoint reports an empty/unready profile store.  It never
            # turns an invalid environment value into a file-serving root.
            return DEFAULT_LOGO_PROFILE_ROOT.resolve(strict=False)
    return DEFAULT_LOGO_PROFILE_ROOT.resolve(strict=False)


def _logo_components() -> tuple[Any, Any]:
    processor_class, profile_store_class = _import_logo_components()
    if processor_class is None or profile_store_class is None:
        raise _LogoUnavailable("logo removal components are unavailable")
    return processor_class, profile_store_class


def _logo_status() -> dict[str, Any]:
    """Return model readiness without constructing/loading a processor."""

    try:
        processor_class, _profile_store_class = _logo_components()
    except _LogoUnavailable as exc:
        return {
            "ready": False,
            "details": {
                "model_root": os.fspath(DEFAULT_LOGO_MODEL_ROOT),
                "reason": str(exc),
            },
        }
    status_method = getattr(processor_class, "status", None)
    details: Any = None
    ready = False
    if callable(status_method):
        try:
            value = status_method(DEFAULT_LOGO_MODEL_ROOT)
            if isinstance(value, Mapping):
                details = dict(value)
                ready = bool(details.get("ready", False))
            else:
                ready = bool(value)
                details = {"model_root": os.fspath(DEFAULT_LOGO_MODEL_ROOT)}
        except Exception as exc:
            details = {"model_root": os.fspath(DEFAULT_LOGO_MODEL_ROOT), "error": str(exc)}
    else:
        # Small test doubles may not expose status.  They remain usable when
        # explicitly installed, while production setup still reports an
        # unready model until LogoProcessor.status says otherwise.
        details = {
            "model_root": os.fspath(DEFAULT_LOGO_MODEL_ROOT),
            "ready": False,
            "reason": "logo processor has no status method",
        }
    details.setdefault("model_root", os.fspath(DEFAULT_LOGO_MODEL_ROOT))
    details.setdefault("ready", ready)
    return {"ready": bool(ready), "details": details}


def _profile_metadata(value: Any) -> dict[str, Any] | None:
    """Convert a ProfileStore public record to a bounded JSON object."""

    if isinstance(value, Mapping):
        source = value
        getter = source.get
    else:
        source = {}

        def getter(key: str, default: Any = None) -> Any:
            return getattr(value, key, default)

    profile_id = getter("id", getter("profile_id", None))
    if not isinstance(profile_id, str) or not profile_id.strip():
        return None
    result: dict[str, Any] = {
        "id": profile_id.strip(),
        "name": str(getter("name", profile_id)).strip() or profile_id.strip(),
    }
    reference_size = getter("reference_size", getter("size", None))
    if reference_size is not None:
        # Keep scalar sizes as-is, but avoid serializing arbitrary objects.
        if isinstance(reference_size, (str, int, float, bool, list, tuple, dict)):
            result["reference_size"] = reference_size
    for key in ("ready", "details", "description"):
        item = getter(key, None)
        if item is not None and isinstance(item, (str, int, float, bool, list, tuple, dict)):
            result[key] = item
    result.setdefault("ready", True)
    return result


def _logo_profiles_public() -> list[dict[str, Any]]:
    try:
        _processor_class, store_class = _logo_components()
        store = store_class(_logo_profile_root())
        list_method = getattr(store, "list", None)
        values = list_method() if callable(list_method) else []
    except Exception:
        return []
    if values is None:
        return []
    if isinstance(values, Mapping):
        values = list(values.values())
    try:
        iterator = list(values)
    except TypeError:
        return []
    result: list[dict[str, Any]] = []
    for value in iterator:
        metadata = _profile_metadata(value)
        if metadata is not None:
            result.append(metadata)
    return result


def _validate_logo_profile_ids(profile_ids: Any) -> list[str]:
    """Validate IDs through ProfileStore without accepting filesystem paths."""

    if profile_ids is None:
        return []
    if not isinstance(profile_ids, list):
        raise HTTPException(status_code=422, detail="logo_profiles must be an array")
    if any(not isinstance(item, str) or not item.strip() for item in profile_ids):
        raise HTTPException(status_code=422, detail="logo_profiles must contain non-empty IDs")
    normalized = [item.strip() for item in profile_ids]
    if any(LOGO_PROFILE_ID_RE.fullmatch(item) is None for item in normalized):
        raise HTTPException(status_code=422, detail="logo_profiles contains an invalid ID")
    if len(set(normalized)) != len(normalized):
        raise HTTPException(status_code=422, detail="logo_profiles must not contain duplicates")
    if not normalized:
        # Empty means automatic profile selection by LogoProcessor.  Do not
        # import or initialize the optional logo runtime for that no-op
        # validation case.
        return []
    try:
        _processor_class, store_class = _logo_components()
        store = store_class(_logo_profile_root())
        load_method = getattr(store, "load", None)
        if not callable(load_method):
            raise _LogoUnavailable("profile store does not provide load")
        loaded = load_method(normalized)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid logo profile selection: {exc}") from exc
    if loaded is None:
        raise HTTPException(status_code=422, detail="invalid logo profile selection")
    try:
        loaded_items = list(loaded)
    except TypeError as exc:
        raise HTTPException(status_code=422, detail="invalid logo profile selection") from exc
    loaded_ids = {
        metadata["id"]
        for metadata in (_profile_metadata(item) for item in loaded_items)
        if metadata is not None
    }
    missing = [item for item in normalized if item not in loaded_ids]
    if missing:
        raise HTTPException(status_code=422, detail=f"unknown logo profile: {missing[0]}")
    return normalized


def _normalize_logo_profile_ids_shape(profile_ids: Any) -> list[str]:
    """Validate the JSON shape without importing the optional logo runtime."""

    if profile_ids is None:
        return []
    if not isinstance(profile_ids, list):
        raise HTTPException(status_code=422, detail="logo_profiles must be an array")
    if any(not isinstance(item, str) or not item.strip() for item in profile_ids):
        raise HTTPException(status_code=422, detail="logo_profiles must contain non-empty IDs")
    normalized = [item.strip() for item in profile_ids]
    if any(LOGO_PROFILE_ID_RE.fullmatch(item) is None for item in normalized):
        raise HTTPException(status_code=422, detail="logo_profiles contains an invalid ID")
    if len(set(normalized)) != len(normalized):
        raise HTTPException(status_code=422, detail="logo_profiles must not contain duplicates")
    return normalized


def _load_logo_profiles(profile_ids: list[str] | None) -> list[Any]:
    """Load validated internal profiles for a logo operation."""

    _processor_class, store_class = _logo_components()
    store = store_class(_logo_profile_root())
    load_method = getattr(store, "load", None)
    if not callable(load_method):
        raise _LogoUnavailable("profile store does not provide load")
    try:
        loaded = load_method(profile_ids or None)
    except TypeError:
        # A tiny adapter may require an explicit list for auto selection.
        try:
            loaded = load_method(profile_ids or [])
        except TypeError:
            loaded = load_method()
    if loaded is None:
        return []
    try:
        return list(loaded)
    except TypeError as exc:
        raise _LogoUnavailable("profile store returned invalid profiles") from exc


def _get_logo_processor() -> Any:
    global _LOGO_PROCESSOR_INSTANCE, _LOGO_PROCESSOR_ROOT
    processor_class, _store_class = _logo_components()
    # LogoProcessor itself performs conservative profile matching first and
    # loads Anime-LaMa only after a registered logo hit.  Do not gate this
    # construction on status(): a missing model must still allow no-hit
    # frames to pass through unchanged, while an actual hit gets an
    # actionable model/GPU error from process().
    with _MODEL_LOCK:
        if _LOGO_PROCESSOR_INSTANCE is not None and _LOGO_PROCESSOR_ROOT == DEFAULT_LOGO_MODEL_ROOT:
            return _LOGO_PROCESSOR_INSTANCE
        try:
            processor = processor_class(model_root=DEFAULT_LOGO_MODEL_ROOT, require_gpu=True)
        except TypeError:
            # Test adapters and older local adapters may not expose the
            # require_gpu keyword.  Production LogoProcessor does.
            processor = processor_class(model_root=DEFAULT_LOGO_MODEL_ROOT)
        _LOGO_PROCESSOR_INSTANCE = processor
        _LOGO_PROCESSOR_ROOT = DEFAULT_LOGO_MODEL_ROOT
        return processor


def _load_model_service(model_root: Path) -> Any:
    global _MODEL_INSTANCE
    if os.fspath(model_root).startswith(("\\\\", "//")):
        raise RuntimeError("network model roots are not allowed")
    with _MODEL_LOCK:
        if _MODEL_INSTANCE is not None:
            return _MODEL_INSTANCE
        service_class = ModelService or _import_model_service()
        if service_class is None:
            raise RuntimeError("anime_picker.models.ModelService is unavailable")
        # Preparation is an explicit setup command.  A browser job must never
        # infer that a missing bundle should be downloaded, since that would
        # violate offline inference and can block the worker for minutes.
        status_method = getattr(service_class, "status", None)
        if callable(status_method):
            try:
                status = status_method(model_root)
            except Exception as exc:
                raise RuntimeError(f"unable to read local model status: {exc}") from exc
            ready = bool(status.get("ready")) if isinstance(status, Mapping) else bool(status)
            if not ready:
                details = status.get("details") if isinstance(status, Mapping) else status
                raise RuntimeError(
                    "local model bundle is not ready; run `python -m anime_picker.models prepare` "
                    f"before starting jobs ({details})"
                )
        try:
            service = service_class(model_root=model_root, require_gpu=True)
        except TypeError:
            # Small test doubles and older local model services may not expose
            # require_gpu; the production contract above remains preferred.
            service = service_class(model_root=model_root)
        _MODEL_INSTANCE = service
        return service


def _reference_to_pil(record: ReferenceRecord) -> Any:
    if Image is None:
        raise RuntimeError("Pillow is required for reference images")
    with Image.open(record.path) as image:
        return image.convert("RGB")


def _normalize_box(box: Any, width: int, height: int) -> list[float] | None:
    if box is None:
        return None
    if isinstance(box, Mapping):
        box = box.get("box", box.get("bbox"))
    if hasattr(box, "tolist"):
        box = box.tolist()
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise HTTPException(status_code=422, detail="reference box must be normalized xyxy")
    values = [_coerce_number(item, name="reference box coordinate", minimum=0.0, maximum=1.0) for item in box]
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=422, detail="reference box must have positive area")
    return [x1, y1, x2, y2]


def _expand_box(box: Iterable[float], margin: float = 0.08) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return [max(0.0, x1 - dx), max(0.0, y1 - dy), min(1.0, x2 + dx), min(1.0, y2 + dy)]


def _crop_normalized(image: Any, box: Iterable[float]) -> Any:
    """Crop a reference image using the UI's normalized ``xyxy`` box."""

    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in box]
    left = max(0, min(width - 1, int(round(x1 * width))))
    top = max(0, min(height - 1, int(round(y1 * height))))
    right = max(left + 1, min(width, int(round(x2 * width))))
    bottom = max(top + 1, min(height, int(round(y2 * height))))
    return image.crop((left, top, right, bottom))


def _detection_box(model: Any, image: Any) -> list[float]:
    detect = getattr(model, "detect", None)
    if not callable(detect):
        raise RuntimeError("model service does not provide detect")
    detections = detect(image)
    if detections is None:
        detections = []
    if not isinstance(detections, (list, tuple)):
        detections = list(detections)

    def kind_of(item: Any) -> str:
        if isinstance(item, Mapping):
            return str(item.get("kind", item.get("label", item.get("class", "person")))).casefold()
        return str(getattr(item, "kind", getattr(item, "label", "person"))).casefold()

    def normalized_box(item: Any) -> list[float]:
        if isinstance(item, Mapping):
            box = item.get("box", item.get("bbox", item.get("xyxy")))
        else:
            box = getattr(item, "box", getattr(item, "bbox", item))
        if hasattr(box, "tolist"):
            box = box.tolist()
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("model returned an invalid reference detection")
        vals = [float(v) for v in box]
        if max(vals) <= 1.0:
            normalized = vals
        else:
            width, height = image.size
            normalized = [vals[0] / width, vals[1] / height, vals[2] / width, vals[3] / height]
        x1, y1, x2, y2 = normalized
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("model returned an invalid reference detection")
        return normalized

    # ModelService returns a person box plus one or more heads/faces inside
    # that person.  Heads are useful for pipeline detections but must not make
    # an otherwise unambiguous single-person reference require a manual crop.
    persons = [item for item in detections if kind_of(item) not in {"head", "face"}]
    if len(persons) > 1:
        raise ValueError("automatic reference detection found multiple people; select a manual crop")
    if len(persons) == 1:
        selected = normalized_box(persons[0])
        # A head outside the sole person box is evidence of another person;
        # only contained head detections are ignored as redundant evidence.
        x1, y1, x2, y2 = selected
        for head in (item for item in detections if kind_of(item) in {"head", "face"}):
            hx1, hy1, hx2, hy2 = normalized_box(head)
            if hx1 < x1 or hy1 < y1 or hx2 > x2 or hy2 > y2:
                raise ValueError("automatic reference detection found another person; select a manual crop")
        return _expand_box(selected)
    else:
        heads = [item for item in detections if kind_of(item) in {"head", "face"}]
        if len(heads) != 1:
            raise ValueError("automatic reference detection requires exactly one person; select a manual crop")
        return _expand_box(normalized_box(heads[0]))


def _read_video_frame(path: Path, time_value: float) -> Any:
    """Read one exact/nearest frame while preserving the PTS-aware path."""

    reader_class = VideoReader or _import_video_reader()
    if reader_class is not None:
        reader = reader_class(path)
        for name in ("get_frame_at_time", "frame_at_time", "read_at", "sample_frame"):
            method = getattr(reader, name, None)
            if callable(method):
                try:
                    result = method(time_value)
                    if hasattr(result, "image"):
                        return result.image
                    return result
                except (LookupError, ValueError, RuntimeError, TypeError):
                    pass
        sample = getattr(reader, "sample", None)
        if callable(sample):
            info_value = getattr(reader, "info", None)
            info_value = info_value() if callable(info_value) else info_value
            if isinstance(info_value, Mapping):
                duration = float(info_value.get("duration") or 0.0)
                fps = float(info_value.get("fps") or 0.0)
            else:
                duration = float(getattr(info_value, "duration", 0.0) or 0.0)
                fps = float(getattr(info_value, "fps", 0.0) or 0.0)
            # VideoReader.sample selects the first PTS at or after a target;
            # requesting a zero-width interval only succeeds when a frame PTS
            # lands on the decimal timestamp exactly.  Read a small forward
            # window so arbitrary UI times still use the PTS-aware reader.
            step = max(0.001, min(0.1, 1.0 / fps if fps > 0 else 0.04))
            end = time_value + max(0.1, step * 2.0)
            if duration > 0:
                end = min(duration, end)
            try:
                iterator = sample(time_value, end, step, max_side=None)
            except TypeError:
                iterator = sample(time_value, end, step)
            try:
                frame = next(iter(iterator))
                return getattr(frame, "image", frame)
            except StopIteration:
                pass

    try:
        import cv2

        capture = cv2.VideoCapture(os.fspath(path))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_value)) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("unable to decode requested video frame")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
        if Image is None:
            return frame
        return Image.fromarray(frame)
    except HTTPException:
        raise
    except Exception as exc:
        raise RuntimeError(f"unable to decode requested video frame: {exc}") from exc


def _save_pil(value: Any, path: Path) -> tuple[int, int]:
    if Image is None:
        raise RuntimeError("Pillow is required for images")
    image = value
    if not hasattr(image, "save"):
        try:
            import numpy as np

            image = Image.fromarray(np.asarray(image))
        except Exception as exc:
            raise RuntimeError("frame result is not an image") from exc
    if getattr(image, "mode", None) not in {"RGB", "RGBA"}:
        image = image.convert("RGB")
    image.save(path, format="PNG")
    return int(image.width), int(image.height)


def _load_pil(path: Path) -> Any:
    if Image is None:
        raise RuntimeError("Pillow is required for images")
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _logo_result_value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _logo_json_value(value: Any) -> Any:
    """Keep processor diagnostics JSON-safe without importing NumPy eagerly."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        try:
            return _logo_json_value(value.tolist())
        except Exception:
            return str(value)
    if isinstance(value, Mapping):
        return {str(key): _logo_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_logo_json_value(item) for item in value]
    return str(value)


def _coerce_edit_mask(mask: Any, size: tuple[int, int]) -> Any:
    """Return a Pillow L mask for a processor edit mask when possible."""

    if mask is None or Image is None:
        return None
    try:
        if hasattr(mask, "convert"):
            value = mask.convert("L")
        else:
            import numpy as np

            array = np.asarray(mask)
            if array.ndim > 2:
                array = array[..., 0]
            if array.dtype == bool:
                array = array.astype("uint8") * 255
            elif array.size:
                maximum = float(array.max())
                if maximum <= 1.0:
                    array = (array * 255.0).astype("uint8")
                else:
                    array = array.astype("uint8")
            value = Image.fromarray(array, mode="L")
        if value.size != size:
            value = value.resize(size, Image.Resampling.NEAREST)
        return value
    except Exception:
        return None


def _logo_cleanup_metadata(
    result: Any,
    *,
    requested: bool,
    profile_ids: list[str],
    original_relative_path: str | None,
) -> dict[str, Any]:
    raw = _logo_result_value(result, "metadata", {})
    if not isinstance(raw, Mapping):
        raw = {}
    metadata = dict(raw)
    # A processor must explicitly report a successful hit.  Merely returning
    # an image is not evidence that an unknown logo was safely identified.
    status = str(metadata.get("status", "unknown"))
    applied = bool(
        metadata.get(
            "applied",
            status.casefold() in {"applied", "processed", "cleaned", "ok", "done", "success"},
        )
    )
    if not requested:
        status = "disabled"
        applied = False
    result_value: dict[str, Any] = {
        "requested": bool(requested),
        "applied": bool(applied),
        "status": status,
        "profile_hits": _logo_json_value(metadata.get("profile_hits", metadata.get("profiles", []))),
        "coords": _logo_json_value(metadata.get("coords", metadata.get("coordinates", metadata.get("bounds", [])))),
        "model_hash": _logo_json_value(metadata.get("model_hash", metadata.get("modelHash", metadata.get("model_sha256")))),
        "precision": _logo_json_value(metadata.get("precision", metadata.get("dtype"))),
        "original_relative_path": original_relative_path,
        "profiles": list(profile_ids),
    }
    # Keep additional core diagnostics available without making the public
    # schema depend on a particular LogoResult implementation.
    for key, value in metadata.items():
        if key not in result_value and isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))):
            result_value[key] = _logo_json_value(value)
    return result_value


def _apply_logo_result(
    original: Any,
    result: Any,
    *,
    profile_ids: list[str] | None = None,
    original_relative_path: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Apply one LogoResult while preserving pixels outside its edit mask."""

    image = _logo_result_value(result, "image")
    if image is None or not hasattr(image, "size"):
        raise _LogoUnavailable("logo processor returned no image")
    if tuple(image.size) != tuple(original.size):
        raise _LogoUnavailable("logo processor changed the native frame dimensions")
    processed = image.convert("RGB") if getattr(image, "mode", "RGB") not in {"RGB", "RGBA"} else image
    mask = _coerce_edit_mask(_logo_result_value(result, "edit_mask"), original.size)
    if mask is not None and Image is not None:
        # Composite with the native frame rather than trusting a processor to
        # leave unrelated pixels alone.  This makes the server-side contract
        # explicit and keeps exact-region tests meaningful.
        processed = Image.composite(processed.convert("RGB"), original.convert("RGB"), mask)
    metadata = _logo_cleanup_metadata(
        result,
        requested=True,
        profile_ids=list(profile_ids or []),
        original_relative_path=original_relative_path,
    )
    return processed, metadata


def _process_logo_frame(
    original: Any,
    profile_ids: list[str],
    *,
    original_relative_path: str | None = None,
    profiles: list[Any] | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    """Process one native frame, returning an image, metadata, warnings."""

    warnings: list[str] = []
    try:
        loaded_profiles = profiles if profiles is not None else _load_logo_profiles(profile_ids)
        # The analysis worker and every logo operation share this lock.  Model
        # initialization and inference therefore serialize on the same GPU
        # boundary while the caller remains on the bounded worker thread.
        with _GPU_LOCK:
            processor = _get_logo_processor()
            # ``force=False`` is intentional: the core's conservative detector
            # must report unknown/no-hit instead of blindly masking a frame.
            result = processor.process(original, loaded_profiles, force=False)
        processed, metadata = _apply_logo_result(
            original,
            result,
            profile_ids=profile_ids,
            original_relative_path=original_relative_path,
        )
        status = str(metadata.get("status", "unknown")).casefold()
        if not metadata.get("applied") or status in {"unknown", "no_hit", "no-hit", "skipped", "disabled"}:
            metadata["applied"] = False
            if status in {"applied", "cleaned", "ok", "done"}:
                metadata["status"] = "no_hit"
            processed = original
            reason = metadata.get("reason")
            suffix = f": {reason}" if reason else ""
            warnings.append(f"logo removal skipped ({metadata.get('status', 'unknown')}){suffix}")
        return processed, metadata, warnings
    except _LogoUnavailable as exc:
        metadata = {
            "requested": True,
            "applied": False,
            "status": "skipped",
            "reason": str(exc),
            "profile_hits": [],
            "coords": [],
            "model_hash": None,
            "precision": None,
            "original_relative_path": original_relative_path,
            "profiles": list(profile_ids),
        }
        warnings.append(f"logo removal skipped: {exc}")
        return original, metadata, warnings
    except Exception as exc:
        metadata = {
            "requested": True,
            "applied": False,
            "status": "error",
            "reason": str(exc),
            "profile_hits": [],
            "coords": [],
            "model_hash": None,
            "precision": None,
            "original_relative_path": original_relative_path,
            "profiles": list(profile_ids),
        }
        warnings.append(f"logo removal error: {exc}")
        return original, metadata, warnings


def _candidate_path(job: JobRecord, candidate: Mapping[str, Any], kind: str) -> Path | None:
    value = candidate.get(f"{kind}_path", candidate.get(kind))
    if not value:
        return None
    try:
        path = Path(os.fspath(value)).resolve(strict=True)
    except (TypeError, ValueError, OSError):
        return None
    # A pipeline may use a temporary directory of its own.  Copy image files
    # into this job's registered result directory before exposing them.
    result_root = job.result_root or (RESULT_ROOT / job.id)
    result_root.mkdir(parents=True, exist_ok=True)
    if not _is_inside(path, CACHE_ROOT):
        safe = result_root / f"{_safe_filename(str(candidate.get('id', 'candidate')))}_{kind}{path.suffix.lower() or '.png'}"
        try:
            shutil.copyfile(path, safe)
            path = safe.resolve()
        except OSError:
            return None
    elif not _is_inside(path, result_root):
        safe = result_root / f"{_safe_filename(str(candidate.get('id', 'candidate')))}_{kind}{path.suffix.lower() or '.png'}"
        try:
            shutil.copyfile(path, safe)
            path = safe.resolve()
        except OSError:
            return None
    if not path.is_file() or path.stat().st_size > MAX_STATIC_BYTES * 20:
        return None
    return path


def _job_public(job: JobRecord) -> dict[str, Any]:
    public_candidates: list[dict[str, Any]] = []
    for candidate in job.candidates:
        item = {
            key: value
            for key, value in candidate.items()
            if key not in {"preview_path", "crop_path", "preview", "crop"}
            and not key.startswith("_")
        }
        cid = str(item.get("id", ""))
        if cid:
            if candidate.get("_preview_file"):
                item["preview_url"] = f"/api/jobs/{job.id}/candidates/{cid}/preview"
            if candidate.get("_crop_file"):
                item["crop_url"] = f"/api/jobs/{job.id}/candidates/{cid}/crop"
            if candidate.get("_logo_preview_file"):
                item["logo_preview_url"] = f"/api/jobs/{job.id}/candidates/{cid}/logo-preview"
        public_candidates.append(item)
    return {
        "id": job.id,
        "status": job.status,
        "progress": float(max(0.0, min(1.0, job.progress))),
        "message": job.message,
        "error": job.error,
        "warnings": job.warnings,
        "candidates": public_candidates,
        "stats": job.stats,
    }


def _call_pipeline(
    pipeline: Callable[..., Any],
    video_path: Path,
    references: list[Any],
    wanted: list[dict[str, str]],
    avoided: list[dict[str, str]],
    settings: dict[str, Any],
    models: Any,
    cache_dir: Path,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
) -> Any:
    kwargs = {
        "video_path": video_path,
        "references": references,
        "wanted": wanted,
        "avoided": avoided,
        "settings": settings,
        "models": models,
        "cache_dir": cache_dir,
        "progress": progress,
        "cancelled": cancelled,
    }
    try:
        return pipeline(**kwargs)
    except TypeError as first_error:
        # Keep compatibility with the early positional pipeline contract while
        # retaining the named contract for production and test doubles.
        try:
            return pipeline(video_path, references, wanted, avoided, settings, models, cache_dir, progress, cancelled)
        except TypeError:
            raise first_error


def _run_job(job_id: str) -> None:
    with _STATE_LOCK:
        job = JOB_REGISTRY.get(job_id)
        if job is None:
            return
        if job.cancel_event.is_set():
            job.status = "cancelled"
            job.message = "キャンセル済み"
            return
        job.status = "running"
        job.message = "モデルを準備中…"
    try:
        with _GPU_LOCK:
            if job.cancel_event.is_set():
                raise _CancelledJob()
            video = _get_video(job.video_id)
            # Model weights are a process-wide GPU resource and remain pinned
            # to the configured local default.  Per-job model_root overrides
            # are intentionally not accepted by the request schema.
            models = _load_model_service(DEFAULT_MODEL_ROOT)
            job.model_revision = getattr(models, "revision", None)
            if callable(job.model_revision):
                try:
                    job.model_revision = job.model_revision()
                except Exception:
                    job.model_revision = None
            if job.cancel_event.is_set():
                raise _CancelledJob()
            references: list[Any] = []
            for item in job.reference_items:
                reference = REFERENCE_REGISTRY[item["id"]]
                image = _reference_to_pil(reference)
                box = item.get("box")
                if box is None:
                    box = _detection_box(models, image)
                item["box"] = box
                # The crop coordinates are supplied by the canvas in normalized
                # form.  Crop before handing references to the pipeline so
                # identity features are computed on exactly the selected
                # target area; retain the normalized box in the job record for
                # metadata and diagnostics.
                references.append(_crop_normalized(image, box))
            pipeline = run_search or _import_pipeline()
            if pipeline is None:
                raise RuntimeError("anime_picker.pipeline.run_search is unavailable")

            def update(progress_value: float, message: str = "解析中…") -> None:
                with _STATE_LOCK:
                    if job.cancel_event.is_set():
                        return
                    try:
                        job.progress = max(0.0, min(1.0, float(progress_value)))
                    except (TypeError, ValueError):
                        pass
                    job.message = str(message or "解析中…")

            with _STATE_LOCK:
                job.message = "候補を探索中…"
                job.progress = max(job.progress, 0.05)
            # The pipeline keeps previews/crops under its per-job result root,
            # but expensive model features and tag outputs are shared across
            # retries.  This trusted path is injected server-side and cannot
            # be supplied by the browser.
            job.settings["_feature_cache_dir"] = (CACHE_ROOT / "features").resolve()
            result = _call_pipeline(
                pipeline,
                video.path,
                references,
                job.wanted,
                job.avoided,
                job.settings,
                models,
                RESULT_ROOT / job.id,
                update,
                job.cancel_event.is_set,
            )
            if job.cancel_event.is_set():
                raise _CancelledJob()
            if result is None:
                result = {}
            if not isinstance(result, Mapping):
                raise RuntimeError("pipeline returned an invalid result")
            candidates = result.get("candidates") or []
            normalized_candidates: list[dict[str, Any]] = []
            job.result_root = RESULT_ROOT / job.id
            job.result_root.mkdir(parents=True, exist_ok=True)
            for index, raw in enumerate(candidates):
                candidate = dict(raw) if isinstance(raw, Mapping) else {"value": raw}
                cid = str(candidate.get("id") or f"c{index + 1:04d}")
                if any(str(existing.get("id")) == cid for existing in normalized_candidates):
                    cid = f"{cid}-{index + 1}"
                candidate["id"] = cid
                preview = _candidate_path(job, candidate, "preview")
                crop = _candidate_path(job, candidate, "crop")
                candidate["_preview_file"] = os.fspath(preview) if preview else None
                candidate["_crop_file"] = os.fspath(crop) if crop else None
                normalized_candidates.append(candidate)
            with _STATE_LOCK:
                job.candidates = normalized_candidates
                job.warnings = list(result.get("warnings") or [])
                job.stats = dict(result.get("stats") or {}) if isinstance(result.get("stats") or {}, Mapping) else {}
                job.progress = 1.0
                job.message = "解析完了"
                job.status = "done"
    except _CancelledJob:
        with _STATE_LOCK:
            job.status = "cancelled"
            job.message = "キャンセル済み"
            job.progress = min(job.progress, 1.0)
    except HTTPException as exc:
        with _STATE_LOCK:
            job.status = "error"
            job.error = str(exc.detail)
            job.message = "解析に失敗しました"
    except Exception as exc:
        with _STATE_LOCK:
            job.status = "error"
            job.error = str(exc)
            job.message = "解析に失敗しました"


class _CancelledJob(Exception):
    pass


def _parse_reference_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise HTTPException(status_code=422, detail="at least one reference is required")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise HTTPException(status_code=422, detail="references must contain objects")
        reference_id = item.get("id")
        if not isinstance(reference_id, str):
            raise HTTPException(status_code=422, detail="reference id is required")
        with _STATE_LOCK:
            reference = REFERENCE_REGISTRY.get(reference_id)
        if reference is None:
            raise HTTPException(status_code=422, detail=f"unknown reference: {reference_id}")
        box = _normalize_box(item.get("box"), reference.width, reference.height)
        result.append({"id": reference_id, "box": box})
    return result


def _parse_settings(value: Any, duration: float) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise HTTPException(status_code=422, detail="settings must be an object")
    settings = dict(DEFAULT_SETTINGS)
    settings.update({key: value[key] for key in DEFAULT_SETTINGS if key in value})
    settings["start"] = _coerce_number(settings["start"], name="start", minimum=0.0)
    settings["end"] = _coerce_number(settings["end"], name="end", minimum=0.0)
    if settings["end"] and settings["end"] < settings["start"]:
        raise HTTPException(status_code=422, detail="end must be >= start")
    settings["coarse_step"] = _coerce_number(settings["coarse_step"], name="coarse_step", minimum=0.001)
    settings["dense_step"] = _coerce_number(settings["dense_step"], name="dense_step", minimum=0.001)
    settings["candidate_count"] = _coerce_int(settings["candidate_count"], name="candidate_count", minimum=1, maximum=1000)
    for key in ("identity_threshold", "tag_threshold", "quality_min", "identity_weight", "condition_weight", "quality_weight"):
        settings[key] = _coerce_number(settings[key], name=key, minimum=0.0, maximum=1.0)
    settings["rescue"] = bool(settings["rescue"])
    settings["logo_removal"] = _coerce_bool(settings["logo_removal"], name="logo_removal")
    profile_ids = _normalize_logo_profile_ids_shape(settings.get("logo_profiles", []))
    settings["logo_profiles"] = _validate_logo_profile_ids(profile_ids) if settings["logo_removal"] and profile_ids else profile_ids
    # Additional paths are accepted only for local operation and are consumed
    # by the worker.  They are never interpreted as URLs or served directly.
    if "model_root" in value:
        raise HTTPException(status_code=422, detail="model_root is fixed by local model setup")
    for key in ("output_root",):
        if key in value and value[key] is not None:
            if not isinstance(value[key], str) or "\x00" in value[key]:
                raise HTTPException(status_code=422, detail=f"{key} must be a local path")
            settings[key] = value[key]
    return settings


async def _read_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with destination.open("wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="uploaded file is too large")
                output.write(chunk)
    except BaseException:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        await upload.close()
    return total


async def _reference_from_upload(upload: UploadFile) -> ReferenceRecord:
    name = Path(upload.filename or "reference.png").name
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".png"
    reference_id = uuid.uuid4().hex
    path = REFERENCE_ROOT / f"{reference_id}{suffix}"
    await _read_upload(upload, path, MAX_REFERENCE_BYTES)
    if Image is None:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Pillow is unavailable")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = int(image.width), int(image.height)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="uploaded reference is not a valid image") from exc
    record = ReferenceRecord(reference_id, path.resolve(), name, width, height, _utc_now())
    with _STATE_LOCK:
        REFERENCE_REGISTRY[record.id] = record
    return record


def _reference_response(record: ReferenceRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "url": f"/api/references/{record.id}/image",
        "width": record.width,
        "height": record.height,
        "name": record.name,
    }


def _origin_is_local(origin: str) -> bool:
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").strip("[]").casefold()
    return host in {"localhost", "127.0.0.1", "::1", "testserver"}


def _host_is_local(host_header: str) -> bool:
    host = (host_header or "").split("@")[-1]
    if host.startswith("["):
        host = host[1:].split("]", 1)[0]
    else:
        host = host.split(":", 1)[0]
    return host.casefold() in {"localhost", "127.0.0.1", "::1", "testserver"}


def _export_with_reader(path: Path, pts: Any, destination: Path) -> None:
    reader_class = VideoReader or _import_video_reader()
    if reader_class is None:
        raise RuntimeError("VideoReader is unavailable; exact PTS export cannot continue")
    reader = reader_class(path)
    export = getattr(reader, "export_png", None)
    if not callable(export):
        raise RuntimeError("VideoReader does not provide exact export_png(pts, destination)")
    export(int(pts), destination)
    if not destination.is_file():
        raise RuntimeError("VideoReader did not create the requested PNG")


def _source_unchanged(video: VideoRecord) -> bool:
    try:
        return _reader_fingerprint(video.path) == video.fingerprint
    except OSError:
        return False


def _effective_logo_options(
    job: JobRecord,
    payload: Mapping[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Resolve export/preview logo options with strict JSON validation."""

    settings = job.settings if isinstance(job.settings, Mapping) else {}
    source = payload or {}
    if "logo_removal" in source:
        requested = _coerce_bool(source.get("logo_removal"), name="logo_removal")
    else:
        configured = settings.get("logo_removal", True)
        requested = _coerce_bool(configured, name="logo_removal") if configured is not None else True
    profile_value = source.get("logo_profiles", settings.get("logo_profiles", []))
    profiles = _normalize_logo_profile_ids_shape(profile_value)
    # Explicit logo-off must remain a native reader-only path and must not
    # import ProfileStore or LogoProcessor at all.
    if requested and profiles:
        profiles = _validate_logo_profile_ids(profiles)
    return requested, profiles


def _logo_cache_key(video: VideoRecord, candidate: Mapping[str, Any], profile_ids: list[str]) -> str:
    return _logo_cache_key_with_signature(video, candidate, profile_ids, "")


def _file_signature(path_value: Any) -> str | None:
    """Hash a local profile asset for preview cache invalidation."""

    if not isinstance(path_value, (str, os.PathLike)):
        return None
    try:
        path = Path(path_value).resolve(strict=True)
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
        return f"sha256:{digest.hexdigest()}:size:{stat.st_size}:mtime_ns:{stat.st_mtime_ns}"
    except (OSError, ValueError):
        return None


def _logo_profiles_signature(profiles: list[Any] | None) -> str:
    """Return a deterministic signature for the validated profile set.

    Auto-selection passes every currently registered profile, so adding or
    removing one changes the signature.  Selected profiles include their
    normalized geometry and the bytes/mtime of profile.json, template, and
    mask assets, so editing a calibration file with the same ID cannot reuse
    a stale preview.
    """

    entries: list[Any] = []
    for profile in profiles or []:
        if isinstance(profile, Mapping):
            public = {
                str(key): value
                for key, value in profile.items()
                if not str(key).startswith("_")
            }
            profile_dir = profile.get("_profile_dir")
            assets = {
                "profile_json": _file_signature(Path(str(profile_dir)) / "profile.json") if profile_dir else None,
                "template": _file_signature(profile.get("_template_path")),
                "custom_mask": _file_signature(profile.get("_custom_mask_path")),
            }
            entries.append({"public": _logo_json_value(public), "assets": assets})
        else:
            entries.append(_logo_json_value(profile))
    try:
        processor_class, _store_class = _logo_components()
        model_hash = getattr(processor_class, "model_sha256", None) or getattr(processor_class, "model_hash", None)
        if model_hash is None:
            try:
                from .logo_removal import MODEL_SHA256

                model_hash = MODEL_SHA256
            except Exception:
                model_hash = None
    except Exception:
        model_hash = None
    payload = {
        "version": "logo-preview-v2",
        "profiles": entries,
        "model_hash": model_hash,
        "policy": {"force": False, "processor": "LogoProcessor"},
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _logo_cache_key_with_signature(
    video: VideoRecord,
    candidate: Mapping[str, Any],
    profile_ids: list[str],
    profile_signature: str,
) -> str:
    value = {
        "source_fingerprint": video.fingerprint,
        "candidate_id": str(candidate.get("id", "")),
        "pts": candidate.get("pts"),
        "logo_removal": True,
        "logo_profiles": list(profile_ids),
        "profile_signature": profile_signature,
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _logo_original_candidate_url(job: JobRecord, candidate: Mapping[str, Any]) -> str | None:
    if candidate.get("_preview_file"):
        cid = str(candidate.get("id", ""))
        if cid:
            return f"/api/jobs/{job.id}/candidates/{cid}/preview"
    return None


def _logo_preview_worker(
    job: JobRecord,
    candidate: dict[str, Any],
    requested: bool,
    profile_ids: list[str],
) -> dict[str, Any]:
    """Create/reuse a logo preview on the bounded worker thread."""

    video = _get_video(job.video_id)
    if not _source_unchanged(video):
        raise HTTPException(status_code=409, detail="source video changed after analysis")
    if not requested:
        cleanup = _logo_cleanup_metadata({}, requested=False, profile_ids=[], original_relative_path=None)
        metadata = {
            "source_fingerprint": video.fingerprint,
            "source_pts": candidate.get("pts"),
            "source_timebase": candidate.get("timebase", video.info.get("time_base", "1/1")),
            "logo_cleanup": cleanup,
            "requested": cleanup.get("requested"),
            "applied": cleanup.get("applied"),
            "status": cleanup.get("status"),
            "profile_hits": cleanup.get("profile_hits"),
            "coords": cleanup.get("coords"),
            "model_hash": cleanup.get("model_hash"),
            "precision": cleanup.get("precision"),
            "original_relative_path": cleanup.get("original_relative_path"),
        }
        return {
            "preview_url": _logo_original_candidate_url(job, candidate),
            "metadata": metadata,
            "warnings": [],
        }
    try:
        loaded_profiles = _load_logo_profiles(profile_ids)
    except Exception:
        # Processing below will preserve the original and report the concrete
        # load error.  The unavailable signature also prevents reuse of an
        # artifact produced from a previously valid profile set.
        loaded_profiles = None
    profile_signature = _logo_profiles_signature(loaded_profiles)
    result_root = (job.result_root or RESULT_ROOT / job.id).resolve()
    cache_dir = result_root / "logo-preview" / _logo_cache_key_with_signature(
        video, candidate, profile_ids, profile_signature
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    original_path = cache_dir / "original.png"
    processed_path = cache_dir / "preview.png"
    metadata_path = cache_dir / "metadata.json"
    original_relative = original_path.relative_to(result_root).as_posix()
    if metadata_path.is_file() and processed_path.is_file():
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached_cleanup = cached.get("logo_cleanup", {}) if isinstance(cached, Mapping) else {}
            if isinstance(cached_cleanup, Mapping) and bool(cached_cleanup.get("applied")):
                candidate["_logo_preview_file"] = os.fspath(processed_path)
                return {
                    "preview_url": f"/api/jobs/{job.id}/candidates/{candidate.get('id')}/logo-preview",
                    "metadata": cached,
                    "warnings": list(cached.get("warnings") or []),
                }
        except (OSError, ValueError, TypeError):
            pass
    pts = candidate.get("pts")
    if pts is None:
        raise HTTPException(status_code=422, detail="candidate has no exact PTS")
    # Recheck immediately before touching the source.  A source edit while a
    # preview waits behind analysis must never produce a stale artifact.
    if not _source_unchanged(video):
        raise HTTPException(status_code=409, detail="source video changed after analysis")
    _export_with_reader(video.path, pts, original_path)
    original = _load_pil(original_path)
    processed, cleanup, warnings = _process_logo_frame(
        original,
        profile_ids,
        original_relative_path=original_relative,
        profiles=loaded_profiles,
    )
    metadata = {
        "candidate_id": candidate.get("id"),
        "source_fingerprint": video.fingerprint,
        "source_pts": candidate.get("pts"),
        "source_timebase": candidate.get("timebase", video.info.get("time_base", "1/1")),
        "logo_profiles": list(profile_ids),
        "profile_signature": profile_signature,
        "logo_cleanup": cleanup,
        "requested": cleanup.get("requested"),
        "applied": cleanup.get("applied"),
        "status": cleanup.get("status"),
        "profile_hits": cleanup.get("profile_hits"),
        "coords": cleanup.get("coords"),
        "model_hash": cleanup.get("model_hash"),
        "precision": cleanup.get("precision"),
        "original_relative_path": cleanup.get("original_relative_path"),
        "warnings": warnings,
        "created_at": _utc_now(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if cleanup.get("applied"):
        _save_pil(processed, processed_path)
        candidate["_logo_preview_file"] = os.fspath(processed_path)
        preview_url = f"/api/jobs/{job.id}/candidates/{candidate.get('id')}/logo-preview"
    else:
        # A no-hit/unknown/error result is explicitly represented and the UI
        # continues to show its original candidate preview.
        preview_url = _logo_original_candidate_url(job, candidate)
    return {"preview_url": preview_url, "metadata": metadata, "warnings": warnings}


def _export_job(
    job: JobRecord,
    candidate_ids: list[str],
    logo_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    video = _get_video(job.video_id)
    if not _source_unchanged(video):
        raise HTTPException(status_code=409, detail="source video changed after analysis")
    selected = [candidate for candidate in job.candidates if str(candidate.get("id")) in candidate_ids]
    if not selected:
        raise HTTPException(status_code=422, detail="no valid candidates selected")
    logo_requested, logo_profile_ids = _effective_logo_options(job, logo_options)
    output_value = job.settings.get("output_root") or DEFAULT_OUTPUT_ROOT
    try:
        output_root = _resolve_output_root(output_value)
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail="invalid output_root") from exc
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"unable to create output directory: {exc}") from exc
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    directory = output_root / f"{_safe_filename(video.path.stem)}_{timestamp}_{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    files: list[str] = []
    original_files: list[str] = []
    metadata_candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        originals_root = directory / "originals"
        if logo_requested:
            originals_root.mkdir(parents=True, exist_ok=True)
        for index, candidate in enumerate(selected, start=1):
            pts = candidate.get("pts")
            if pts is None:
                raise RuntimeError(f"candidate {candidate.get('id')} has no exact PTS")
            destination = directory / f"frame_{index:04d}.png"
            original_relative_path: str | None = None
            logo_cleanup: dict[str, Any]
            if logo_requested:
                original_destination = originals_root / destination.name
                original_relative_path = original_destination.relative_to(directory).as_posix()
                _export_with_reader(video.path, pts, original_destination)
                if not _source_unchanged(video):
                    raise HTTPException(status_code=409, detail="source video changed after analysis")
                original_image = _load_pil(original_destination)
                processed, logo_cleanup, frame_warnings = _process_logo_frame(
                    original_image,
                    logo_profile_ids,
                    original_relative_path=original_relative_path,
                )
                warnings.extend(f"frame {index}: {warning}" for warning in frame_warnings)
                if logo_cleanup.get("applied"):
                    _save_pil(processed, destination)
                else:
                    # Copying preserves the exact decoded native pixels for a
                    # no-hit/unknown/error result and avoids a PNG re-encode.
                    shutil.copyfile(original_destination, destination)
                original_files.append(os.fspath(original_destination))
            else:
                # This branch intentionally does not resolve/import any logo
                # component and stays on the original exact-PTS reader path.
                _export_with_reader(video.path, pts, destination)
                logo_cleanup = _logo_cleanup_metadata(
                    {}, requested=False, profile_ids=[], original_relative_path=None
                )
            if not destination.is_file():
                raise RuntimeError(f"candidate {candidate.get('id')} was not exported")
            files.append(os.fspath(destination))
            metadata_candidates.append(
                {
                    "id": candidate.get("id"),
                    "pts": candidate.get("pts"),
                    "time": candidate.get("time"),
                    "timebase": candidate.get("timebase", video.info.get("time_base", "1/1")),
                    "crop": candidate.get("bbox", candidate.get("crop")),
                    "scores": {
                        key: candidate.get(key)
                        for key in ("identity", "distance", "quality", "condition", "score", "tags")
                        if key in candidate
                    },
                    "logo_cleanup": logo_cleanup,
                    # Alias kept for consumers that use the setting name as
                    # the metadata namespace.
                    "logo_removal": logo_cleanup,
                    "logo": logo_cleanup,
                    "logo_requested": logo_cleanup.get("requested"),
                    "logo_applied": logo_cleanup.get("applied"),
                    "logo_status": logo_cleanup.get("status"),
                    "logo_profile_hits": logo_cleanup.get("profile_hits"),
                    "logo_coords": logo_cleanup.get("coords"),
                    "logo_model_hash": logo_cleanup.get("model_hash"),
                    "logo_precision": logo_cleanup.get("precision"),
                    "original_relative_path": original_relative_path,
                }
            )
        metadata: dict[str, Any] = {
            "source_fingerprint": video.fingerprint,
            "source_pts": [candidate.get("pts") for candidate in metadata_candidates],
            "source_timebase": video.info.get("time_base", "1/1"),
            "source_path": os.fspath(video.path),
            "models": {"revision": job.model_revision},
            "settings": job.settings,
            "candidates": metadata_candidates,
            "logo_removal": {
                "requested": logo_requested,
                "profiles": list(logo_profile_ids),
            },
            "originals": original_files,
            "original_files": original_files,
            "warnings": warnings,
            "created_at": _utc_now(),
        }
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        files.append(os.fspath(metadata_path))
        return {
            "directory": os.fspath(directory),
            "files": files,
            "originals": original_files,
            "original_files": original_files,
            "metadata": metadata,
            "warnings": warnings,
            "metadata_path": os.fspath(metadata_path),
        }
    except HTTPException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"export failed: {exc}") from exc


def create_app() -> FastAPI:
    _ensure_dirs()
    api = FastAPI(title="Anime Frame Picker", docs_url=None, redoc_url=None)

    @api.middleware("http")
    async def local_only(request: Request, call_next: Callable[..., Any]) -> Any:
        if not _host_is_local(request.headers.get("host", "")):
            return JSONResponse({"detail": "loopback host required"}, status_code=400)
        origin = request.headers.get("origin")
        if origin and not _origin_is_local(origin):
            return JSONResponse({"detail": "cross-origin requests are not allowed"}, status_code=403)
        return await call_next(request)

    if STATIC_ROOT.is_dir():
        api.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

    @api.get("/")
    async def index() -> Any:
        index_path = STATIC_ROOT / "index.html"
        if not index_path.is_file():
            return JSONResponse({"service": "anime-picker"})
        return FileResponse(index_path, media_type="text/html")

    @api.get("/api/bootstrap")
    async def bootstrap() -> dict[str, Any]:
        videos: list[dict[str, Any]] = []
        for path in _discover_videos():
            # Do not probe or fingerprint every source during page load: a
            # local episode can be hundreds of megabytes.  Metadata and the
            # source fingerprint are acquired only after the user selects a
            # video through /inspect.
            videos.append({"path": os.fspath(path), "name": path.name})
        bootstrap_aliases, bootstrap_vocab = _aliases_and_vocabulary()
        bootstrap_aliases = {alias: target for alias, target in bootstrap_aliases.items() if target in bootstrap_vocab}
        return {
            "videos": videos,
            "defaults": {
                "model_root": os.fspath(DEFAULT_MODEL_ROOT),
                "output_root": os.fspath(DEFAULT_OUTPUT_ROOT),
            },
            "models": _model_status(),
            "aliases": bootstrap_aliases,
            "settings": dict(DEFAULT_SETTINGS),
        }

    @api.post("/api/videos/inspect")
    async def inspect_video(payload: dict[str, Any]) -> dict[str, Any]:
        path = _check_video_path(payload.get("path"))
        record = _register_video(path)
        return _info_response(record)

    @api.post("/api/videos/upload")
    async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
        name = Path(file.filename or "upload.mp4").name
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(status_code=422, detail="only MP4 and MKV videos are supported")
        path = UPLOAD_ROOT / f"{uuid.uuid4().hex}{suffix}"
        await _read_upload(file, path, MAX_UPLOAD_BYTES)
        try:
            record = _register_video(path)
        except HTTPException:
            path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=f"uploaded video is not readable: {exc}") from exc
        return _info_response(record)

    @api.post("/api/references")
    async def upload_references(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        if not files:
            raise HTTPException(status_code=422, detail="at least one reference image is required")
        records: list[ReferenceRecord] = []
        try:
            for file in files:
                records.append(await _reference_from_upload(file))
        except HTTPException:
            for record in records:
                record.path.unlink(missing_ok=True)
                with _STATE_LOCK:
                    REFERENCE_REGISTRY.pop(record.id, None)
            raise
        return {"references": [_reference_response(record) for record in records]}

    @api.get("/api/references/{reference_id}/image")
    async def reference_image(reference_id: str) -> Any:
        with _STATE_LOCK:
            record = REFERENCE_REGISTRY.get(reference_id)
        if record is None or not re.fullmatch(r"[0-9a-fA-F]{16,64}", reference_id or ""):
            raise HTTPException(status_code=404, detail="reference not found")
        if not record.path.is_file() or not _is_inside(record.path, REFERENCE_ROOT):
            raise HTTPException(status_code=404, detail="reference not found")
        media_type = mimetypes.guess_type(record.path.name)[0] or "application/octet-stream"
        return FileResponse(record.path, media_type=media_type)

    @api.post("/api/videos/{video_id}/reference")
    async def capture_reference(video_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        video = _get_video(video_id)
        time_value = _coerce_number(payload.get("time", 0.0), name="time", minimum=0.0)
        if video.info.get("duration") and time_value > float(video.info["duration"]):
            raise HTTPException(status_code=422, detail="time is outside video duration")
        try:
            image = _read_video_frame(video.path, time_value)
            reference_id = uuid.uuid4().hex
            path = REFERENCE_ROOT / f"{reference_id}.png"
            width, height = _save_pil(image, path)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"unable to read video frame: {exc}") from exc
        record = ReferenceRecord(reference_id, path.resolve(), f"{video.path.stem}_{time_value:.3f}s.png", width, height, _utc_now())
        with _STATE_LOCK:
            REFERENCE_REGISTRY[record.id] = record
        result = _reference_response(record)
        result["time"] = time_value
        return result

    @api.get("/api/vocabulary")
    async def vocabulary(q: str = "") -> dict[str, Any]:
        aliases, vocab = _aliases_and_vocabulary()
        aliases = {alias: target for alias, target in aliases.items() if target in vocab}
        query = q.strip().casefold()
        tags = []
        for name, category in sorted(vocab.items(), key=lambda item: item[0].casefold()):
            if query and query not in name.casefold() and not any(query in alias.casefold() and target == name for alias, target in aliases.items()):
                continue
            tags.append({"name": name, "category": category, "scope": _scope_for_category(category)})
        return {"tags": tags, "aliases": aliases, "ready": _model_status()["ready"]}

    @api.get("/api/logo-profiles")
    async def logo_profiles() -> dict[str, Any]:
        # Listing profiles reads only the local profile manifest.  It never
        # prepares/downloads a model and does not initialize LogoProcessor.
        loop = asyncio.get_running_loop()
        profiles, model = await asyncio.gather(
            loop.run_in_executor(_executor(), _logo_profiles_public),
            loop.run_in_executor(_executor(), _logo_status),
        )
        return {
            "profiles": profiles,
            "default_enabled": True,
            "model": model,
            "profile_root": os.fspath(_logo_profile_root()),
        }

    @api.post("/api/jobs")
    async def create_job(payload: dict[str, Any]) -> dict[str, Any]:
        video_id = payload.get("video_id")
        if not isinstance(video_id, str):
            raise HTTPException(status_code=422, detail="video_id is required")
        video = _get_video(video_id)
        references = _parse_reference_items(payload.get("references"))
        wanted = _normalize_tags(payload.get("wanted", []), field="wanted")
        avoided = _normalize_tags(payload.get("avoided", []), field="avoided")
        settings = _parse_settings(payload.get("settings"), float(video.info.get("duration") or 0.0))
        job = JobRecord(uuid.uuid4().hex, video_id, references, wanted, avoided, settings)
        with _STATE_LOCK:
            JOB_REGISTRY[job.id] = job
        job.future = _executor().submit(_run_job, job.id)
        return {"job_id": job.id}

    @api.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        with _STATE_LOCK:
            job = JOB_REGISTRY.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_public(job)

    @api.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        with _STATE_LOCK:
            job = JOB_REGISTRY.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="job not found")
            if job.status in {"done", "error", "cancelled"}:
                return _job_public(job)
            job.cancel_event.set()
            job.message = "キャンセル要求済み"
            if job.status == "queued":
                job.status = "cancelled"
        return _job_public(job)

    @api.get("/api/jobs/{job_id}/candidates/{candidate_id}/{kind}")
    async def candidate_asset(job_id: str, candidate_id: str, kind: str) -> Any:
        if kind not in {"preview", "crop", "logo-preview"}:
            raise HTTPException(status_code=404, detail="asset not found")
        with _STATE_LOCK:
            job = JOB_REGISTRY.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", candidate_id):
            raise HTTPException(status_code=404, detail="asset not found")
        candidate = next((item for item in job.candidates if str(item.get("id")) == candidate_id), None)
        if candidate is None:
            raise HTTPException(status_code=404, detail="asset not found")
        path_key = "_logo_preview_file" if kind == "logo-preview" else f"_{kind}_file"
        path_value = candidate.get(path_key)
        if not path_value:
            raise HTTPException(status_code=404, detail="asset not found")
        try:
            path = Path(path_value).resolve(strict=True)
        except (OSError, ValueError):
            raise HTTPException(status_code=404, detail="asset not found")
        result_root = (job.result_root or RESULT_ROOT / job.id).resolve()
        if not _is_inside(path, result_root) or not path.is_file():
            raise HTTPException(status_code=404, detail="asset not found")
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "image/png")

    @api.post("/api/jobs/{job_id}/candidates/{candidate_id}/logo-preview")
    async def logo_preview(job_id: str, candidate_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with _STATE_LOCK:
            job = JOB_REGISTRY.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "done":
            raise HTTPException(status_code=409, detail="job is not complete")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", candidate_id or ""):
            raise HTTPException(status_code=404, detail="candidate not found")
        candidate = next((item for item in job.candidates if str(item.get("id")) == candidate_id), None)
        if candidate is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        body = payload or {}
        if not isinstance(body, Mapping):
            raise HTTPException(status_code=422, detail="logo preview payload must be an object")
        requested, profile_ids = _effective_logo_options(job, body)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _executor(),
            _logo_preview_worker,
            job,
            candidate,
            requested,
            profile_ids,
        )
        return result

    @api.post("/api/jobs/{job_id}/export")
    async def export_job(job_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with _STATE_LOCK:
            job = JOB_REGISTRY.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "done":
            raise HTTPException(status_code=409, detail="job is not complete")
        body = payload or {}
        if not isinstance(body, Mapping):
            raise HTTPException(status_code=422, detail="export payload must be an object")
        candidate_ids = body.get("candidate_ids", [])
        if not isinstance(candidate_ids, list):
            raise HTTPException(status_code=422, detail="candidate_ids must be an array")
        selected_ids = [str(item) for item in candidate_ids] if candidate_ids else [str(item.get("id")) for item in job.candidates]
        logo_options = {
            key: body[key]
            for key in ("logo_removal", "logo_profiles")
            if key in body
        }
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor(), _export_job, job, selected_ids, logo_options)

    return api


app = create_app()


def shutdown() -> None:
    """Release the local worker during tests or embedding shutdown."""

    global _EXECUTOR
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _EXECUTOR = None


__all__ = [
    "app",
    "create_app",
    "shutdown",
    "VIDEO_REGISTRY",
    "REFERENCE_REGISTRY",
    "JOB_REGISTRY",
    "VideoRecord",
    "ReferenceRecord",
    "JobRecord",
]
