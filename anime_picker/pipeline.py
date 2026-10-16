"""Local coarse-to-dense anime character frame search.

The pipeline owns orchestration only.  Video decoding is supplied by
``VideoReader`` and identity/tag inference is supplied by ``ModelService`` (the
parent application wires both in).  A candidate is created from a detector crop;
there is intentionally no full-frame identity fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover - the server installs Pillow
    Image = Any  # type: ignore[assignment,misc]

from .scoring import (
    aggregate_reference_distances,
    clamp01,
    compact_tag_evidence,
    combine_score,
    deduplicate_candidates,
    dhash,
    evaluate_tags,
    identity_match,
    normalize_weights,
    phash,
    quality_metrics,
    subject_tag_ambiguity,
)

try:  # The parent application can replace this symbol in tests or at startup.
    from .video import VideoReader  # type: ignore
except Exception:  # pragma: no cover - import is intentionally lazy/adapter-friendly
    VideoReader = None  # type: ignore[assignment,misc]


Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]


class PipelineError(RuntimeError):
    """Raised by internal adapters for malformed model/video responses."""


_FATAL_MODEL_ERROR_NAMES = {
    "ModelUnavailable",
    "GPURequiredError",
    "ImportError",
    "ModuleNotFoundError",
}


def _is_fatal_model_error(error: BaseException) -> bool:
    """Recognize infrastructure failures without importing the model module."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        if name in _FATAL_MODEL_ERROR_NAMES or isinstance(current, (ImportError, ModuleNotFoundError)):
            return True
        message = str(current).lower()
        if "cuda" in message and any(word in message for word in ("unavailable", "required", "provider", "runtime")):
            return True
        current = current.__cause__ or current.__context__
    return False


def _model_failure(stage: str, error: BaseException) -> PipelineError:
    if _is_fatal_model_error(error):
        return PipelineError(f"{stage} model runtime unavailable: {error}")
    return PipelineError(f"{stage} model inference failed repeatedly: {error}")


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _as_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    return str(value)


def _revision_token(*values: Any) -> str:
    payload = json.dumps([_safe_json_value(value) for value in values], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _image_digest(image: Any) -> str:
    """Digest image pixels, size, and mode for model/tag cache keys."""

    if hasattr(image, "convert"):
        rgb = image.convert("RGB")
        raw = np.asarray(rgb, dtype=np.uint8)
        descriptor = f"RGB:{rgb.width}x{rgb.height}:".encode("ascii")
    else:
        raw = np.asarray(image)
        descriptor = f"array:{raw.shape}:{raw.dtype}:".encode("ascii")
    return hashlib.sha256(descriptor + raw.tobytes()).hexdigest()


def _ensure_rgb(image: Any) -> Any:
    if hasattr(image, "convert"):
        return image.convert("RGB")
    if Image is not Any:
        return Image.fromarray(np.asarray(image).astype(np.uint8), mode="RGB")
    return image


class _BoundedCache:
    """Small JSON/NumPy cache with deterministic keys and size pruning.

    The cache never serializes model objects or arbitrary Python values.  Feature
    arrays use ``.npy`` with pickle disabled; tags and metadata use JSON.
    """

    def __init__(self, root: Path, *, max_items: int = 2048, max_bytes: int = 256 * 1024 * 1024) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.max_items = max(1, int(max_items))
        self.max_bytes = max(1, int(max_bytes))
        self.index: dict[str, dict[str, Any]] = {}
        try:
            loaded = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                self.index = {str(key): dict(value) for key, value in loaded.items() if isinstance(value, Mapping)}
        except (OSError, ValueError, TypeError):
            self.index = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(kind: str, digest: str, revision: str) -> str:
        return hashlib.sha256(f"{kind}|{revision}|{digest}".encode("utf-8")).hexdigest()

    def _path(self, key: str, suffix: str) -> Path:
        return self.root / f"{key}.{suffix}"

    def _touch(self, key: str, path: Path) -> None:
        now = time.time()
        entry = self.index.setdefault(key, {})
        entry.update({"path": path.name, "bytes": path.stat().st_size if path.exists() else 0, "atime": now})
        try:
            self.index_path.write_text(json.dumps(self.index, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        self._prune()

    def _drop(self, key: str) -> None:
        entry = self.index.pop(key, {})
        path_name = entry.get("path")
        if path_name:
            try:
                (self.root / str(path_name)).unlink(missing_ok=True)
            except OSError:
                pass

    def _prune(self) -> None:
        def size_of(entry: Mapping[str, Any]) -> int:
            return max(0, _as_int(entry.get("bytes")))

        while len(self.index) > self.max_items or sum(size_of(entry) for entry in self.index.values()) > self.max_bytes:
            if not self.index:
                break
            oldest = min(self.index, key=lambda item: _as_number(self.index[item].get("atime"), 0.0))
            self._drop(oldest)
        try:
            self.index_path.write_text(json.dumps(self.index, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def get_array(self, key: str) -> np.ndarray | None:
        path = self._path(key, "npy")
        if not path.exists():
            self.misses += 1
            return None
        try:
            array = np.load(path, allow_pickle=False)
            self.hits += 1
            self._touch(key, path)
            return np.asarray(array)
        except (OSError, ValueError, TypeError):
            self.misses += 1
            self._drop(key)
            return None

    def put_array(self, key: str, array: np.ndarray) -> None:
        path = self._path(key, "npy")
        try:
            np.save(path, np.asarray(array), allow_pickle=False)
            self._touch(key, path)
        except (OSError, ValueError, TypeError):
            pass

    def get_json(self, key: str) -> Any | None:
        path = self._path(key, "json")
        if not path.exists():
            self.misses += 1
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            self.hits += 1
            self._touch(key, path)
            return value
        except (OSError, ValueError, TypeError):
            self.misses += 1
            self._drop(key)
            return None

    def put_json(self, key: str, value: Any) -> None:
        path = self._path(key, "json")
        try:
            path.write_text(json.dumps(_safe_json_value(value), ensure_ascii=False), encoding="utf-8")
            self._touch(key, path)
        except (OSError, TypeError, ValueError):
            pass


@dataclass
class _ModelBundle:
    detector: Any
    identity: Any
    tagger: Any
    revision: Any
    identity_model: Any


def _resolve_model_bundle(models: Any) -> _ModelBundle | None:
    """Accept a ModelService or a small mapping of detector/identity/tagger services."""

    if models is None:
        return None
    if isinstance(models, Mapping):
        identity = models.get("identity") or models.get("ccip") or models.get("service") or models.get("model")
        detector = models.get("detector") or models.get("detection") or identity
        tagger = models.get("tagger") or models.get("tags") or models.get("service") or identity
        if identity is None and callable(models.get("embed")):
            identity = models
        if detector is None:
            detector = models
        if tagger is None:
            tagger = models
    else:
        identity = getattr(models, "identity", None) or getattr(models, "ccip", None) or models
        detector = getattr(models, "detector", None) or models
        tagger = getattr(models, "tagger", None) or getattr(models, "tags", None) or models
    if identity is None or not callable(getattr(identity, "embed", None)) or not callable(getattr(identity, "distance", None)):
        return None
    if not callable(getattr(detector, "detect", None)):
        return None
    # Identity features must follow the manifest revision even when the model
    # service also exposes a finer tagger/runtime revision.
    revision = getattr(identity, "identity_revision", None)
    if callable(revision):
        try:
            revision = revision()
        except Exception:
            revision = None
    if revision is None:
        revision = getattr(identity, "revision", None)
    if callable(revision):
        try:
            revision = revision()
        except Exception:
            revision = None
    if revision is None:
        revision = (
            getattr(models, "identity_revision", None)
            or getattr(models, "revision", None)
            if not isinstance(models, Mapping)
            else models.get("identity_revision") or models.get("revision")
        )
        if callable(revision):
            try:
                revision = revision()
            except Exception:
                revision = None
    identity_model = getattr(identity, "identity_model", None)
    if callable(identity_model):
        try:
            identity_model = identity_model()
        except Exception:
            identity_model = None
    if identity_model is None:
        identity_model = getattr(models, "identity_model", None) if not isinstance(models, Mapping) else models.get("identity_model")
        if callable(identity_model):
            try:
                identity_model = identity_model()
            except Exception:
                identity_model = None
    return _ModelBundle(detector, identity, tagger, revision or "unknown", identity_model or "ccip")


def _model_threshold(identity: Any, settings: Mapping[str, Any]) -> float:
    model_threshold = _as_number(getattr(identity, "threshold", 0.0), 0.0)
    if callable(getattr(identity, "threshold", None)):
        try:
            model_threshold = _as_number(identity.threshold(), model_threshold)
        except Exception:
            pass
    configured = _first(settings, "identity_threshold", "identityThreshold", default=None)
    if configured is None:
        configured = model_threshold
    threshold = _as_number(configured, model_threshold)
    if threshold < 0:
        threshold = model_threshold
    # A UI sensitivity can tighten the official boundary, but must not make an
    # official CCIP match permissive beyond the service's manifest threshold.
    if model_threshold > 0 and threshold > model_threshold:
        threshold = model_threshold
    return threshold


def _frame_value(frame: Any, key: str, default: Any = None) -> Any:
    if isinstance(frame, Mapping):
        return frame.get(key, default)
    return getattr(frame, key, default)


def _frame_time_value(frame: Any) -> float | None:
    """Read a relative frame timestamp without inventing one for a neighbor."""

    if frame is None:
        return None
    value = _frame_value(frame, "time", _frame_value(frame, "timestamp", None))
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _frame_image(frame: Any) -> Any | None:
    image = _frame_value(frame, "image")
    if image is not None:
        return image
    # A few light adapters expose PIL frames as a tuple-like object.
    if hasattr(frame, "__array__") or hasattr(frame, "convert"):
        return frame
    return None


def _coerce_frames(value: Any) -> Iterator[Any]:
    if value is None:
        return
    if isinstance(value, Mapping) and "image" in value:
        yield value
        return
    if _frame_image(value) is not None:
        yield value
        return
    try:
        yield from iter(value)
    except TypeError:
        return


def _video_info_value(info: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(info, Mapping):
        return _first(info, *keys, default=default)
    for key in keys:
        value = getattr(info, key, None)
        if value is not None:
            return value
    return default


def _parse_detection(raw: Any, width: int, height: int, index: int, min_score: float) -> dict[str, Any] | None:
    if isinstance(raw, Mapping):
        box = raw.get("box", raw.get("bbox"))
        confidence = _as_number(raw.get("score", raw.get("confidence", 1.0)), 0.0)
        kind = str(raw.get("kind", "person")).lower()
    else:
        # Adapters may return ``(xyxy, confidence)`` while the public contract
        # uses dictionaries.  Accept both without changing detector semantics.
        if isinstance(raw, (tuple, list, np.ndarray)) and len(raw) == 2 and isinstance(raw[0], (tuple, list, np.ndarray)):
            box = raw[0]
            confidence = _as_number(raw[1], 0.0)
        else:
            box = raw
            confidence = 1.0
        kind = "person"
    if confidence < min_score or box is None:
        return None
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    # The service contract is pixel coordinates.  Normalized coordinates are
    # accepted only as a harmless adapter convenience for lightweight callers.
    if max(abs(value) for value in values) <= 1.5 and max(width, height) > 4:
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    x0, x1 = sorted((max(0.0, min(width, values[0])), max(0.0, min(width, values[2]))))
    y0, y1 = sorted((max(0.0, min(height, values[1])), max(0.0, min(height, values[3]))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return {
        "box": [x0, y0, x1, y1],
        "score": confidence,
        "kind": "head" if kind == "head" else "person",
        "index": index,
    }


def _normalized_box(box: Sequence[float], width: int, height: int) -> list[float]:
    return [
        clamp01(_as_number(box[0]) / max(1, width)),
        clamp01(_as_number(box[1]) / max(1, height)),
        clamp01(_as_number(box[2]) / max(1, width)),
        clamp01(_as_number(box[3]) / max(1, height)),
    ]


def _crop_image(image: Any, box: Sequence[float]) -> Any:
    if not hasattr(image, "crop"):
        raise PipelineError("VideoReader frames must expose PIL-like crop")
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    return _ensure_rgb(image.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))))


def _box_contains(outer: Sequence[float], inner: Sequence[float], tolerance: float = 1e-6) -> bool:
    return (
        float(outer[0]) <= float(inner[0]) + tolerance
        and float(outer[1]) <= float(inner[1]) + tolerance
        and float(outer[2]) >= float(inner[2]) - tolerance
        and float(outer[3]) >= float(inner[3]) - tolerance
    )


def _head_crop_box(
    detection: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    settings: Mapping[str, Any],
) -> list[float]:
    """Add a small hair/context margin to a head crop when it is safe."""

    original = [float(value) for value in detection["box"]]
    if detection.get("kind") != "head":
        return original
    margin = max(0.0, _as_number(_first(settings, "head_margin", default=0.10), 0.10))
    if margin <= 0:
        return original
    x0, y0, x1, y1 = original
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    expanded = [
        max(0.0, x0 - dx),
        max(0.0, y0 - dy),
        min(float(width), x1 + dx),
        min(float(height), y1 + dy),
    ]
    overlap_iou = _as_number(_first(settings, "overlap_iou", default=0.35), 0.35)
    for other in detections:
        if other is detection:
            continue
        other_box = other["box"]
        # A head inside its owning person box is expected and should not disable
        # the margin. Adjacent person/head boxes are treated as contamination.
        if other.get("kind") == "person" and _box_contains(other_box, original):
            continue
        if _box_iou_pixel(expanded, other_box) >= overlap_iou:
            return original
    return expanded


def _intervals_around(centers: Sequence[float], radius: float, start: float, end: float) -> list[tuple[float, float]]:
    intervals: list[list[float]] = []
    for center in sorted(float(value) for value in centers):
        left, right = max(start, center - radius), min(end, center + radius)
        if right <= left:
            continue
        if intervals and left <= intervals[-1][1] + 1e-6:
            intervals[-1][1] = max(intervals[-1][1], right)
        else:
            intervals.append([left, right])
    return [(left, right) for left, right in intervals]


def _clean_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:150]


def _save_image(image: Any, path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = _ensure_rgb(image)
        rgb.save(path, format="JPEG", quality=90, optimize=True)
        return path.exists()
    except (OSError, ValueError, TypeError):
        return False


def _match_rank(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    """Rank an identity match before or after tag scoring."""

    # The score combines identity/condition/quality; identity and official
    # distance break ties in favor of the stronger body/head crop. No
    # perceptual or CCIP duplicate logic is introduced here.
    return (
        _as_number(candidate.get("score"), 0.0),
        _as_number(candidate.get("identity"), 0.0),
        -_as_number(candidate.get("distance"), float("inf")),
        _as_number(candidate.get("quality"), 0.0),
        _as_number(candidate.get("condition"), 0.0),
    )


def _collapse_matches_per_pts(matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep one strongest surviving match for each exact presentation timestamp."""

    grouped: dict[int, dict[str, Any]] = {}
    order: list[int] = []

    for index, raw in enumerate(matches):
        candidate = dict(raw)
        pts = _as_int(candidate.get("pts"), index)
        candidate["_merged_detections"] = max(1, _as_int(candidate.get("_merged_detections"), 1))
        existing = grouped.get(pts)
        if existing is None:
            grouped[pts] = candidate
            order.append(pts)
            continue
        merged_count = _as_int(existing.get("_merged_detections"), 1) + candidate["_merged_detections"]
        if _match_rank(candidate) > _match_rank(existing):
            candidate["_merged_detections"] = merged_count
            grouped[pts] = candidate
        else:
            existing["_merged_detections"] = merged_count
    return [grouped[pts] for pts in order]


def _nested_head_person_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two same-PTS matches are nested head/person detections."""

    if _as_int(left.get("pts"), -1) != _as_int(right.get("pts"), -1):
        return False
    kinds = {str(left.get("_detection_kind", "")).lower(), str(right.get("_detection_kind", "")).lower()}
    if kinds != {"head", "person"}:
        return False
    left_box = left.get("_detector_bbox", left.get("bbox"))
    right_box = right.get("_detector_bbox", right.get("bbox"))
    if not isinstance(left_box, Sequence) or not isinstance(right_box, Sequence) or len(left_box) != 4 or len(right_box) != 4:
        return False
    return _box_contains(left_box, right_box) or _box_contains(right_box, left_box) or _box_iou_pixel(left_box, right_box) >= 0.50


def _collapse_nested_head_person(matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse only contained head/person duplicates, preserving distinct people."""

    result: list[dict[str, Any]] = []
    consumed: set[int] = set()
    for index, raw in enumerate(matches):
        if index in consumed:
            continue
        best = dict(raw)
        merged_count = max(1, _as_int(best.get("_merged_detections"), 1))
        for other_index in range(index + 1, len(matches)):
            if other_index in consumed or not _nested_head_person_matches(best, matches[other_index]):
                continue
            other = dict(matches[other_index])
            merged_count += max(1, _as_int(other.get("_merged_detections"), 1))
            consumed.add(other_index)
            if _match_rank(other) > _match_rank(best):
                best = other
        best["_merged_detections"] = merged_count
        result.append(best)
    return result


def _pixel_diverse_tag_pool(
    matches: Sequence[Mapping[str, Any]],
    *,
    hash_distance: int,
    time_window: float,
    exact_only: bool = False,
) -> list[dict[str, Any]]:
    """Reduce near-identical frames before the expensive tagger stage.

    Distinct detections at one PTS are protected until scope gating has run;
    final export selection still enforces one candidate per PTS.  This keeps a
    second person in the same frame available when only that crop satisfies a
    subject tag.
    """

    by_pts: dict[int, list[Mapping[str, Any]]] = {}
    for index, item in enumerate(matches):
        by_pts.setdefault(_as_int(item.get("pts"), index), []).append(item)
    protected: list[dict[str, Any]] = []
    unprotected: list[Mapping[str, Any]] = []
    for group in by_pts.values():
        if len(group) > 1:
            protected.extend(dict(item) for item in group)
        else:
            unprotected.extend(group)
    if exact_only:
        # Condition searches must retain visually different crops until the
        # expensive tagger has evaluated them. Collapse only byte-identical
        # frame+crop pixels (the feature cache already keys by the same digest).
        ordered = sorted(
            (dict(item) for item in unprotected),
            key=lambda item: (-_as_number(item.get("score"), 0.0), _as_number(item.get("time"), 0.0), str(item.get("id", ""))),
        )
        diverse: list[dict[str, Any]] = []
        exact_seen: list[tuple[dict[str, Any], str, str]] = []
        for candidate in ordered:
            frame_digest = _image_digest(candidate["_frame"])
            crop_digest = _image_digest(candidate["_crop"])
            duplicate = any(
                frame_digest == old_frame
                and crop_digest == old_crop
                and abs(_as_number(candidate.get("time"), 0.0) - _as_number(old.get("time"), 0.0)) <= time_window
                for old, old_frame, old_crop in exact_seen
            )
            if duplicate:
                continue
            diverse.append(candidate)
            exact_seen.append((candidate, frame_digest, crop_digest))
    else:
        diverse = deduplicate_candidates(
            unprotected,
            hash_distance=hash_distance,
            time_window=time_window,
        )
    return protected + diverse


def _call_progress(progress: Progress, value: float, message: str) -> None:
    try:
        progress(clamp01(value), message)
    except Exception:
        # UI progress is advisory; a callback failure must not corrupt a search.
        pass


def _is_cancelled(cancelled: Cancelled) -> bool:
    try:
        return bool(cancelled())
    except Exception:
        return False


def _embed_cached(
    service: Any,
    images: Sequence[Any],
    cache: _BoundedCache,
    revision: str,
    kind: str,
) -> np.ndarray:
    if not images:
        return np.empty((0, 0), dtype=np.float32)
    digests = [_image_digest(image) for image in images]
    keys = [cache.key(kind, digest, revision) for digest in digests]
    found: list[np.ndarray | None] = [cache.get_array(key) for key in keys]
    missing_indexes = [index for index, item in enumerate(found) if item is None]
    if missing_indexes:
        computed = service.embed([images[index] for index in missing_indexes])
        computed_array = np.asarray(computed)
        if computed_array.ndim == 1:
            computed_array = computed_array.reshape(1, -1)
        if computed_array.ndim != 2 or computed_array.shape[0] != len(missing_indexes):
            raise PipelineError("ModelService.embed must return one feature row per image")
        for offset, index in enumerate(missing_indexes):
            found[index] = np.asarray(computed_array[offset])
            cache.put_array(keys[index], found[index])
    try:
        return np.stack([np.asarray(item) for item in found if item is not None]).astype(np.float32)
    except (ValueError, TypeError) as error:
        raise PipelineError("cached identity features have inconsistent shapes") from error


def _tag_cached(
    service: Any,
    image: Any,
    cache: _BoundedCache,
    revision: str,
    kind: str,
) -> Mapping[str, Any]:
    key = cache.key(kind, _image_digest(image), revision)
    cached = cache.get_json(key)
    if isinstance(cached, Mapping):
        return cached
    if not callable(getattr(service, "tag", None)):
        return {}
    value = service.tag(image)
    if not isinstance(value, Mapping):
        value = {}
    cache.put_json(key, value)
    return value


def _distance_matrix(service: Any, references: np.ndarray, features: np.ndarray) -> np.ndarray:
    distances = np.asarray(service.distance(references, features))
    if distances.ndim == 0:
        distances = distances.reshape(1, 1)
    elif distances.ndim == 1:
        if references.shape[0] == 1:
            distances = distances.reshape(1, -1)
        elif features.shape[0] == 1:
            distances = distances.reshape(-1, 1)
    if distances.shape != (references.shape[0], features.shape[0]):
        raise PipelineError(
            f"ModelService.distance returned {distances.shape}; expected {(references.shape[0], features.shape[0])}"
        )
    return distances.astype(np.float32)


def run_search(
    video_path: str,
    references: list[Any],
    wanted: list[dict[str, Any]],
    avoided: list[dict[str, Any]],
    settings: dict[str, Any],
    models: Any,
    cache_dir: Path,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> dict[str, Any]:
    """Search *video_path* for reference-matching character crops.

    Stages run as coarse sampling, local dense refinement around promising
    matches, near-frame refinement, then an optional global rescue scan when the
    requested result count is still short.  The returned paths are local cache
    paths; the HTTP layer can map them to URLs.
    """

    settings = settings or {}
    progress = progress or (lambda _value, _message: None)
    cancelled = cancelled or (lambda: False)
    warnings: list[str] = []
    warning_keys: set[str] = set()
    stats: dict[str, Any] = {
        "sampled": 0,
        "coarse_frames": 0,
        "dense_frames": 0,
        "near_frames": 0,
        "rescue_frames": 0,
        "sample_calls": 0,
        "detections": 0,
        "identity_crops": 0,
        "identity_matches": 0,
        "tagged": 0,
        "subject_ambiguity": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "rejected": {},
        "cancelled": False,
        "budget_exhausted": False,
    }
    result_empty = {"candidates": [], "warnings": warnings, "stats": stats}

    def warn_once(message: str) -> None:
        text = str(message)
        if text not in warning_keys:
            warning_keys.add(text)
            warnings.append(text)

    def reject(reason: str, count: int = 1) -> None:
        rejected = stats.setdefault("rejected", {})
        rejected[reason] = int(rejected.get(reason, 0)) + count

    def stop(reason: str | None = None) -> bool:
        if _is_cancelled(cancelled):
            stats["cancelled"] = True
            warn_once("cancelled")
            return True
        if reason and reason not in warnings:
            warn_once(reason)
        return False

    bundle = _resolve_model_bundle(models)
    if bundle is None:
        warn_once("identity/detection ModelService is unavailable")
        stats["error"] = "models_unavailable"
        return result_empty
    if _is_cancelled(cancelled):
        stats["cancelled"] = True
        warn_once("cancelled")
        return result_empty
    if not references:
        warn_once("no references supplied")
        stats["error"] = "no_references"
        return result_empty
    if VideoReader is None:
        warn_once("VideoReader is unavailable")
        stats["error"] = "video_reader_unavailable"
        return result_empty

    root = Path(cache_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        warn_once(f"cache directory unavailable: {error}")
        stats["error"] = "cache_unavailable"
        return result_empty
    model_revision = _revision_token(bundle.identity_model, bundle.revision)
    tag_model = getattr(bundle.tagger, "tag_model", None)
    tag_model_revision = getattr(bundle.tagger, "tag_revision", None)
    if tag_model_revision is None:
        tag_model_revision = getattr(bundle.tagger, "precision_revision", None)
    if tag_model_revision is None:
        tag_model_revision = getattr(bundle.tagger, "revision", None)
    if callable(tag_model):
        try:
            tag_model = tag_model()
        except Exception:
            tag_model = None
    if callable(tag_model_revision):
        try:
            tag_model_revision = tag_model_revision()
        except Exception:
            tag_model_revision = None
    tag_revision = _revision_token(tag_model, tag_model_revision, model_revision)
    # Result files remain job-local so the HTTP layer can safely expose them.
    # The server may provide an internal shared cache root for model outputs;
    # it is deliberately underscore-prefixed and is never accepted from the
    # public settings parser.  Model revision + pixel digest keys keep different
    # model bundles isolated within that root.
    shared_feature_cache = _first(settings, "_feature_cache_dir", default=None)
    feature_cache_root = Path(shared_feature_cache) if shared_feature_cache else root / "model-cache"
    cache = _BoundedCache(
        feature_cache_root,
        max_items=_as_int(_first(settings, "cache_max_items", "max_cache_items", default=2048), 2048),
        max_bytes=_as_int(_first(settings, "cache_max_bytes", default=256 * 1024 * 1024), 256 * 1024 * 1024),
    )

    try:
        reader = VideoReader(video_path)
    except Exception as error:
        warn_once(f"could not open video: {error}")
        stats["error"] = "video_open_failed"
        return result_empty

    info = getattr(reader, "info", None)
    duration = _as_number(_video_info_value(info, "duration", "duration_seconds", "length", "duration_sec", default=0.0), 0.0)
    start = max(0.0, _as_number(_first(settings, "start", "start_time", "startTime", default=0.0), 0.0))
    configured_end = _as_number(_first(settings, "end", "end_time", "endTime", default=0.0), 0.0)
    end = configured_end if configured_end > start else duration
    if end <= start:
        end = start + max(1.0, _as_number(_first(settings, "fallback_duration", default=60.0), 60.0))
        warn_once("video duration is unavailable; using fallback search interval")
    if duration > 0:
        end = min(end, duration)
    if end <= start:
        warn_once("empty search interval")
        stats["error"] = "empty_interval"
        return result_empty

    target_count = max(1, _as_int(_first(settings, "candidate_count", "candidateCount", "target_count", "limit", default=30), 30))
    coarse_step = max(0.01, _as_number(_first(settings, "coarse_step", "coarseStep", default=2.0), 2.0))
    dense_step = max(0.005, _as_number(_first(settings, "dense_step", "denseStep", default=0.25), 0.25))
    fps = _as_number(_video_info_value(info, "fps", "frame_rate", "frameRate", default=0.0), 0.0)
    frame_step = 1.0 / fps if fps > 0 else dense_step / 4.0
    near_default = frame_step
    near_step = max(0.002, _as_number(_first(settings, "near_step", "nearStep", default=near_default), near_default))
    temporal_max_gap = max(2.0 / fps, 0.12) if fps > 0 else 0.12
    rescue_step = max(0.002, _as_number(_first(settings, "rescue_step", "rescueStep", default=dense_step / 2.0), dense_step / 2.0))
    max_samples = max(1, _as_int(_first(settings, "max_samples", "max_sample_budget", "sample_budget", default=2500), 2500))
    identity_threshold = _model_threshold(bundle.identity, settings)
    quality_min = clamp01(_as_number(_first(settings, "quality_min", "qualityMin", default=0.0), 0.0))
    tag_threshold = clamp01(_as_number(_first(settings, "tag_threshold", "tagThreshold", default=0.17), 0.17))
    min_detection_score = clamp01(_as_number(_first(settings, "min_detection_score", "detection_threshold", default=0.0), 0.0))
    max_tagged = max(target_count * 5, _as_int(_first(settings, "max_tagged", default=target_count * 8), target_count * 8))
    weights = normalize_weights(
        _first(settings, "weights", default={
            "identity": _first(settings, "identity_weight", "identityWeight", default=0.35),
            "condition": _first(settings, "condition_weight", "conditionWeight", default=0.30),
            "quality": _first(settings, "quality_weight", "qualityWeight", default=0.35),
        })
    )
    stats.update({"target": target_count, "duration": duration, "start": start, "end": end, "fps": fps, "frame_step": frame_step, "near_step": near_step, "temporal_max_gap": temporal_max_gap, "identity_threshold": identity_threshold, "weights": weights})
    fingerprint = getattr(reader, "fingerprint", None)
    if callable(fingerprint):
        try:
            fingerprint = fingerprint()
        except Exception:
            fingerprint = None
    if fingerprint is not None:
        stats["fingerprint"] = str(fingerprint)

    try:
        reference_images = [_ensure_rgb(image) for image in references]
        reference_features = _embed_cached(bundle.identity, reference_images, cache, model_revision, "identity-ref")
    except PipelineError:
        raise
    except Exception as error:
        if _is_fatal_model_error(error):
            raise _model_failure("reference embedding", error) from error
        warn_once(f"reference embedding failed: {error}")
        stats["error"] = "reference_embedding_failed"
        return result_empty

    seen_frames: set[tuple[Any, int]] = set()
    raw_matches: list[dict[str, Any]] = []
    same_pts_collapsed = 0
    sample_counter = 0
    unique_frame_counter = {"coarse": 0, "dense": 0, "near": 0, "rescue": 0}
    warned_budget = False
    detection_error_streak = 0
    identity_error_streak = 0
    tag_error_streak = 0

    def sample_interval(left: float, right: float, step: float, stage: str, progress_start: float, progress_end: float) -> None:
        nonlocal sample_counter, warned_budget
        if stop():
            return
        if stats["sampled"] >= max_samples:
            stats["budget_exhausted"] = True
            if not warned_budget:
                warn_once(f"sample budget {max_samples} reached during {stage} scan")
                warned_budget = True
            return
        stats["sample_calls"] += 1
        try:
            sampled = reader.sample(left, right, step, max_side=960)
        except TypeError:
            # Keep compatibility with a test adapter that predates max_side while
            # preferring the required API on normal readers.
            try:
                sampled = reader.sample(left, right, step)
            except Exception as error:
                warn_once(f"{stage} sampling failed: {error}")
                reject(f"{stage}_sampling_failed")
                return
        except Exception as error:
            warn_once(f"{stage} sampling failed: {error}")
            reject(f"{stage}_sampling_failed")
            return
        iterator = iter(_coerce_frames(sampled))
        previous: Any | None = None
        current = next(iterator, None)
        following = next(iterator, None)
        while current is not None:
            if stop():
                return
            image = _frame_image(current)
            if image is None:
                reject("invalid_frame")
            else:
                pts = _as_int(_frame_value(current, "pts", sample_counter), sample_counter)
                stream_index = _frame_value(current, "stream_index", 0)
                frame_key = (stream_index, pts)
                if frame_key not in seen_frames:
                    if stats["sampled"] >= max_samples:
                        stats["budget_exhausted"] = True
                        if not warned_budget:
                            warn_once(f"sample budget {max_samples} reached during {stage} scan")
                            warned_budget = True
                        return
                    seen_frames.add(frame_key)
                    stats["sampled"] += 1
                    sample_counter += 1
                    unique_frame_counter[stage] += 1
                    stats[f"{stage}_frames"] = unique_frame_counter[stage]
                    process_frame(current, previous, following, stage)
                    _call_progress(
                        progress,
                        progress_start + (progress_end - progress_start) * clamp01(stats["sampled"] / max_samples),
                        f"{stage} scan ({stats['sampled']} frames)",
                    )
            previous = current
            current = following
            following = next(iterator, None)

    def process_frame(frame: Any, previous: Any | None, following: Any | None, stage: str) -> None:
        nonlocal detection_error_streak, identity_error_streak
        image = _frame_image(frame)
        if image is None:
            return
        image = _ensure_rgb(image)
        width, height = image.size if hasattr(image, "size") else (np.asarray(image).shape[1], np.asarray(image).shape[0])
        current_time_value = _frame_time_value(frame)
        frame_time = current_time_value if current_time_value is not None else 0.0
        previous_time = _frame_time_value(previous)
        following_time = _frame_time_value(following)
        quality_settings = dict(settings)
        quality_settings["temporal_require_times"] = True
        quality_settings["temporal_max_gap"] = temporal_max_gap
        try:
            detections_raw = bundle.detector.detect(image)
        except Exception as error:
            detection_error_streak += 1
            if _is_fatal_model_error(error) or detection_error_streak >= 3:
                raise _model_failure("detection", error) from error
            reject("detection_failed")
            warn_once(f"detection inference failed (will abort after 3 consecutive errors): {error}")
            return
        detection_error_streak = 0
        detections: list[dict[str, Any]] = []
        for index, raw_detection in enumerate(detections_raw or []):
            parsed = _parse_detection(raw_detection, int(width), int(height), index, min_detection_score)
            if parsed is not None:
                detections.append(parsed)
        stats["detections"] += len(detections)
        if not detections:
            identity_error_streak = 0
            reject("no_person_detection")
            return
        valid_crops: list[tuple[dict[str, Any], Any, dict[str, Any], bool]] = []
        for detection in detections:
            box = detection["box"]
            area_fraction = ((box[2] - box[0]) * (box[3] - box[1])) / max(1.0, width * height)
            # A detector box that is effectively the whole frame cannot prove
            # which person supplied the identity evidence (even if it reports a
            # single box), so never send that box through CCIP or expose it as a
            # candidate.
            if area_fraction >= 0.985:
                reject("whole_frame_detection" if len(detections) == 1 else "multi_person_full_frame")
                continue
            crop_box = _head_crop_box(detection, detections, int(width), int(height), settings)
            try:
                crop = _crop_image(image, crop_box)
            except Exception:
                reject("crop_failed")
                continue
            normalized = _normalized_box(crop_box, int(width), int(height))
            # A second overlapping person makes identity evidence ambiguous.  A
            # head inside its own person box is allowed by the kind distinction.
            severe_overlap = False
            for other in detections:
                if other is detection:
                    continue
                if detection["kind"] == "head" and other["kind"] == "person":
                    continue
                if other["kind"] == "head" and detection["kind"] == "person":
                    continue
                if _box_iou_pixel(box, other["box"]) >= _as_number(_first(settings, "overlap_iou", default=0.35), 0.35):
                    severe_overlap = True
                    break
            metrics = quality_metrics(
                image,
                normalized,
                _frame_image(previous),
                _frame_image(following),
                quality_settings,
                previous_time=previous_time,
                following_time=following_time,
                current_time=current_time_value,
            )
            if metrics["quality"] < quality_min:
                reject("quality_below_min")
                continue
            if metrics["transition_penalty"] > 0.70:
                reject("transition")
                continue
            if severe_overlap and bool(_first(settings, "reject_severe_overlap", default=True)):
                reject("severe_overlap")
                continue
            valid_crops.append((detection, crop, metrics, severe_overlap))
        if not valid_crops:
            identity_error_streak = 0
            return
        stats["identity_crops"] += len(valid_crops)
        try:
            crop_features = _embed_cached(bundle.identity, [item[1] for item in valid_crops], cache, model_revision, "identity-crop")
            distances = _distance_matrix(bundle.identity, reference_features, crop_features)
            identity_error_streak = 0
        except PipelineError:
            raise
        except Exception as error:
            identity_error_streak += 1
            if _is_fatal_model_error(error) or identity_error_streak >= 3:
                raise _model_failure("identity", error) from error
            reject("identity_failed", len(valid_crops))
            warn_once(f"identity inference failed (will abort after 3 consecutive errors): {error}")
            return
        frame_pts = _as_int(_frame_value(frame, "pts", 0), 0)
        stream_index = _frame_value(frame, "stream_index", 0)
        for crop_index, (detection, crop, metrics, severe_overlap) in enumerate(valid_crops):
            match = aggregate_reference_distances(distances[:, crop_index], identity_threshold)
            if not match["accepted"]:
                reject("identity_threshold")
                continue
            stats["identity_matches"] += 1
            candidate_id = f"pts-{frame_pts}-stream-{stream_index}-box-{detection['index']}"
            preliminary = combine_score(match["identity"], metrics["condition"], metrics["quality"], weights)
            raw_matches.append(
                {
                    "id": candidate_id,
                    "time": frame_time,
                    "pts": frame_pts,
                    "stream_index": stream_index,
                    "bbox": _normalized_box(detection["box"], int(width), int(height)),
                    "distance": float(match["distance"]),
                    "identity": float(match["identity"]),
                    "corroboration": float(match["corroboration"]),
                    "quality": float(metrics["quality"]),
                    "condition": float(metrics["condition"] * (0.55 if severe_overlap else 1.0)),
                    "score": float(preliminary),
                    "quality_metrics": metrics,
                    "stage": stage,
                    "tags": {"subject": {}, "frame": {}},
                    "_detection_kind": detection["kind"],
                    "_detector_bbox": _normalized_box(detection["box"], int(width), int(height)),
                    "_frame": image,
                    "_crop": crop,
                    "_severe_overlap": severe_overlap,
                }
            )

    def sample_intervals(intervals: Sequence[tuple[float, float]], step: float, stage: str, progress_start: float, progress_end: float) -> None:
        if not intervals:
            return
        for index, (left, right) in enumerate(intervals):
            if stop() or stats["budget_exhausted"]:
                return
            base = progress_start + (progress_end - progress_start) * index / max(1, len(intervals))
            tail = progress_start + (progress_end - progress_start) * (index + 1) / max(1, len(intervals))
            sample_interval(left, right, step, stage, base, tail)

    dedup_hash_distance = max(0, _as_int(_first(settings, "dedup_hash_distance", default=6), 6))
    dedup_time_window = max(0.0, _as_number(_first(settings, "dedup_time_window", default=2.0), 2.0))
    subject_scope_requested = any(
        str(item.get("scope", "subject")).lower() != "frame"
        for item in [*(wanted or []), *(avoided or [])]
    )
    conditions_requested = bool(wanted or avoided)

    def prepare_pixel_diverse_pool(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        for item in items:
            item.setdefault(
                "hashes",
                {
                    "phash": phash(item["_frame"]),
                    "dhash": dhash(item["_frame"]),
                    "crop_phash": phash(item["_crop"]),
                    "crop_dhash": dhash(item["_crop"]),
                },
            )
        return _pixel_diverse_tag_pool(
            items,
            hash_distance=dedup_hash_distance,
            time_window=dedup_time_window,
            exact_only=conditions_requested,
        )

    coarse_intervals = [(start, end)]
    sample_intervals(coarse_intervals, coarse_step, "coarse", 0.0, 0.25)

    def centers_from_matches(items: Sequence[Mapping[str, Any]], maximum: int) -> list[float]:
        ordered = sorted(items, key=lambda item: (float(item.get("distance", float("inf"))), -float(item.get("quality", 0.0))))
        centers: list[float] = []
        for item in ordered:
            center = _as_number(item.get("time"), 0.0)
            if all(abs(center - existing) > max(near_step, dense_step) * 0.65 for existing in centers):
                centers.append(center)
            if len(centers) >= maximum:
                break
        return centers

    if not stop() and not stats["budget_exhausted"]:
        dense_centers = centers_from_matches(raw_matches, max(3, min(target_count * 3, 60)))
        dense_intervals = _intervals_around(dense_centers, max(coarse_step, dense_step * 2.0), start, end)
        sample_intervals(dense_intervals, dense_step, "dense", 0.25, 0.55)

    if not stop() and not stats["budget_exhausted"]:
        near_centers = centers_from_matches(raw_matches, max(3, min(target_count * 2, 40)))
        near_intervals = _intervals_around(near_centers, max(dense_step * 2.0, near_step * 3.0), start, end)
        sample_intervals(near_intervals, near_step, "near", 0.55, 0.74)

    # Person and head detectors commonly report nested boxes at one PTS. For
    # subject-scope conditions, retain both alternatives through tag gating so a
    # body/context crop can satisfy full-body/pose tags when a head crop cannot.
    # Identity-only and frame-only searches can premerge the same-target pair.
    if not subject_scope_requested:
        before_pts_collapse = len(raw_matches)
        raw_matches = _collapse_nested_head_person(raw_matches)
        same_pts_collapsed += before_pts_collapse - len(raw_matches)

    rescue_enabled = bool(_first(settings, "rescue", "rescue_enabled", default=True))
    # Reserve a second global pass whenever the identity shortlist is too small
    # to survive tag gating and pixel deduplication. This also covers the case
    # where the first pass found enough raw matches but all were near-duplicates.
    # Hashes are pixel-only and do not invoke the identity model. This pool is
    # both the rescue-capacity signal and the input to expensive tag inference.
    pixel_diverse_matches = prepare_pixel_diverse_pool(raw_matches)
    pre_dedup_capacity = len(pixel_diverse_matches)
    rescue_needed = len(raw_matches) < max(target_count * 2, target_count + 4) or pre_dedup_capacity < target_count
    if rescue_enabled and rescue_needed and not stop() and not stats["budget_exhausted"]:
        sample_intervals([(start, end)], rescue_step, "rescue", 0.74, 0.90)

    # Repeat the same policy after optional rescue has added more detections.
    if not subject_scope_requested:
        before_pts_collapse = len(raw_matches)
        raw_matches = _collapse_nested_head_person(raw_matches)
        same_pts_collapsed += before_pts_collapse - len(raw_matches)

    pixel_diverse_matches = prepare_pixel_diverse_pool(raw_matches)
    pre_dedup_capacity = len(pixel_diverse_matches)

    # Tagging is deliberately delayed until after identity matching and refinement.
    prelim_matches = sorted(pixel_diverse_matches, key=lambda item: (-float(item["score"]), float(item["distance"]), float(item["time"])))
    tagged_matches: list[dict[str, Any]] = []
    tags_requested = conditions_requested
    subject_tag_requests = [
        item for item in [*(wanted or []), *(avoided or [])] if str(item.get("scope", "subject")).lower() != "frame"
    ]
    frame_tag_requests = [
        item for item in [*(wanted or []), *(avoided or [])] if str(item.get("scope", "subject")).lower() == "frame"
    ]
    for index, candidate in enumerate(prelim_matches[:max_tagged]):
        if stop():
            break
        subject_tags: Mapping[str, Any] = {}
        frame_tags: Mapping[str, Any] = {}
        if tags_requested:
            try:
                if any(str(item.get("scope", "subject")).lower() != "frame" for item in [*(wanted or []), *(avoided or [])]):
                    subject_tags = _tag_cached(bundle.tagger, candidate["_crop"], cache, tag_revision, "tag-subject")
                if any(str(item.get("scope", "subject")).lower() == "frame" for item in [*(wanted or []), *(avoided or [])]):
                    frame_tags = _tag_cached(bundle.tagger, candidate["_frame"], cache, tag_revision, "tag-frame")
                stats["tagged"] += 1
                tag_error_streak = 0
            except PipelineError:
                raise
            except Exception as error:
                tag_error_streak += 1
                if _is_fatal_model_error(error) or tag_error_streak >= 3:
                    raise _model_failure("tag", error) from error
                warn_once(f"tag inference failed (will abort after 3 consecutive errors): {error}")
                reject("tag_failed")
                continue
            if subject_tag_requests:
                ambiguity_threshold = clamp01(
                    _as_number(_first(settings, "_subject_ambiguity_threshold", default=0.8), 0.8)
                )
                ambiguity = subject_tag_ambiguity(subject_tags, ambiguity_threshold)
                if ambiguity:
                    stats["subject_ambiguity"] += 1
                    reject("subject_ambiguous_multiple")
                    warn_once(
                        "subject tag gate rejected a crop with strong multi-person general tags "
                        f"(threshold {ambiguity_threshold:.2f})"
                    )
                    continue
            tag_result = evaluate_tags(subject_tags, frame_tags, wanted, avoided, tag_threshold)
            if not tag_result["passed"]:
                reject("tag_gate")
                continue
            candidate["tag_score"] = float(tag_result["score"])
            candidate["tag_reasons"] = tag_result["reasons"]
            candidate["_score_condition"] = clamp01(float(candidate["condition"]) * float(tag_result["score"]))
        else:
            candidate["tag_score"] = 1.0
            candidate["_score_condition"] = candidate["condition"]
        # Keep full maps available only for the gate above and the local cache;
        # public result evidence is bounded and retains explicit low-confidence
        # requested tags for explainability.
        candidate["tags"] = {
            "subject": compact_tag_evidence(subject_tags, subject_tag_requests, tag_threshold) if tags_requested else {},
            "frame": compact_tag_evidence(frame_tags, frame_tag_requests, tag_threshold) if tags_requested else {},
        }
        candidate["score"] = combine_score(candidate["identity"], candidate["_score_condition"], candidate["quality"], weights)
        # Pixel-only hashes are computed after identity/tag filtering; they are
        # solely for deduplication and never feed the CCIP decision.
        candidate.setdefault(
            "hashes",
            {
                "phash": phash(candidate["_frame"]),
                "dhash": dhash(candidate["_frame"]),
                "crop_phash": phash(candidate["_crop"]),
                "crop_dhash": dhash(candidate["_crop"]),
            },
        )
        tagged_matches.append(candidate)
        _call_progress(progress, 0.90 + 0.09 * (index + 1) / max(1, min(max_tagged, len(prelim_matches))), "tag and rank")

    # Apply the one-result-per-PTS export policy only after desired/avoided tag
    # gating and final score calculation, so a distinct person can satisfy a
    # subject tag even when another person in the same frame cannot.
    before_pts_collapse = len(tagged_matches)
    tagged_matches = _collapse_matches_per_pts(tagged_matches)
    same_pts_collapsed += before_pts_collapse - len(tagged_matches)

    deduped = deduplicate_candidates(
        tagged_matches,
        hash_distance=max(0, _as_int(_first(settings, "dedup_hash_distance", default=6), 6)),
        time_window=max(0.0, _as_number(_first(settings, "dedup_time_window", default=0.55), 0.55)),
    )
    output_candidates: list[dict[str, Any]] = []
    output_root = root / "results"
    for candidate in deduped[:target_count]:
        if stop():
            break
        safe_id = _clean_filename(str(candidate["id"]))
        preview_path = output_root / f"{safe_id}-preview.jpg"
        crop_path = output_root / f"{safe_id}-crop.jpg"
        if not _save_image(candidate["_frame"], preview_path):
            warnings.append(f"could not save preview for {candidate['id']}")
        if not _save_image(candidate["_crop"], crop_path):
            warnings.append(f"could not save crop for {candidate['id']}")
        output_candidates.append(
            {
                "id": candidate["id"],
                "time": float(candidate["time"]),
                "pts": int(candidate["pts"]),
                "stream_index": candidate.get("stream_index", 0),
                "bbox": [float(value) for value in candidate["bbox"]],
                "identity": float(candidate["identity"]),
                "distance": float(candidate["distance"]),
                "corroboration": float(candidate.get("corroboration", 0.0)),
                "quality": float(candidate["quality"]),
                "condition": float(candidate["condition"]),
                "uncertain": bool(candidate.get("_severe_overlap", False)),
                "merged_detections": int(candidate.get("_merged_detections", 1)),
                "score": float(candidate["score"]),
                "tags": candidate.get("tags", {"subject": {}, "frame": {}}),
                "tag_score": float(candidate.get("tag_score", 1.0)),
                "quality_metrics": _safe_json_value(candidate.get("quality_metrics", {})),
                "preview_path": str(preview_path.resolve()),
                "crop_path": str(crop_path.resolve()),
            }
        )
    shortage = max(0, target_count - len(output_candidates))
    if shortage:
        if not raw_matches:
            shortage_reason = "no_identity_match"
        elif not tagged_matches:
            shortage_reason = "tag_or_quality_gate"
        elif stats["budget_exhausted"]:
            shortage_reason = "sample_budget"
        elif len(deduped) < len(tagged_matches):
            shortage_reason = "pixel_deduplication"
        else:
            shortage_reason = "insufficient_matching_frames"
        warnings.append(f"shortage: {shortage} candidate(s) below requested {target_count} ({shortage_reason})")
    stats.update(
        {
            "returned": len(output_candidates),
            "shortage": shortage,
            "shortage_reason": shortage_reason if shortage else None,
            "raw_matches": len(raw_matches),
            "same_pts_collapsed": same_pts_collapsed,
            "pre_dedup_capacity": pre_dedup_capacity,
            "pixel_diverse_pool": len(pixel_diverse_matches),
            "tagged_pool": min(len(pixel_diverse_matches), max_tagged),
            "tagged_matches": len(tagged_matches),
            "deduped": len(deduped),
            "metrics": {
                "sampled_frames": stats["sampled"],
                "detections": stats["detections"],
                "identity_crops": stats["identity_crops"],
                "identity_matches": stats["identity_matches"],
                "tagged": stats["tagged"],
                "subject_ambiguity": stats["subject_ambiguity"],
                "same_pts_collapsed": same_pts_collapsed,
                "rejected": dict(stats.get("rejected", {})),
            },
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "model_revision": str(bundle.revision),
            "identity_model": str(bundle.identity_model),
        }
    )
    _call_progress(progress, 1.0, "complete")
    return {"candidates": output_candidates, "warnings": warnings, "stats": stats}


def _box_iou_pixel(left: Sequence[float], right: Sequence[float]) -> float:
    lx0, ly0, lx1, ly1 = map(float, left)
    rx0, ry0, rx1, ry1 = map(float, right)
    ix0, iy0, ix1, iy1 = max(lx0, rx0), max(ly0, ry0), min(lx1, rx1), min(ly1, ry1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


__all__ = ["PipelineError", "run_search"]
