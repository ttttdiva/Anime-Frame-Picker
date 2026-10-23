"""Reusable, resumable low-cost frame analysis for headless batches.

The Web UI's reference search remains in ``pipeline.run_search``.  This module
provides the separate batch path: scan every presentation frame cheaply, pick
scene/visual-change candidates, then run detector and CCIP once for those PTS.
Its checkpoints are keyed by source fingerprint and the relevant processing
revisions, so all/manual/automatic batch modes can share compatible evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np
from PIL import Image

SCAN_REVISION = "batch-scan-v3"
IDENTITY_PREPROCESS_REVISION = "detector960-ccip384-rgb-v1"
IDENTITY_CHECKPOINT_REVISION = "identity-checkpoint-v2"
SCAN_MAX_SIDE = 960
THUMBNAIL_SIZE = (48, 27)
DEFAULT_CHUNK_SECONDS = 5.0


@dataclass(frozen=True)
class ScannedFrame:
    pts: int
    time: float
    sharpness: float
    brightness: float
    contrast: float
    thumbnail: np.ndarray


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex[:8] + ".tmp")
    try:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _json_digest(value: Any) -> str:
    packed = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _metadata_array(value: Mapping[str, Any]) -> np.ndarray:
    packed = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return np.frombuffer(packed.encode("utf-8"), dtype=np.uint8)


def _metadata_from_array(value: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype != np.dtype(np.uint8):
        raise ValueError("checkpoint metadata must be a one-dimensional byte array")
    decoded = json.loads(raw.tobytes().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("checkpoint metadata must contain a JSON object")
    return decoded


def _gray_small(image: Image.Image) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image)
    return np.asarray(image.convert("L").resize(THUMBNAIL_SIZE, resampling.BILINEAR), dtype=np.uint8)


def _sharpness(image: Image.Image) -> tuple[float, float, float]:
    """Small, non-gating image diagnostics; dark/flat frames remain eligible."""

    resampling = getattr(Image, "Resampling", Image)
    gray = np.asarray(image.convert("L").resize((160, 90), resampling.BILINEAR), dtype=np.float32)
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(np.var(lap)), float(np.mean(gray)), float(np.std(gray))


def _load_chunk(npz_path: Path, expected: Mapping[str, Any]) -> list[ScannedFrame] | None:
    if not npz_path.is_file():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            metadata = _metadata_from_array(data["metadata"])
            raw_pts = np.asarray(data["pts"])
            raw_times = np.asarray(data["times"])
            raw_sharpness = np.asarray(data["sharpness"])
            raw_brightness = np.asarray(data["brightness"])
            raw_contrast = np.asarray(data["contrast"])
            raw_thumbnails = np.asarray(data["thumbnails"])

        if metadata.get("key") != expected.get("key"):
            return None
        frame_count = metadata.get("frame_count")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 0:
            return None
        if frame_count != len(raw_pts):
            return None
        for field in ("start", "end"):
            actual = metadata.get(field)
            wanted = expected.get(field)
            if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                    or not math.isfinite(float(actual)) or wanted is None
                    or not math.isclose(float(actual), float(wanted), rel_tol=0.0, abs_tol=1e-9)):
                return None
        include_end = expected.get("include_end")
        if not isinstance(include_end, bool) or metadata.get("include_end") is not include_end:
            return None
        if raw_pts.ndim != 1 or raw_pts.dtype != np.dtype(np.int64):
            return None
        if raw_times.ndim != 1 or raw_times.dtype != np.dtype(np.float64):
            return None
        if any(array.ndim != 1 or array.dtype != np.dtype(np.float32) for array in
               (raw_sharpness, raw_brightness, raw_contrast)):
            return None
        if raw_thumbnails.dtype != np.dtype(np.uint8) or raw_thumbnails.shape != (frame_count, *THUMBNAIL_SIZE[::-1]):
            return None
        if not (len(raw_pts) == len(raw_times) == len(raw_sharpness) == len(raw_brightness) == len(raw_contrast)):
            return None
        if len(np.unique(raw_pts)) != frame_count:
            return None
        times = raw_times.astype(np.float64, copy=False)
        sharpness = raw_sharpness.astype(np.float32, copy=False)
        brightness = raw_brightness.astype(np.float32, copy=False)
        contrast = raw_contrast.astype(np.float32, copy=False)
        if not (np.isfinite(times).all() and np.isfinite(sharpness).all()
                and np.isfinite(brightness).all() and np.isfinite(contrast).all()):
            return None
        start, end = float(expected["start"]), float(expected["end"])
        if np.any(times < start) or (np.any(times > end) if include_end else np.any(times >= end)):
            return None
        if len(times) > 1 and (np.any(np.diff(times) <= 0.0) or np.any(np.diff(raw_pts) <= 0)):
            return None
        return [
            ScannedFrame(int(p), float(t), float(s), float(b), float(c), raw_thumbnails[i].copy())
            for i, (p, t, s, b, c) in enumerate(zip(raw_pts, times, sharpness, brightness, contrast))
        ]
    except Exception:
        # Cache corruption is a cache miss; source decoding will rebuild it.
        return None


def scan_video(
    reader: Any,
    cache_root: Path,
    fingerprint: str,
    *,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[ScannedFrame], dict[str, Any], str]:
    """Analyze all source frames sequentially and persist resumable chunks."""

    if not math.isfinite(float(chunk_seconds)) or chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive and finite")
    scan_key_payload = {
        "revision": SCAN_REVISION,
        "fingerprint": str(fingerprint),
        "max_side": SCAN_MAX_SIDE,
        "thumbnail_size": THUMBNAIL_SIZE,
        "chunk_seconds": float(chunk_seconds),
    }
    scan_key = _json_digest(scan_key_payload)
    root = Path(cache_root) / "scan" / scan_key
    duration = max(0.0, float(reader.info.duration))
    chunk_count = max(1, int(math.ceil(duration / chunk_seconds)))
    frames: list[ScannedFrame] = []
    cache_hits = 0
    cache_misses = 0
    started = time.perf_counter()
    for chunk_index in range(chunk_count):
        left = chunk_index * chunk_seconds
        right = min(duration, (chunk_index + 1) * chunk_seconds)
        include_end = chunk_index == chunk_count - 1
        chunk_key = _json_digest({
            **scan_key_payload,
            "chunk_index": chunk_index,
            "left": left,
            "right": right,
            "include_end": include_end,
        })
        data_path = root / f"chunk_{chunk_index:06d}.npz"
        expected = {"key": chunk_key, "start": left, "end": right, "include_end": include_end}
        cached = _load_chunk(data_path, expected)
        if cached is not None:
            frames.extend(cached)
            cache_hits += 1
            if progress:
                progress(chunk_index + 1, chunk_count)
            continue

        values: list[tuple[int, float, float, float, float, np.ndarray]] = []
        iterator = reader.iter_frames(left, right, max_side=SCAN_MAX_SIDE)
        for frame in iterator:
            frame_time = float(frame.time)
            # Some decoders include a small tolerance at both interval ends.
            # Enforce our own half-open partition so boundary frames belong to
            # exactly one chunk even when the decoder returns them twice.
            if frame_time < left or (frame_time > right if include_end else frame_time >= right):
                continue
            sharpness, brightness, contrast = _sharpness(frame.image)
            values.append((int(frame.pts), frame_time, sharpness, brightness, contrast, _gray_small(frame.image)))
        if values:
            _atomic_npz(
                data_path,
                metadata=_metadata_array({
                    "key": chunk_key,
                    "frame_count": len(values),
                    "start": left,
                    "end": right,
                    "include_end": include_end,
                }),
                pts=np.asarray([item[0] for item in values], dtype=np.int64),
                times=np.asarray([item[1] for item in values], dtype=np.float64),
                sharpness=np.asarray([item[2] for item in values], dtype=np.float32),
                brightness=np.asarray([item[3] for item in values], dtype=np.float32),
                contrast=np.asarray([item[4] for item in values], dtype=np.float32),
                thumbnails=np.stack([item[5] for item in values]).astype(np.uint8, copy=False),
            )
        else:
            _atomic_npz(
                data_path,
                metadata=_metadata_array({
                    "key": chunk_key,
                    "frame_count": 0,
                    "start": left,
                    "end": right,
                    "include_end": include_end,
                }),
                pts=np.empty((0,), dtype=np.int64),
                times=np.empty((0,), dtype=np.float64),
                sharpness=np.empty((0,), dtype=np.float32),
                brightness=np.empty((0,), dtype=np.float32),
                contrast=np.empty((0,), dtype=np.float32),
                thumbnails=np.empty((0, *THUMBNAIL_SIZE[::-1]), dtype=np.uint8),
            )
        frames.extend(ScannedFrame(*item) for item in values)
        cache_misses += 1
        if progress:
            progress(chunk_index + 1, chunk_count)

    frames.sort(key=lambda item: (item.time, item.pts))
    if not frames:
        raise RuntimeError("No decodable presentation frames were found")
    fps = max(0.0, float(reader.info.fps))
    expected_frames = int(round(duration * fps)) if fps > 0 else None
    stats = {
        "frames_decoded": len(frames),
        "duration": duration,
        "fps_metadata": fps,
        "expected_frames_estimate": expected_frames,
        "coverage_ratio_estimate": (len(frames) / expected_frames if expected_frames else None),
        "chunk_seconds": float(chunk_seconds),
        "chunks": chunk_count,
        "chunks_cache_hit": cache_hits,
        "chunks_rebuilt": cache_misses,
        "seconds": time.perf_counter() - started,
        "scan_revision": SCAN_REVISION,
    }
    return frames, stats, scan_key


def select_candidates(
    frames: Sequence[ScannedFrame],
    *,
    spacing_seconds: float = 1.25,
    video_path: str | os.PathLike[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select scene anchors and local visual-change peaks without a count cap.

    ``video_path`` remains accepted for callers from the earlier API, but scene
    boundaries come only from the already-decoded scan summaries. Reopening the
    source here would defeat scan checkpoint reuse.
    """

    if not frames:
        return [], {"decoded_frames": 0, "selected": 0}
    if not math.isfinite(float(spacing_seconds)) or spacing_seconds <= 0:
        raise ValueError("spacing_seconds must be positive and finite")
    ordered = sorted(frames, key=lambda item: (item.time, item.pts))
    deltas = np.zeros(len(ordered), dtype=np.float32)
    local_deltas = np.zeros(len(ordered), dtype=np.float32)
    for index in range(1, len(ordered)):
        left, right = ordered[index - 1].thumbnail, ordered[index].thumbnail
        diff = np.abs(left.astype(np.float32) - right.astype(np.float32)) / 255.0
        deltas[index] = float(np.mean(diff))
        local_deltas[index] = float(np.max(diff.reshape(3, 9, 4, 12).mean(axis=(1, 3))))

    nonzero = deltas[1:]
    if nonzero.size:
        median = float(np.median(nonzero))
        mad = float(np.median(np.abs(nonzero - median)))
        adaptive_cut = max(0.18, median + 8.0 * max(mad, 0.005))
    else:
        adaptive_cut = 1.0
    cut_indices = {index for index in range(1, len(ordered)) if deltas[index] >= adaptive_cut}

    # Scene boundaries partition the anchor windows so each new shot gets a
    # representative even when it begins just before a regular spacing mark.
    starts = [0, *sorted(cut_indices)]
    starts = sorted(set(starts))
    anchors: set[int] = set()
    for segment_index, segment_start in enumerate(starts):
        segment_end = starts[segment_index + 1] if segment_index + 1 < len(starts) else len(ordered)
        if segment_end <= segment_start:
            continue
        anchors.add(segment_start)
        segment_time = ordered[segment_start].time
        segment_stop = ordered[segment_end - 1].time
        bucket = 0
        cursor = segment_start
        while segment_time + bucket * spacing_seconds <= segment_stop + 1e-9:
            low = segment_time + bucket * spacing_seconds
            high = min(segment_stop + 1e-9, low + spacing_seconds)
            while cursor < segment_end and ordered[cursor].time < low:
                cursor += 1
            window_end = cursor
            while window_end < segment_end and ordered[window_end].time <= high:
                window_end += 1
            if window_end > cursor:
                anchors.add(max(range(cursor, window_end), key=lambda i: (ordered[i].sharpness, -deltas[i], -ordered[i].pts)))
            bucket += 1

    # Localized changes can occupy only a small part of the picture (for
    # example eyes or mouth).  A 4x3 thumbnail grid keeps those peaks visible.
    event_threshold = float(np.quantile(local_deltas[1:], 0.95)) if len(local_deltas) > 1 else 1.0
    event_threshold = max(0.045, event_threshold)
    events: set[int] = set()
    for index in range(1, len(ordered) - 1):
        if local_deltas[index] < event_threshold:
            continue
        if local_deltas[index] < local_deltas[index - 1] or local_deltas[index] < local_deltas[index + 1]:
            continue
        if ordered[index + 1].time - ordered[index - 1].time > 0.8:
            continue
        events.add(index)

    reasons: dict[int, set[str]] = {}
    for index in anchors:
        reason = "scene_boundary" if index in cut_indices else "coverage_anchor"
        reasons.setdefault(index, set()).add(reason)
    for index in events:
        reasons.setdefault(index, set()).add("local_visual_change")

    # The thumbnail is only a prefilter. It cannot prove that two full-size
    # frames are near duplicates, so retain every selected candidate here.
    selected: list[int] = []
    for index in sorted(reasons, key=lambda i: (ordered[i].time, ordered[i].pts)):
        selected.append(index)

    output = [
        {
            "pts": ordered[index].pts,
            "time": ordered[index].time,
            "reasons": sorted(reasons[index]),
            "scan_sharpness": ordered[index].sharpness,
            "brightness": ordered[index].brightness,
            "contrast": ordered[index].contrast,
            "local_change": float(local_deltas[index]),
            "scene_change": float(deltas[index]),
            "thumbnail": ordered[index].thumbnail,
        }
        for index in selected
    ]
    return output, {
        "decoded_frames": len(ordered),
        "scene_boundaries": len(cut_indices),
        "pyscenedetect_boundaries": 0,
        "scene_boundary_source": "already-decoded grayscale scan summaries",
        "coverage_anchor_frames": len(anchors),
        "local_change_frames": len(events),
        "selected": len(output),
        "selection_spacing_seconds": float(spacing_seconds),
        "quality_gate_applied": False,
        "selection_note": "relative sharpness, scene changes, and local visual changes; no universal brightness/contrast gate",
    }


def _load_identity_checkpoint(
    npz_path: Path,
    *,
    key: str,
    wanted_pts: Sequence[int],
    candidate_times: Mapping[int, float],
    start: float,
    end: float,
) -> tuple[list[dict[str, Any]], np.ndarray] | None:
    """Restore a complete, validated identity chunk or treat it as a miss."""

    if not npz_path.is_file():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            metadata = _metadata_from_array(data["metadata"])
            raw_features = np.asarray(data["features"])
        if metadata.get("key") != key or metadata.get("pts") != list(wanted_pts):
            return None
        records_value = metadata.get("records")
        if not isinstance(records_value, list):
            return None
        if raw_features.ndim != 2 or raw_features.dtype != np.dtype(np.float32):
            return None
        features = raw_features.astype(np.float32, copy=False)
        if len(features) and features.shape[1] == 0:
            return None
        if not np.isfinite(features).all():
            return None

        restored: list[dict[str, Any]] = []
        feature_indexes: list[int] = []
        wanted_set = set(wanted_pts)
        for value in records_value:
            if not isinstance(value, dict):
                return None
            record = dict(value)
            pts_value = record.get("pts")
            if isinstance(pts_value, bool) or not isinstance(pts_value, int) or pts_value not in wanted_set:
                return None
            time_value = record.get("time")
            if isinstance(time_value, bool) or not isinstance(time_value, (int, float)):
                return None
            record_time = float(time_value)
            expected_time = candidate_times.get(pts_value)
            if (not math.isfinite(record_time) or record_time < start - 1e-6 or record_time > end + 1e-6
                    or expected_time is None or not math.isclose(record_time, expected_time, rel_tol=0.0, abs_tol=1e-6)):
                return None
            bbox = record.get("bbox")
            if (not isinstance(bbox, list) or len(bbox) != 4
                    or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in bbox)):
                return None
            x0, y0, x1, y1 = map(float, bbox)
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                return None
            confidence = record.get("detector_confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
                return None
            kind = record.get("detector_kind")
            if not isinstance(kind, str) or not kind:
                return None
            feature_index = record.get("feature_index")
            if isinstance(feature_index, bool) or not isinstance(feature_index, int):
                return None
            if feature_index < 0 or feature_index >= len(features):
                return None
            feature_indexes.append(feature_index)
            restored.append(record)

        if len(features) != len(restored) or sorted(feature_indexes) != list(range(len(features))):
            return None
        for record in restored:
            feature_index = int(record.pop("feature_index"))
            record["feature"] = features[feature_index].copy()
        return restored, features
    except Exception:
        # Cache corruption is a cache miss; the identity chunk will be rebuilt.
        return None


def _atomic_identity_checkpoint(npz_path: Path, metadata: Mapping[str, Any], features: np.ndarray) -> None:
    # Metadata and feature vectors live in one atomically replaced file. A
    # process interruption cannot leave a new NPZ paired with stale JSON.
    _atomic_npz(npz_path, metadata=_metadata_array(metadata), features=np.asarray(features, dtype=np.float32))


def analyze_identity(
    reader: Any,
    model: Any,
    candidates: Sequence[Mapping[str, Any]],
    cache_root: Path,
    scan_key: str,
    model_revision: str,
    *,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    progress: Callable[[int, int], None] | None = None,
    character_boxes: Callable[[Sequence[Any], tuple[int, int], int | None], list[tuple[list[float], float, str]]],
    crop_image: Callable[[Image.Image, Sequence[float]], Image.Image],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Detect and embed selected PTS once, with model-revision checkpoints."""

    identity_key = _json_digest({
        "scan_key": scan_key,
        "model_revision": str(model_revision),
        "preprocess_revision": IDENTITY_PREPROCESS_REVISION,
        "checkpoint_revision": IDENTITY_CHECKPOINT_REVISION,
        "chunk_seconds": float(chunk_seconds),
    })
    root = Path(cache_root) / "identity" / identity_key
    chunk_count = max(1, int(math.ceil(float(reader.info.duration) / chunk_seconds)))
    by_chunk: dict[int, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        chunk_index = min(chunk_count - 1, max(0, int(float(candidate["time"]) // chunk_seconds)))
        by_chunk.setdefault(chunk_index, []).append(candidate)
    records: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    model_time = {"detect_seconds": 0.0, "embed_seconds": 0.0, "detect_calls": 0, "embed_calls": 0, "embedded_crops": 0}
    total_chunks = len(by_chunk)

    for progress_index, chunk_index in enumerate(sorted(by_chunk), 1):
        wanted_candidates = by_chunk[chunk_index]
        key = _json_digest({"identity_key": identity_key, "chunk_index": chunk_index, "pts": [int(c["pts"]) for c in wanted_candidates]})
        npz_path = root / f"chunk_{chunk_index:06d}.npz"
        wanted_pts_list = [int(item["pts"]) for item in wanted_candidates]
        candidate_times: dict[int, float] = {}
        for candidate in wanted_candidates:
            candidate_times[int(candidate["pts"])] = float(candidate["time"])
        left = chunk_index * chunk_seconds
        right = min(float(reader.info.duration), (chunk_index + 1) * chunk_seconds)
        loaded = _load_identity_checkpoint(
            npz_path,
            key=key,
            wanted_pts=wanted_pts_list,
            candidate_times=candidate_times,
            start=left,
            end=right,
        )
        if loaded is not None:
            restored_records, _vectors = loaded
            records.extend(restored_records)
            cache_hits += 1
            if progress:
                progress(progress_index, total_chunks)
            continue

        wanted_pts = {int(item["pts"]) for item in wanted_candidates}
        metadata_records: list[dict[str, Any]] = []
        crop_items: list[tuple[dict[str, Any], Image.Image]] = []
        for frame in reader.iter_frames(left, right, max_side=SCAN_MAX_SIDE):
            if int(frame.pts) not in wanted_pts:
                continue
            started = time.perf_counter()
            detections = model.detect(frame.image) or []
            model_time["detect_seconds"] += time.perf_counter() - started
            model_time["detect_calls"] += 1
            # ``None`` means every qualified detection is eligible. The batch
            # extraction path must not impose a per-frame identity cap here.
            boxes = character_boxes(detections, frame.image.size, None)
            for box, score, kind in boxes:
                crop = crop_image(frame.image, box)
                item = {
                    "pts": int(frame.pts),
                    "time": float(frame.time),
                    "bbox": [float(box[0]) / frame.image.width, float(box[1]) / frame.image.height,
                             float(box[2]) / frame.image.width, float(box[3]) / frame.image.height],
                    "detector_confidence": float(score),
                    "detector_kind": str(kind),
                    "feature_index": None,
                }
                metadata_records.append(item)
                crop_items.append((item, crop))

        vectors: list[np.ndarray] = []
        for offset in range(0, len(crop_items), 8):
            part = crop_items[offset:offset + 8]
            started = time.perf_counter()
            embedded = np.asarray(model.embed([crop for _, crop in part]), dtype=np.float32)
            model_time["embed_seconds"] += time.perf_counter() - started
            model_time["embed_calls"] += 1
            if embedded.ndim != 2 or embedded.shape[0] != len(part) or embedded.shape[1] == 0:
                raise RuntimeError("CCIP returned an invalid feature batch while building the batch cache")
            if not np.isfinite(embedded).all():
                raise RuntimeError("CCIP returned non-finite features while building the batch cache")
            for local_index, (item, _crop) in enumerate(part):
                item["feature_index"] = len(vectors)
                vectors.append(embedded[local_index].copy())
        features = np.stack(vectors).astype(np.float32, copy=False) if vectors else np.empty((0, 0), dtype=np.float32)
        model_time["embedded_crops"] += len(vectors)
        _atomic_identity_checkpoint(
            npz_path,
            {"key": key, "pts": wanted_pts_list, "records": metadata_records},
            features,
        )
        for item in metadata_records:
            record = dict(item)
            feature_index = record.pop("feature_index", None)
            record["feature"] = features[int(feature_index)].copy() if feature_index is not None else None
            records.append(record)
        cache_misses += 1
        if progress:
            progress(progress_index, total_chunks)

    records.sort(key=lambda item: (float(item["time"]), int(item["pts"]), str(item["detector_kind"]), item["bbox"]))
    model_time.update({"chunks_cache_hit": cache_hits, "chunks_rebuilt": cache_misses, "preprocess_revision": IDENTITY_PREPROCESS_REVISION})
    return records, model_time


__all__ = [
    "DEFAULT_CHUNK_SECONDS",
    "IDENTITY_PREPROCESS_REVISION",
    "SCAN_MAX_SIDE",
    "SCAN_REVISION",
    "ScannedFrame",
    "analyze_identity",
    "scan_video",
    "select_candidates",
]
