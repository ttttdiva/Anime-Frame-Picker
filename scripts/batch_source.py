#!/usr/bin/env python3
"""Headless, sequential processing of every MP4/MKV in the local source folder.

The default cute mode keeps character-present, visually distinct candidates.
Scene-only and identity-oriented modes remain explicit alternatives. No browser
or server is required.
"""

from __future__ import annotations

import argparse
import contextlib
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
import uuid
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anime_picker.batch_analysis import (
    DEFAULT_CHUNK_SECONDS,
    analyze_identity,
    scan_video,
    select_candidates,
)
from anime_picker import exclusion
from anime_picker.output_layout import default_state_root, publication_state_root, publish_image_set, register_public_receipts

VIDEO_EXTENSIONS = {".mp4", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_MODEL_ROOT = Path(r"D:\AI\models\Cold_F\image")
SELECTION_REVISION = "comprehensive-identities-topn-v2"
DEDUP_REVISION = "higher-resolution-rgb-pixel-confirmation-v5-exact-repeat-only"
OUTPUT_MANIFEST_REVISION = 2
CUTE_SELECTION_REVISION_V1 = "character-presence-cut-quality-visual-near-duplicate-v1"
CUTE_CURATION_REVISION_V1 = "character-presence-cut-quality-visual-near-duplicate-v1"
CUTE_SELECTION_REVISION = "character-presence-cut-quality-visual-near-duplicate-v2"
CUTE_CURATION_REVISION = "character-presence-cut-quality-global-visual-near-duplicate-v9"


@dataclass
class AutoSample:
    crop: Image.Image
    frame_index: int
    time: float
    pts: int
    box: list[float]
    score: float
    area: float
    kind: str


@dataclass
class ReferenceGroup:
    images: list[Image.Image]
    evidence: list[dict[str, Any]]
    mode: str
    support_frames: int | None = None


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex[:8] + ".tmp")
    try:
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _clean(value: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    return (name[:78].rstrip(" .") or "untitled")


def _publish_batch_images(
    folder: Path,
    marker: dict[str, Any],
    *,
    output_root: Path,
    layout_state_root: Path,
    video: Path,
    fingerprint: str,
    receipt_paths: Sequence[Path] = (),
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    """Expose verified run PNGs in the shared flat public images directory."""

    entries = marker.get("outputs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("completed batch marker has no output entries")
    sources: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("malformed batch output entry")
        relative = entry.get("file")
        if not isinstance(relative, str):
            raise ValueError("batch output entry is missing its internal file path")
        source_path = _safe_output_path(folder, relative)
        if source_path is None:
            raise ValueError(f"batch output path escapes internal state: {relative}")
        digest = str(entry.get("sha256", ""))
        if not digest or _file_digest(source_path) != digest:
            raise ValueError(f"internal batch PNG digest mismatch: {source_path}")
        pts = int(entry["pts"])
        sources.append({"source": source_path, "pts": pts, "sha256": digest})
    images, public_files, names_by_key, active = publish_image_set(
        output_root,
        video.name,
        fingerprint,
        sources,
        state_root=layout_state_root,
        run_id=str(marker.get("config_hash") or marker.get("batch_id") or fingerprint),
        receipt_paths=receipt_paths,
    )
    for entry in entries:
        key = (int(entry["pts"]), str(entry["sha256"]))
        name = names_by_key[key]
        entry["public_file"] = (images.relative_to(output_root.resolve()) / name).as_posix()
        entry["public_sha256"] = str(entry["sha256"])
    marker["images_directory"] = images.relative_to(output_root.resolve()).as_posix()
    marker["public_files"] = public_files
    marker["public_publication_id"] = active.get("publication_id")
    return images, public_files, active


def discover_videos(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Source folder not found: {root}")
    return sorted(
        (p.resolve() for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda p: p.name.casefold(),
    )


def _box_from_detection(raw: Any, size: tuple[int, int]) -> tuple[list[float], float, str] | None:
    if not isinstance(raw, Mapping):
        return None
    values = raw.get("box", raw.get("bbox"))
    if values is None:
        return None
    try:
        coords = [float(v) for v in values]
        score = float(raw.get("score", raw.get("confidence", 1.0)))
    except (TypeError, ValueError, OverflowError):
        return None
    if len(coords) != 4 or not all(math.isfinite(v) for v in coords) or not math.isfinite(score):
        return None
    width, height = size
    if max(abs(v) for v in coords) <= 1.5:
        coords = [coords[0] * width, coords[1] * height, coords[2] * width, coords[3] * height]
    x0, x1 = sorted((max(0.0, min(width, coords[0])), max(0.0, min(width, coords[2]))))
    y0, y1 = sorted((max(0.0, min(height, coords[1])), max(0.0, min(height, coords[3]))))
    area = ((x1 - x0) * (y1 - y0)) / max(1, width * height)
    if x1 - x0 < 24 or y1 - y0 < 24 or not 0.008 <= area <= 0.90:
        return None
    return [x0, y0, x1, y1], score, "head" if str(raw.get("kind", "person")).lower() == "head" else "person"


def _contains(big: Sequence[float], small: Sequence[float], tolerance: float = 4.0) -> bool:
    return (
        big[0] <= small[0] + tolerance and big[1] <= small[1] + tolerance
        and big[2] >= small[2] - tolerance and big[3] >= small[3] - tolerance
    )


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / max(1.0, area_a + area_b - overlap)


def character_boxes(detections: Sequence[Any], size: tuple[int, int], limit: int | None = None) -> list[tuple[list[float], float, str]]:
    """Select unambiguous crops, avoiding nested person/head duplicates.

    By default every non-overlapping detection is retained.  Callers that need
    a deliberately bounded preview may pass ``limit`` explicitly.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when supplied")
    parsed = [result for item in detections if (result := _box_from_detection(item, size)) is not None]
    persons = [item for item in parsed if item[2] == "person"]
    heads = [item for item in parsed if item[2] == "head" and not any(_contains(p[0], item[0]) for p in persons)]
    candidates = sorted(persons + heads, key=lambda item: (item[1], (item[0][2] - item[0][0]) * (item[0][3] - item[0][1])), reverse=True)
    chosen: list[tuple[list[float], float, str]] = []
    for item in candidates:
        if any(_iou(item[0], old[0]) > 0.42 for old in chosen):
            continue
        chosen.append(item)
        if limit is not None and len(chosen) >= limit:
            break
    return chosen


def _crop(image: Image.Image, box: Sequence[float]) -> Image.Image:
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    return image.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))).convert("RGB")


def _auto_interval(duration: float, count: int) -> tuple[float, float, float]:
    # Cover the full episode, but avoid the first and last ~5% where credits
    # and production cards dominate; never exclude more than 60 s at an end.
    margin = min(60.0, max(0.0, duration * 0.05))
    if duration <= 180:
        margin = 0.0
    start, end = margin, max(margin, duration - margin)
    return start, end, max(0.1, (end - start) / max(1, count - 1))


def auto_references(
    reader: Any,
    model: Any,
    *,
    max_groups: int,
    sample_count: int,
    max_crops_per_frame: int | None = None,
) -> list[ReferenceGroup]:
    """Heuristically choose recurring, isolated character crops from a video."""
    duration = float(reader.info.duration)
    if duration <= 0:
        raise ValueError("Cannot auto-discover characters without a video duration")
    start, end, step = _auto_interval(duration, sample_count)
    samples: list[AutoSample] = []
    seen_frames = 0
    print(f"  [auto] scanning {sample_count} representative frames over {start:.1f}-{end:.1f}s", flush=True)
    for frame_index, frame in enumerate(reader.sample(start, end, step, max_side=960)):
        seen_frames += 1
        image = frame.image.convert("RGB")
        boxes = character_boxes(model.detect(image) or [], image.size, max_crops_per_frame)
        for box, score, kind in boxes:
            area = (box[2] - box[0]) * (box[3] - box[1]) / (image.width * image.height)
            samples.append(AutoSample(_crop(image, box), frame_index, float(frame.time), int(frame.pts), box, score, area, kind))
    if not samples:
        raise RuntimeError(f"No isolated characters detected in {seen_frames} sampled frames")
    print(f"  [auto] found {len(samples)} possible character crops", flush=True)

    features: list[np.ndarray] = []
    for index in range(0, len(samples), 8):
        part = np.asarray(model.embed([sample.crop for sample in samples[index:index + 8]]), dtype=np.float32)
        if part.ndim != 2 or part.shape[0] != len(samples[index:index + 8]):
            raise RuntimeError("CCIP returned an invalid feature batch")
        features.append(part)
    embeddings = np.concatenate(features)
    distances = np.asarray(model.distance(embeddings, embeddings), dtype=np.float32)
    if distances.shape != (len(samples), len(samples)):
        raise RuntimeError(f"Invalid CCIP distance matrix shape: {distances.shape}")

    threshold = float(getattr(model, "threshold", 0.2132311898))
    # Each *distinct sampled frame* counts once. Head/person duplicates in the
    # same frame must not make a character appear more frequent.
    supports = [
        len({s.frame_index for j, s in enumerate(samples) if math.isfinite(float(distances[i, j])) and distances[i, j] <= threshold})
        for i in range(len(samples))
    ]
    order = sorted(
        range(len(samples)),
        key=lambda i: (supports[i], samples[i].score, samples[i].area, -i),
        reverse=True,
    )
    # Avoid selecting unrelated one-off background characters just to reach the
    # requested group count. If no one reappears at all, keep one tentative
    # representative and flag its one-frame support in metadata.
    eligible = [i for i in order if supports[i] >= 2]
    if not eligible:
        eligible = order[:1]
        print("  [auto:warning] no recurring identity found; one tentative reference only", flush=True)
    selected: list[int] = []
    groups: list[ReferenceGroup] = []
    for anchor in eligible:
        if any(float(distances[anchor, other]) <= threshold for other in selected):
            continue
        selected.append(anchor)
        neighbor_indices = sorted(
            (
                j for j in range(len(samples))
                if math.isfinite(float(distances[anchor, j])) and distances[anchor, j] <= threshold
            ),
            key=lambda j: (j != anchor, float(distances[anchor, j]), -samples[j].score),
        )
        unique_frames: set[int] = set()
        picked: list[int] = []
        for i in neighbor_indices:
            if samples[i].frame_index in unique_frames:
                continue
            unique_frames.add(samples[i].frame_index)
            picked.append(i)
            if len(picked) >= 3:
                break
        evidence = [
            {
                "source_time": samples[i].time,
                "source_pts": samples[i].pts,
                "sample_box": samples[i].box,
                "detector_kind": samples[i].kind,
                "detector_confidence": samples[i].score,
            }
            for i in picked
        ]
        groups.append(ReferenceGroup([samples[i].crop for i in picked], evidence, "automatic", supports[anchor]))
        if len(groups) >= max_groups:
            break
    print("  [auto] selected support per character: " + ", ".join(str(g.support_frames) for g in groups), flush=True)
    return groups


def reference_files(video: Path, reference_dir: Path, shared_reference: Path | None) -> list[Path]:
    if shared_reference is not None:
        if not shared_reference.is_file():
            raise FileNotFoundError(f"Shared reference not found: {shared_reference}")
        return [shared_reference]
    if not reference_dir.is_dir():
        return []
    stem = video.stem.casefold()
    return sorted(
        (
            path for path in reference_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and (path.stem.casefold() == stem or path.stem.casefold().startswith(stem + "__"))
        ),
        key=lambda p: p.name.casefold(),
    )


def _file_digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def manual_references(paths: Sequence[Path]) -> list[ReferenceGroup]:
    refs: list[Image.Image] = []
    evidence: list[dict[str, Any]] = []
    for path in paths:
        with Image.open(path) as image:
            refs.append(image.convert("RGB").copy())
        evidence.append({"path": str(path.resolve()), "sha256": _file_digest(path)})
    # The caller must supply one already-cropped character per image; grouping
    # images from the same video means they all represent one target character.
    return [ReferenceGroup(refs, evidence, "character")]


def _conditions(items: Sequence[str]) -> list[dict[str, str]]:
    result = []
    for item in items:
        scope, delimiter, name = item.partition(":")
        if delimiter != ":" or scope not in {"subject", "frame"} or not name.strip():
            raise ValueError(f"Expected subject:tag or frame:tag: {item!r}")
        result.append({"scope": scope, "name": name.strip()})
    return result


def _validate_conditions(model: Any, wanted: Sequence[Mapping[str, str]], avoided: Sequence[Mapping[str, str]]) -> None:
    if not wanted and not avoided:
        return
    vocabulary = model.vocabulary()
    for tag in [*wanted, *avoided]:
        if tag["name"] not in vocabulary or vocabulary[tag["name"]] != "general":
            raise ValueError(f"Unsupported general PixAI tag: {tag['name']!r} (scope={tag['scope']})")


def _progress(label: str):
    state = [-1, ""]
    def update(value: float, message: str) -> None:
        band = int(float(value) * 10)
        if band > state[0] or message in {"complete", "tag and rank"} and message != state[1]:
            print(f"    [{label}] {float(value):5.1%} {message}", flush=True)
            state[:] = [band, message]
    return update


def _export_character(
    reader: Any,
    candidates: Sequence[Mapping[str, Any]],
    folder: Path,
    *,
    logo_removal: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    folder.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, candidate in enumerate(candidates, 1):
        if candidate.get("pts") is None:
            raise ValueError("A candidate is missing its exact source PTS")
        name = f"frame_{index:04d}_pts_{int(candidate['pts'])}.png"
        destination = folder / name
        original_relative: str | None = None
        if logo_removal:
            from anime_picker.server import _process_logo_frame
            original = folder / "originals" / name
            original.parent.mkdir(parents=True, exist_ok=True)
            reader.export_png(int(candidate["pts"]), original)
            original_relative = original.relative_to(folder).as_posix()
            with Image.open(original) as image:
                processed, logo_metadata, frame_warnings = _process_logo_frame(
                    image.convert("RGB"), [], original_relative_path=original_relative
                )
            warnings.extend(f"{name}: {warning}" for warning in frame_warnings)
            if logo_metadata.get("applied"):
                processed.save(destination, format="PNG")
            else:
                shutil.copyfile(original, destination)
        else:
            reader.export_png(int(candidate["pts"]), destination)
            logo_metadata = {"requested": False, "applied": False, "status": "disabled"}
        exported.append(
            {
                "file": destination.relative_to(folder).as_posix(),
                "pts": int(candidate["pts"]),
                "time": float(candidate["time"]),
                "bbox": candidate.get("bbox"),
                "score": candidate.get("score"),
                "identity": candidate.get("identity"),
                "distance": candidate.get("distance"),
                "quality": candidate.get("quality"),
                "tags": candidate.get("tags"),
                "original_relative_path": original_relative,
                "logo_cleanup": logo_metadata,
            }
        )
    return exported, warnings


def _gpu_memory_snapshot() -> dict[str, Any]:
    """Collect process-local PyTorch CUDA counters when the runtime is available."""

    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        device = torch.cuda.current_device()
        return {
            "available": True,
            "device": int(device),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "scope": "PyTorch process allocator only; excludes ONNX Runtime and other processes",
        }
    except Exception as error:
        return {"available": None, "error": f"{type(error).__name__}: {error}"}


def _relative_quality(candidate: Mapping[str, Any], frames: Sequence[Any]) -> float:
    """Return a local sharpness rank without absolute brightness/contrast gates."""

    try:
        center = float(candidate.get("time", 0.0))
        value = float(candidate.get("scan_sharpness", 0.0))
        neighborhood = [
            float(frame.sharpness) for frame in frames
            if abs(float(frame.time) - center) <= 2.5
        ]
        if len(neighborhood) <= 1:
            return 0.5
        return sum(item <= value for item in neighborhood) / len(neighborhood)
    except (TypeError, ValueError, AttributeError):
        return 0.5


def _quality_ranker(frames: Sequence[Any]):
    frame_times = np.asarray([float(frame.time) for frame in frames], dtype=np.float64)
    sharpness = np.asarray([float(frame.sharpness) for frame in frames], dtype=np.float32)

    def rank(candidate: Mapping[str, Any]) -> float:
        try:
            center = float(candidate.get("time", 0.0))
            value = float(candidate.get("scan_sharpness", 0.0))
            left = int(np.searchsorted(frame_times, center - 2.5, side="left"))
            right = int(np.searchsorted(frame_times, center + 2.5, side="right"))
            local = sharpness[left:right]
            return float(np.count_nonzero(local <= value) / len(local)) if len(local) > 1 else 0.5
        except (TypeError, ValueError, OverflowError):
            return 0.5

    return rank


def _cute_candidate_for_curation(
    scan_candidate: Mapping[str, Any],
    technical_quality: float,
) -> dict[str, Any]:
    """Translate scan evidence without counting the sharpness percentile twice.

    Scan candidates have no independent aesthetic/source score, so curation gets
    a neutral source component. The local sharpness percentile is supplied once
    as technical quality; the public legacy ``score`` field is not fed back as a
    second source-quality signal.
    """

    item = dict(scan_candidate)
    rank = max(0.0, min(1.0, float(technical_quality)))
    item["quality"] = rank
    item["technical_quality"] = rank
    item["source_score"] = 0.5
    item.pop("score", None)
    return item


def _thumb_frame(candidate: Mapping[str, Any]) -> Image.Image | None:
    thumbnail = candidate.get("thumbnail")
    if thumbnail is None:
        return None
    try:
        gray = np.asarray(thumbnail, dtype=np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        return Image.fromarray(rgb, mode="RGB")
    except (TypeError, ValueError):
        return None


def _target_thumb(candidate: Mapping[str, Any]) -> Image.Image | None:
    image = _thumb_frame(candidate)
    bbox = candidate.get("bbox")
    if image is None or not isinstance(bbox, Sequence) or len(bbox) != 4:
        return image
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox]
        left = max(0, min(image.width - 1, int(round(x0 * image.width))))
        top = max(0, min(image.height - 1, int(round(y0 * image.height))))
        right = max(left + 1, min(image.width, int(round(x1 * image.width))))
        bottom = max(top + 1, min(image.height, int(round(y1 * image.height))))
        return image.crop((left, top, right, bottom))
    except (TypeError, ValueError, OverflowError):
        return image


def _best_at_pts_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best_at_pts: dict[int, dict[str, Any]] = {}
    for raw in candidates:
        candidate = dict(raw)
        pts = int(candidate.get("pts", -1))
        old = best_at_pts.get(pts)
        if old is None or float(candidate.get("score", 0.0)) > float(old.get("score", 0.0)):
            best_at_pts[pts] = candidate
    return sorted(best_at_pts.values(), key=lambda item: (float(item.get("time", 0.0)), int(item.get("pts", 0))))


def _is_protected_scan_event(candidate: Mapping[str, Any]) -> bool:
    protected_reasons = {"local_visual_change", "scene_boundary"}
    reasons = candidate.get("reasons") or ()
    if isinstance(reasons, str):
        reasons = (reasons,)
    try:
        return not protected_reasons.isdisjoint(reasons)
    except TypeError:
        return False


def _dedup_frame_pts(ordered: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return PTS that can participate in an unchanged high-resolution check."""

    def thumbnail_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
        try:
            a = np.asarray(left["thumbnail"], dtype=np.uint8)
            b = np.asarray(right["thumbnail"], dtype=np.uint8)
            if a.shape != b.shape or not a.size:
                return None
            return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))) / 255.0
        except (KeyError, TypeError, ValueError):
            return None

    needed: set[int] = set()
    for index, left in enumerate(ordered):
        left_time = float(left.get("time", 0.0))
        for later in range(index + 1, len(ordered)):
            right = ordered[later]
            if float(right.get("time", 0.0)) - left_time > 2.0:
                break
            delta = thumbnail_delta(left, right)
            if delta is None or delta <= 0.20:
                needed.add(int(left["pts"]))
                needed.add(int(right["pts"]))
    return [int(item["pts"]) for item in ordered if int(item["pts"]) in needed]


def _empty_dedup_stats() -> dict[str, Any]:
    return {
        "candidates_completed": 0,
        "candidates_total": 0,
        "pixel_comparisons": 0,
        "frame_lookups": 0,
        "reader_opens": 0,
        "decoded_frames": 0,
        "seek_fallbacks": 0,
        "decode_seconds": 0.0,
        "rgb_seconds": 0.0,
        "lookup_seconds": 0.0,
        "validation_seconds": 0.0,
        "prefilter_seconds": 0.0,
        "aborted_pass_seconds": 0.0,
        "dedup_pass_seconds": 0.0,
        "dedup_seconds": 0.0,
        "dedup_wall_seconds": 0.0,
        "wall_seconds": 0.0,
        "optimized_attempted": False,
        "optimized_used": False,
        "sequential_fast_path": False,
        "strict_fallback": False,
        "fallback_reason": None,
        "fallback_reasons": [],
        "max_cached_images": 0,
    }


class _OrderedDedupFrameLookup:
    """Exact reader adapter with a bounded, time-windowed image cache."""

    def __init__(
        self,
        reader: Any,
        ordered: Sequence[Mapping[str, Any]],
        requested_pts: Sequence[int],
        stats: dict[str, Any],
    ) -> None:
        self.reader = reader
        self.stats = stats
        requested_set = {int(value) for value in requested_pts}
        self.times = {
            int(item["pts"]): float(item.get("time", 0.0))
            for item in ordered
            if int(item["pts"]) in requested_set
        }
        self.cache: OrderedDict[int, Image.Image] = OrderedDict()
        self.enabled = False
        self._iterator: Any | None = None
        self._metrics_supported = False

        if not requested_pts:
            self.stats["fallback_reason"] = "no candidate pairs require image evidence"
            self.stats.setdefault("fallback_reasons", []).append(
                "no candidate pairs require image evidence"
            )
            return
        if not hasattr(reader, "validate_sequential_pts") or not hasattr(reader, "iter_selected_frames"):
            reason = "reader lacks ordered selected-PTS support"
            self.stats.update({"strict_fallback": True, "fallback_reason": reason})
            self.stats.setdefault("fallback_reasons", []).append(reason)
            return

        self._metrics_supported = True
        validation_before = float(stats.get("validation_seconds", 0.0))
        validation_started = time.perf_counter()
        try:
            safe, reason = reader.validate_sequential_pts(list(requested_pts), metrics=stats)
        except Exception as exc:
            safe = False
            reason = f"sequential PTS validation failed ({type(exc).__name__})"
        validation_elapsed = time.perf_counter() - validation_started
        validation_reported = float(stats.get("validation_seconds", 0.0)) - validation_before
        if validation_reported <= 0.0:
            stats["validation_seconds"] = float(stats.get("validation_seconds", 0.0)) + validation_elapsed
        if not safe:
            self.stats.update({"strict_fallback": True, "fallback_reason": reason})
            self.stats.setdefault("fallback_reasons", []).append(str(reason))
            return
        self.enabled = True
        self._metrics_supported = True
        self.stats.update({
            "optimized_attempted": True,
            "sequential_fast_path": True,
            "optimized_used": False,
            "fallback_reason": None,
        })
        self._iterator = reader.iter_selected_frames(list(requested_pts), max_side=960, metrics=stats)

    def __call__(self, pts: int) -> Image.Image:
        pts = int(pts)
        started = time.perf_counter()
        self.stats["frame_lookups"] += 1
        try:
            if self.enabled:
                now = self.times.get(pts, 0.0)
                for old_pts in list(self.cache):
                    if now - self.times.get(old_pts, now) > 2.0:
                        self.cache.pop(old_pts, None)
                if pts in self.cache:
                    return self.cache[pts]
                if self._iterator is None:
                    raise RuntimeError("ordered reader was not initialized")
                try:
                    frame = next(self._iterator)
                except StopIteration as exc:
                    from anime_picker.video import _UnsafeSequentialReadError

                    raise _UnsafeSequentialReadError("ordered reader ended before a requested PTS") from exc
                except Exception as exc:
                    from anime_picker.video import _UnsafeSequentialReadError

                    if isinstance(exc, _UnsafeSequentialReadError):
                        raise
                    raise _UnsafeSequentialReadError(
                        f"ordered reader failed ({type(exc).__name__})"
                    ) from exc
                if int(frame.pts) != pts:
                    from anime_picker.video import _UnsafeSequentialReadError

                    raise _UnsafeSequentialReadError(f"ordered reader returned PTS {frame.pts} while {pts} was requested")
                self.cache[pts] = frame.image
                self.stats["max_cached_images"] = max(self.stats["max_cached_images"], len(self.cache))
                return frame.image

            if self._metrics_supported:
                frame = self.reader.get_frame(pts, max_side=960, metrics=self.stats)
            else:
                frame = self.reader.get_frame(pts, max_side=960)
            return frame.image if hasattr(frame, "image") else frame
        finally:
            self.stats["lookup_seconds"] += time.perf_counter() - started

    def release(self, pts: int) -> None:
        self.cache.pop(int(pts), None)

    def close(self) -> None:
        if self._iterator is not None:
            close = getattr(self._iterator, "close", None)
            if close:
                close()
            self._iterator = None


def _clear_dedup_evidence(candidates: Sequence[Mapping[str, Any]]) -> None:
    """Remove all transient image/hash evidence from an aborted pass."""

    transient = {
        "_frame",
        "_crop",
        "_rgb96",
        "_crop_rgb64",
        "_frame_lookup_attempted",
        "_detection_kind",
        "phash",
        "dhash",
        "crop_phash",
        "crop_dhash",
    }
    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in transient:
                candidate.pop(key, None)


def _deduplicate_group_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    frame_lookup: Any | None = None,
    preserve_scan_events: bool = False,
    pixel_hash_distance: int = 5,
    rgb_similarity_threshold: float = 0.02,
    frame_release: Callable[[int], None] | None = None,
    preload_pts: set[int] | None = None,
    stats: dict[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Collapse only pixel duplicates confirmed from higher-resolution frames.

    The 48x27 scan thumbnail is used only to avoid loading frames whose visual
    change is plainly too large for a duplicate.  It cannot reject a candidate:
    without higher-resolution pixel evidence, candidates are retained.  In
    comprehensive all mode, scan-reasoned local-change and scene-boundary
    candidates can also be protected from duplicate collapse.
    """

    from anime_picker.scoring import are_pixel_duplicates, dhash, phash

    def thumbnail_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
        try:
            a = np.asarray(left["thumbnail"], dtype=np.uint8)
            b = np.asarray(right["thumbnail"], dtype=np.uint8)
            if a.shape != b.shape or not a.size:
                return None
            return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))) / 255.0
        except (KeyError, TypeError, ValueError):
            return None

    active_stats = stats if stats is not None else _empty_dedup_stats()
    ordered = _best_at_pts_candidates(candidates)
    total = len(ordered)
    active_stats["candidates_total"] = max(int(active_stats.get("candidates_total", 0)), total)
    wall_started = time.perf_counter()
    if progress:
        progress(0, total)

    def release_evidence(candidate: dict[str, Any]) -> None:
        candidate.pop("_frame", None)
        candidate.pop("_crop", None)
        candidate.pop("_rgb96", None)
        candidate.pop("_crop_rgb64", None)
        candidate.pop("_frame_lookup_attempted", None)
        if frame_release:
            frame_release(int(candidate.get("pts", -1)))

    def add_frame_evidence(candidate: dict[str, Any]) -> bool:
        if frame_lookup is None:
            return False
        try:
            frame = candidate.get("_frame")
            if frame is None:
                if candidate.get("_frame_lookup_attempted"):
                    return False
                frame = frame_lookup(int(candidate["pts"]))
            if frame is None:
                return False
            if hasattr(frame, "image"):
                frame = frame.image
            frame = frame.convert("RGB") if hasattr(frame, "convert") else Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")
            candidate["_frame"] = frame
            candidate["_detection_kind"] = candidate.get("detector_kind", "")
            candidate["phash"] = phash(frame)
            candidate["dhash"] = dhash(frame)
            bbox = candidate.get("bbox")
            if isinstance(bbox, Sequence) and len(bbox) == 4:
                x0, y0, x1, y1 = [float(value) for value in bbox]
                box = [x0 * frame.width, y0 * frame.height, x1 * frame.width, y1 * frame.height]
                crop = _crop(frame, box)
            else:
                crop = frame
            candidate["_crop"] = crop
            candidate["crop_phash"] = phash(crop)
            candidate["crop_dhash"] = dhash(crop)
            return True
        except (KeyError, TypeError, ValueError, OSError, OverflowError):
            return False

    def rgb_similarity_guard(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        """Require close RGB pixels as well as grayscale hash agreement."""

        try:
            started = time.perf_counter()
            resampling = getattr(Image, "Resampling", Image).BILINEAR

            def resized_rgb(candidate: Mapping[str, Any], image: Image.Image, size: tuple[int, int], key: str) -> np.ndarray:
                cached = candidate.get(key)
                if cached is None:
                    cached = np.asarray(image.convert("RGB").resize(size, resampling), dtype=np.float32)
                    if isinstance(candidate, dict):
                        candidate[key] = cached
                return cached

            full_left, full_right = left["_frame"], right["_frame"]
            full_mad = float(np.mean(np.abs(
                resized_rgb(left, full_left, (96, 96), "_rgb96")
                - resized_rgb(right, full_right, (96, 96), "_rgb96")
            ))) / 255.0
            if full_mad > rgb_similarity_threshold:
                return False
            crop_left, crop_right = left["_crop"], right["_crop"]
            crop_mad = float(np.mean(np.abs(
                resized_rgb(left, crop_left, (64, 64), "_crop_rgb64")
                - resized_rgb(right, crop_right, (64, 64), "_crop_rgb64")
            ))) / 255.0
            return crop_mad <= rgb_similarity_threshold
        except (KeyError, TypeError, ValueError, OSError):
            return False
        finally:
            active_stats["rgb_seconds"] += time.perf_counter() - started

    def is_protected_event(candidate: Mapping[str, Any]) -> bool:
        return preserve_scan_events and _is_protected_scan_event(candidate)
    kept: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    for completed, candidate in enumerate(ordered, 1):
        now = float(candidate.get("time", 0.0))
        expired = [item for item in active if now - float(item.get("time", 0.0)) > 2.0]
        for item in expired:
            release_evidence(item)
        expired_ids = {id(item) for item in expired}
        active = [item for item in active if id(item) not in expired_ids]

        # Scene starts and local-change peaks are the explicit events the
        # comprehensive selector found. Full-frame similarity can hide a
        # changed expression or small action inside a mostly static scene, so
        # retain those event candidates and keep them out of later duplicate
        # comparisons. Regular coverage anchors remain eligible for collapse.
        if is_protected_event(candidate):
            kept.append(candidate)
            active.append(candidate)
            active_stats["candidates_completed"] = completed
            if progress and (completed % 128 == 0 or completed == total):
                progress(completed, total)
            continue

        if frame_lookup is not None and preload_pts is not None and int(candidate["pts"]) in preload_pts:
            candidate["_frame_lookup_attempted"] = True
            try:
                frame = frame_lookup(int(candidate["pts"]))
                if frame is not None:
                    candidate["_frame"] = frame.image if hasattr(frame, "image") else frame
            except (KeyError, TypeError, ValueError, OSError, OverflowError):
                pass

        duplicate_existing: list[dict[str, Any]] = []
        may_compare = [
            item for item in active
            if not is_protected_event(item)
            and abs(now - float(item.get("time", 0.0))) <= 2.0
        ]
        needs_frame = any(
            (delta := thumbnail_delta(candidate, existing)) is None or delta <= 0.20
            for existing in may_compare
        )
        candidate_has_frame = add_frame_evidence(candidate) if needs_frame else False
        if candidate_has_frame:
            for existing in may_compare:
                delta = thumbnail_delta(candidate, existing)
                if delta is not None and delta > 0.20:
                    continue
                if ("_frame" not in existing or "_crop" not in existing or "phash" not in existing) and not add_frame_evidence(existing):
                    continue
                if rgb_similarity_guard(candidate, existing):
                    active_stats["pixel_comparisons"] = int(active_stats.get("pixel_comparisons", 0)) + 1
                    if are_pixel_duplicates(
                        candidate,
                        existing,
                        hash_distance=pixel_hash_distance,
                        time_window=2.0,
                    ):
                        duplicate_existing.append(existing)

        if duplicate_existing:
            strongest = max(duplicate_existing, key=lambda item: float(item.get("score", 0.0)))
            if float(candidate.get("score", 0.0)) <= float(strongest.get("score", 0.0)):
                release_evidence(candidate)
                active_stats["candidates_completed"] = completed
                if progress and (completed % 128 == 0 or completed == total):
                    progress(completed, total)
                continue
            for old in duplicate_existing:
                active = [item for item in active if item is not old]
                kept = [item for item in kept if item is not old]
                release_evidence(old)

        kept.append(candidate)
        active.append(candidate)
        active_stats["candidates_completed"] = completed
        if progress and (completed % 128 == 0 or completed == total):
            progress(completed, total)

    for candidate in kept:
        release_evidence(candidate)
    elapsed = time.perf_counter() - wall_started
    # Keep stage costs additive for diagnosis.  ``dedup_seconds`` is the
    # complete wall time for this pass; decode/RGB work is intentionally not
    # subtracted from it.
    active_stats["dedup_pass_seconds"] = float(active_stats.get("dedup_pass_seconds", 0.0)) + elapsed
    active_stats["dedup_seconds"] = float(active_stats.get("dedup_seconds", 0.0)) + elapsed
    active_stats["dedup_wall_seconds"] = float(active_stats.get("dedup_wall_seconds", 0.0)) + elapsed
    active_stats["wall_seconds"] = float(active_stats.get("wall_seconds", 0.0)) + elapsed
    return sorted(kept, key=lambda item: (float(item.get("time", 0.0)), int(item.get("pts", 0))))


def _deduplicate_with_reader(
    candidates: Sequence[Mapping[str, Any]],
    reader: Any,
    *,
    progress: Callable[[int, int], None] | None = None,
    preserve_scan_events: bool = False,
    pixel_hash_distance: int = 5,
    rgb_similarity_threshold: float = 0.02,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use a proven ordered reader when safe and retain exact lookup fallback."""

    from anime_picker.video import _UnsafeSequentialReadError

    stats = _empty_dedup_stats()
    wall_started = time.perf_counter()
    ordered = _best_at_pts_candidates(candidates)
    prefilter_started = time.perf_counter()
    prefilter_candidates = (
        [item for item in ordered if not _is_protected_scan_event(item)]
        if preserve_scan_events
        else ordered
    )
    requested_pts = _dedup_frame_pts(prefilter_candidates)
    stats["prefilter_seconds"] = time.perf_counter() - prefilter_started

    # Keep independent clean copies.  The optimized pass annotates candidates
    # with hashes and images while it yields; a later strict fallback must
    # reproduce legacy lazy exact-PTS behavior from untouched input.
    optimized_candidates = [dict(item) for item in ordered]
    fallback_candidates = [dict(item) for item in ordered]
    _clear_dedup_evidence(optimized_candidates)
    _clear_dedup_evidence(fallback_candidates)

    source = _OrderedDedupFrameLookup(reader, optimized_candidates, requested_pts, stats)
    stats["optimized_attempted"] = bool(source.enabled)
    try:
        if source.enabled:
            optimized_started = time.perf_counter()
            try:
                kept = _deduplicate_group_candidates(
                    optimized_candidates,
                    frame_lookup=source,
                    frame_release=source.release,
                    preload_pts=set(requested_pts),
                    stats=stats,
                    preserve_scan_events=preserve_scan_events,
                    pixel_hash_distance=pixel_hash_distance,
                    rgb_similarity_threshold=rgb_similarity_threshold,
                    progress=progress,
                )
                stats.update({
                    "optimized_used": True,
                    "sequential_fast_path": True,
                    "strict_fallback": False,
                    "fallback_reason": None,
                })
            except _UnsafeSequentialReadError as exc:
                aborted_elapsed = time.perf_counter() - optimized_started
                stats["aborted_pass_seconds"] = float(stats.get("aborted_pass_seconds", 0.0)) + aborted_elapsed
                stats["dedup_pass_seconds"] = float(stats.get("dedup_pass_seconds", 0.0)) + aborted_elapsed
                # The optimized pass may have yielded and annotated only a
                # prefix.  Drop every transient field and restart against the
                # untouched ordered candidates.  In particular, do not pass
                # preload_pts here: strict fallback must remain lazy.
                _clear_dedup_evidence(optimized_candidates)
                source.close()
                source.enabled = False
                source.cache.clear()
                reason = str(exc)
                stats.update({
                    "optimized_used": False,
                    "sequential_fast_path": False,
                    "strict_fallback": True,
                    "fallback_reason": reason,
                })
                stats.setdefault("fallback_reasons", []).append(reason)
                kept = _deduplicate_group_candidates(
                    fallback_candidates,
                    frame_lookup=source,
                    frame_release=source.release,
                    preload_pts=None,
                    stats=stats,
                    preserve_scan_events=preserve_scan_events,
                    pixel_hash_distance=pixel_hash_distance,
                    rgb_similarity_threshold=rgb_similarity_threshold,
                    progress=progress,
                )
        else:
            kept = _deduplicate_group_candidates(
                fallback_candidates,
                frame_lookup=source if requested_pts else None,
                frame_release=source.release if requested_pts else None,
                preload_pts=None,
                stats=stats,
                preserve_scan_events=preserve_scan_events,
                pixel_hash_distance=pixel_hash_distance,
                rgb_similarity_threshold=rgb_similarity_threshold,
                progress=progress,
            )
            stats["optimized_used"] = False
        return kept, stats
    finally:
        source.close()
        elapsed = time.perf_counter() - wall_started
        # This is the complete operation wall time, including prefilter,
        # validation, any aborted optimized pass, and strict fallback.  Keep
        # decode/RGB/lookup/validation as separate nested stage costs.
        stats["dedup_seconds"] = elapsed
        stats["dedup_wall_seconds"] = elapsed
        stats["wall_seconds"] = elapsed


def _identity_scores(model: Any, references: np.ndarray, records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare cached crop features in bounded chunks using the official CCIP API."""

    from anime_picker.scoring import aggregate_reference_distances

    valid = [(index, np.asarray(item["feature"], dtype=np.float32)) for index, item in enumerate(records) if item.get("feature") is not None]
    if not valid:
        return [], {"compared_crops": 0, "matched_crops": 0, "distance_seconds": 0.0, "distance_calls": 0}
    threshold = float(getattr(model, "threshold", 0.2132311898))
    output: list[dict[str, Any]] = []
    elapsed = 0.0
    calls = 0
    matches = 0
    for offset in range(0, len(valid), 48):
        block = valid[offset:offset + 48]
        vectors = np.stack([item[1] for item in block]).astype(np.float32, copy=False)
        started = time.perf_counter()
        distance_matrix = np.asarray(model.distance(references, vectors), dtype=np.float32)
        elapsed += time.perf_counter() - started
        calls += 1
        if distance_matrix.shape != (references.shape[0], len(block)):
            raise RuntimeError(f"CCIP distance returned unexpected shape {distance_matrix.shape}")
        for column, (record_index, _feature) in enumerate(block):
            match = aggregate_reference_distances(distance_matrix[:, column], threshold)
            if not match["accepted"]:
                continue
            record = dict(records[record_index])
            record.update({
                "distance": float(match["distance"]),
                "identity": float(match["identity"]),
                "identity_corroboration": float(match["corroboration"]),
            })
            output.append(record)
            matches += 1
    return output, {"compared_crops": len(valid), "matched_crops": matches, "distance_seconds": elapsed, "distance_calls": calls, "threshold": threshold}


def _automatic_reference_groups(model: Any, records: Sequence[Mapping[str, Any]], maximum: int) -> tuple[list[ReferenceGroup], dict[str, Any]]:
    """Rank recurring crop identities using blockwise official CCIP distances."""

    valid_records = [dict(item) for item in records if item.get("feature") is not None]
    if not valid_records:
        raise RuntimeError("No detected character crops were available for frequency selection")
    features = np.stack([np.asarray(item["feature"], dtype=np.float32) for item in valid_records])
    count = len(valid_records)
    threshold = float(getattr(model, "threshold", 0.2132311898))
    supports = np.zeros(count, dtype=np.int64)
    distance_seconds = 0.0
    distance_calls = 0
    frame_units_map: dict[int, list[int]] = {}
    for index, record in enumerate(valid_records):
        frame_units_map.setdefault(int(record["pts"]), []).append(index)
    frame_units = list(frame_units_map.values())
    block_size = 96  # frame groups; crop count varies with the detector output
    for left_start in range(0, len(frame_units), block_size):
        left_end = min(len(frame_units), left_start + block_size)
        left_units = frame_units[left_start:left_end]
        left_indices = [index for unit in left_units for index in unit]
        for right_start in range(left_start, len(frame_units), block_size):
            right_end = min(len(frame_units), right_start + block_size)
            right_units = frame_units[right_start:right_end]
            right_indices = [index for unit in right_units for index in unit]
            started = time.perf_counter()
            distances = np.asarray(model.distance(features[left_indices], features[right_indices]), dtype=np.float32)
            distance_seconds += time.perf_counter() - started
            distance_calls += 1
            if distances.shape != (len(left_indices), len(right_indices)):
                raise RuntimeError(f"CCIP returned invalid auto-selection distance block {distances.shape}")
            matched = np.isfinite(distances) & (distances <= threshold)
            left_offsets: list[tuple[int, int]] = []
            left_positions: dict[int, int] = {}
            offset = 0
            for unit in left_units:
                left_offsets.append((offset, offset + len(unit)))
                for local_index, record_index in enumerate(unit, offset):
                    left_positions[record_index] = local_index
                offset += len(unit)
            right_offsets: list[tuple[int, int]] = []
            right_positions: dict[int, int] = {}
            offset = 0
            for unit in right_units:
                right_offsets.append((offset, offset + len(unit)))
                for local_index, record_index in enumerate(unit, offset):
                    right_positions[record_index] = local_index
                offset += len(unit)
            for unit_index, unit in enumerate(left_units):
                for record_index in unit:
                    local_row = left_positions[record_index]
                    supports[record_index] += sum(
                        bool(np.any(matched[local_row, right_slice[0]:right_slice[1]]))
                        for right_slice in right_offsets
                    )
            if right_start != left_start:
                for unit_index, unit in enumerate(right_units):
                    for record_index in unit:
                        local_column = right_positions[record_index]
                        supports[record_index] += sum(
                            bool(np.any(matched[left_slice[0]:left_slice[1], local_column]))
                            for left_slice in left_offsets
                        )
    order = sorted(range(count), key=lambda index: (int(supports[index]), float(valid_records[index].get("detector_confidence", 0.0)), -float(valid_records[index].get("time", 0.0))), reverse=True)
    anchors: list[int] = []
    groups: list[ReferenceGroup] = []
    for anchor in order:
        if supports[anchor] < 2 and anchors:
            continue
        if anchors:
            started = time.perf_counter()
            already = np.stack([features[index] for index in anchors])
            separation = np.asarray(model.distance(already, features[anchor:anchor + 1]), dtype=np.float32).reshape(-1)
            distance_seconds += time.perf_counter() - started
            distance_calls += 1
            if np.any(separation <= threshold):
                continue
        neighbor_start = 0
        neighbors: list[int] = []
        while neighbor_start < count:
            neighbor_end = min(count, neighbor_start + block_size)
            started = time.perf_counter()
            distances = np.asarray(model.distance(features[anchor:anchor + 1], features[neighbor_start:neighbor_end]), dtype=np.float32).reshape(-1)
            distance_seconds += time.perf_counter() - started
            distance_calls += 1
            neighbors.extend(neighbor_start + int(i) for i, value in enumerate(distances) if math.isfinite(float(value)) and float(value) <= threshold)
            neighbor_start = neighbor_end
        neighbors.sort(key=lambda index: (-int(supports[index]), -float(valid_records[index].get("detector_confidence", 0.0)), float(valid_records[index].get("time", 0.0))))
        picked: list[int] = []
        seen_times: set[int] = set()
        for index in neighbors:
            record = valid_records[index]
            time_bucket = int(round(float(record.get("time", 0.0)) / 0.5))
            if time_bucket in seen_times:
                continue
            seen_times.add(time_bucket)
            picked.append(index)
            if len(picked) >= 3:
                break
        if not picked:
            picked = [anchor]
        images: list[Image.Image] = []
        evidence: list[dict[str, Any]] = []
        # Reference crops are read from analysis images in a later pass.  Store
        # normalized boxes/PTS here; the caller fills actual crops below.
        for index in picked:
            item = valid_records[index]
            evidence.append({
                "source_time": float(item["time"]),
                "source_pts": int(item["pts"]),
                "bbox": item["bbox"],
                "detector_kind": item["detector_kind"],
                "detector_confidence": float(item["detector_confidence"]),
            })
        groups.append(ReferenceGroup(images, evidence, "auto-topn", int(supports[anchor])))
        anchors.append(anchor)
        if len(groups) >= maximum:
            break
    if not groups:
        raise RuntimeError("No automatic frequency groups could be selected")
    stats = {"candidate_crops": count, "candidate_frames": len(frame_units), "identity_groups": len(groups), "support_frames": [group.support_frames for group in groups], "distance_seconds": distance_seconds, "distance_calls": distance_calls, "distance_block_size_frames": block_size, "threshold": threshold, "support_definition": "distinct PTS frames; multiple crops from one frame count once"}
    return groups, stats


def _atomic_save_image(image: Image.Image, destination: Path, *, format: str = "PNG") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp")
    try:
        image.save(temp, format=format)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def _valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _safe_output_path(folder: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    path = Path(relative)
    if path.is_absolute() or any(part in {"..", ""} for part in path.parts):
        return None
    try:
        root = folder.resolve()
        resolved = folder.joinpath(*path.parts).resolve(strict=False)
        if resolved == root or not resolved.is_relative_to(root):
            return None
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


def _matching_partial_export(
    folder: Path,
    group_relative: str,
    pts: int,
    fingerprint: str,
    config_hash: str,
) -> tuple[Path, Path, Mapping[str, Any]] | None:
    """Return a PTS record only when its per-group provenance matches this run."""

    try:
        group_dir = _safe_output_path(folder, group_relative)
        metadata_path = _safe_output_path(folder, f"{group_relative}/metadata.json")
        output_path = _safe_output_path(folder, f"{group_relative}/frame_pts_{pts}.png")
        if group_dir is None or metadata_path is None or output_path is None:
            return None
        if metadata_path.parent != group_dir or output_path.parent != group_dir:
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            return None
        if (
            metadata.get("source_fingerprint") != fingerprint
            or metadata.get("config_hash") != config_hash
            or metadata.get("output_manifest_revision") != OUTPUT_MANIFEST_REVISION
        ):
            return None
        exports = metadata.get("exports")
        if not isinstance(exports, list) or any(not isinstance(item, Mapping) for item in exports):
            return None
        name = f"frame_pts_{pts}.png"
        matches = [
            item for item in metadata.get("exports", [])
            if item.get("file") == name
            and int(item.get("pts", -1)) == pts
            and int(item.get("source_pts", -1)) == pts
        ]
        return (group_dir, output_path, dict(matches[0])) if len(matches) == 1 else None
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, json.JSONDecodeError):
        return None


def _verified_partial_png(path: Path, digest: Any) -> bool:
    return isinstance(digest, str) and bool(digest) and _valid_image(path) and _file_digest(path) == digest


def _verified_partial_original(group_dir: Path, export: Mapping[str, Any], pts: int) -> bool:
    expected_relative = f"originals/frame_pts_{pts}.png"
    if export.get("original_relative_path") != expected_relative:
        return False
    path = _safe_output_path(group_dir, expected_relative)
    return path is not None and _verified_partial_png(path, export.get("original_sha256"))


def _verified_completed_outputs(folder: Path, marker: Mapping[str, Any], fingerprint: str, config_hash: str) -> bool:
    """Validate PNG bytes and their exact-PTS metadata before resume skips."""

    if (
        marker.get("status") != "completed"
        or marker.get("fingerprint") != fingerprint
        or marker.get("config_hash") != config_hash
        or marker.get("output_manifest_revision") != OUTPUT_MANIFEST_REVISION
    ):
        return False
    saved_files = marker.get("saved_files")
    outputs = marker.get("outputs")
    if not isinstance(saved_files, list) or not saved_files or not isinstance(outputs, list) or len(outputs) != len(saved_files):
        return False
    if any(not isinstance(item, str) for item in saved_files) or len(set(saved_files)) != len(saved_files):
        return False
    if any(not isinstance(item, Mapping) for item in outputs):
        return False
    try:
        if set(saved_files) != {item.get("file") for item in outputs}:
            return False
        for output in outputs:
            relative = output.get("file")
            pts_value = output.get("pts")
            if type(pts_value) is not int:
                return False
            pts = pts_value
            image_path = _safe_output_path(folder, relative)
            metadata_path = _safe_output_path(folder, output.get("metadata"))
            if image_path is None or metadata_path is None:
                return False
            expected_name = f"frame_pts_{pts}.png"
            if image_path.name != expected_name or image_path.parent != metadata_path.parent:
                return False
            digest = str(output.get("sha256", ""))
            if not digest or not _valid_image(image_path) or _file_digest(image_path) != digest:
                return False
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, Mapping):
                return False
            if (
                metadata.get("source_fingerprint") != fingerprint
                or metadata.get("config_hash") != config_hash
                or metadata.get("output_manifest_revision") != OUTPUT_MANIFEST_REVISION
            ):
                return False
            metadata_exports = metadata.get("exports")
            if not isinstance(metadata_exports, list) or any(not isinstance(item, Mapping) for item in metadata_exports):
                return False
            matching_exports = [
                item for item in metadata_exports
                if item.get("file") == image_path.name
                and type(item.get("pts")) is int and item.get("pts") == pts
                and type(item.get("source_pts")) is int and item.get("source_pts") == pts
            ]
            if len(matching_exports) != 1:
                return False
            export = matching_exports[0]
            if (
                export.get("sha256") != digest
            ):
                return False
            original_relative = output.get("original_file")
            original_digest = output.get("original_sha256")
            if original_relative is None:
                if original_digest is not None or export.get("original_relative_path") is not None or export.get("original_sha256") is not None:
                    return False
            else:
                original_path = _safe_output_path(folder, original_relative)
                if (
                    original_path is None
                    or original_path.name != expected_name
                    or original_path.parent.name != "originals"
                    or original_path.parent.parent != image_path.parent
                    or not _verified_partial_png(original_path, original_digest)
                    or export.get("original_relative_path") != f"originals/{expected_name}"
                    or export.get("original_sha256") != original_digest
                ):
                    return False
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RuntimeError, json.JSONDecodeError):
        return False
    return True


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process every source/*.mp4 or *.mkv without a Web UI")
    parser.add_argument("--source", type=Path, default=ROOT / "source")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs", help="Public results root; each video gets <video>/images/*.png")
    parser.add_argument("--state-root", type=Path, default=None, help="Private run manifests/reports root (default: .local/anime-picker/batch)")
    parser.add_argument("--cache-root", type=Path, default=ROOT / ".cache" / "batch")
    parser.add_argument("--model-root", type=Path, default=Path(os.environ.get("ANIME_PICKER_MODEL_ROOT", DEFAULT_MODEL_ROOT)))
    parser.add_argument(
        "--model-cache-root",
        type=Path,
        default=None,
        help="Runtime Hugging Face/ONNX helper cache (defaults to <model-root>/cache)",
    )
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "source" / "refs")
    parser.add_argument("--reference", type=Path, help="A single tightly-cropped character image to use for every video")
    parser.add_argument(
        "--mode",
        choices=("cute", "all", "character", "auto-topn", "manual", "auto"),
        default="cute",
        help="cute character-present, near-duplicate-curated cuts (default), all scene candidates, explicit-reference matches, or opt-in frequency-ranked identity subset",
    )
    parser.add_argument(
        "--top-n", "--characters-per-video", dest="characters_per_video", type=int, default=2,
        help="Maximum frequency-ranked identity groups only for explicit auto-topn mode (default 2; cute mode has no top-N or character-count cap)",
    )
    parser.add_argument("--candidate-spacing", type=float, default=1.25, help="Coverage-anchor spacing in seconds; local visual-change peaks are added independently")
    parser.add_argument(
        "--cut-quality-min",
        type=float,
        default=0.55,
        help="Minimum 0..1 cute-cut score for automatic character mode (default 0.55; no result-count cap)",
    )
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS, help="Resumable scan/checkpoint chunk duration")
    parser.add_argument("--decode-threads", type=int, default=1, help="Maximum decoder threads per video (default 1)")
    parser.add_argument("--want", action="append", default=[], metavar="subject:smile")
    parser.add_argument("--avoid", action="append", default=[], metavar="frame:multiple_people")
    parser.add_argument("--no-logo-removal", action="store_true", help="Export source-exact native PNGs only")
    parser.add_argument("--force", action="store_true", help="Generate a new output even for previously completed files")
    parser.add_argument("--no-archive", action="store_true", help="Keep completed source videos/subtitles in source/")
    parser.add_argument("--dry-run", action="store_true", help="List planned inputs without loading GPU models or decoding footage")
    opts = parser.parse_args(argv)
    if opts.characters_per_video < 1:
        parser.error("--top-n must be at least 1")
    if opts.decode_threads < 1:
        parser.error("--decode-threads must be at least 1")
    # Keep the old spellings accepted for existing scripts while reports and
    # metadata use the unambiguous canonical names.
    opts.mode = {"manual": "character", "auto": "auto-topn"}.get(opts.mode, opts.mode)
    if not math.isfinite(opts.candidate_spacing) or opts.candidate_spacing <= 0 or not math.isfinite(opts.chunk_seconds) or opts.chunk_seconds <= 0:
        parser.error("--candidate-spacing and --chunk-seconds must be positive and finite")
    if not math.isfinite(opts.cut_quality_min) or not 0 <= opts.cut_quality_min <= 1:
        parser.error("--cut-quality-min must be finite and between 0 and 1")
    return opts


def _read_or_compute_tags(
    model: Any,
    image: Image.Image,
    cache_root: Path,
    tag_revision: str,
) -> tuple[dict[str, dict[str, float]], bool]:
    pixels = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    path = cache_root / "tags" / hashlib.sha256(tag_revision.encode("utf-8")).hexdigest()[:16] / f"{digest}.json"
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value, True
        except (OSError, json.JSONDecodeError):
            pass
    value = model.tag(image)
    _write_json(path, value)
    return value, False


def _tag_curated_candidates(
    model: Any,
    records: Sequence[dict[str, Any]],
    cache_root: Path,
    tag_revision: str,
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Attach cached PixAI scores, batching only images absent from cache.

    This path intentionally requires ``tag_many``. It never falls back to a
    CPU or one-at-a-time model call if the required GPU batch inference fails.
    """

    if not records:
        return {"tagged": 0, "cache_hits": 0, "batches": 0, "revision": tag_revision}
    tag_many = getattr(model, "tag_many", None)
    if not callable(tag_many):
        raise RuntimeError("cute mode requires GPU-backed ModelService.tag_many; no fallback is available")
    revision_dir = hashlib.sha256(tag_revision.encode("utf-8")).hexdigest()[:16]
    pending: list[tuple[dict[str, Any], Path]] = []
    cache_hits = 0
    for record in records:
        path = Path(str(record["image"]))
        with Image.open(path) as source_image:
            pixels = np.ascontiguousarray(np.asarray(source_image.convert("RGB"), dtype=np.uint8))
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        tag_path = cache_root / "tags" / revision_dir / f"{digest}.json"
        if tag_path.is_file():
            try:
                cached = json.loads(tag_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict):
                    record["tags"] = cached
                    cache_hits += 1
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        pending.append((record, tag_path))

    batches = 0
    for offset in range(0, len(pending), batch_size):
        part = pending[offset:offset + batch_size]
        images: list[Image.Image] = []
        try:
            for record, _tag_path in part:
                with Image.open(str(record["image"])) as source_image:
                    images.append(source_image.convert("RGB").copy())
            # Let GPURequiredError / model errors propagate. A CPU retry would
            # silently change both performance and the stated execution policy.
            results = tag_many(images, batch_size=batch_size)
            if not isinstance(results, Sequence) or isinstance(results, (str, bytes)) or len(results) != len(part):
                raise RuntimeError(
                    f"PixAI batch returned {len(results) if hasattr(results, '__len__') else 'invalid'} rows "
                    f"for {len(part)} candidate images"
                )
            for (record, tag_path), tags in zip(part, results):
                if not isinstance(tags, Mapping):
                    raise RuntimeError("PixAI batch returned a non-mapping tag row")
                normalized = {
                    str(category): {
                        str(name): float(score)
                        for name, score in values.items()
                    }
                    for category, values in tags.items()
                    if isinstance(values, Mapping)
                }
                record["tags"] = normalized
                _write_json(tag_path, normalized)
            batches += 1
            print(
                f"  [tag] candidates {min(offset + len(part), len(pending))}/{len(pending)} "
                f"(cache hits {cache_hits}/{len(records)})",
                flush=True,
            )
        finally:
            for image in images:
                image.close()
    return {
        "tagged": len(records),
        "cache_hits": cache_hits,
        "cache_misses": len(pending),
        "batches": batches,
        "batch_size": batch_size,
        "revision": tag_revision,
    }


def _curate_character_candidates(
    model: Any,
    reader: Any,
    candidates: Sequence[Mapping[str, Any]],
    *,
    folder: Path,
    cache_root: Path,
    source_fingerprint: str,
    tagger_revision: str,
    diagnostics: Mapping[str, Any],
    minimum_cut_quality: float = 0.55,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Tag scan candidates and curate them before any public export."""

    from anime_picker.curation import curate_candidates

    if not bool(getattr(model, "require_gpu", False)):
        raise RuntimeError("cute mode requires ModelService(require_gpu=True)")
    input_root = _safe_output_path(folder, "curation-inputs")
    if input_root is None:
        raise RuntimeError("Curation input path escapes its private run folder")
    input_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    candidates_by_id: dict[str, dict[str, Any]] = {}
    curation_input_cache_hits = 0
    curation_input_decodes = 0
    source_hash = str(source_fingerprint).rsplit(":", 1)[-1][:12]
    prior_run_folders = sorted(
        (
            path
            for path in folder.parent.glob(f"*_{source_hash}_*")
            if path.is_dir() and path.resolve() != folder.resolve()
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for source_candidate in candidates:
        candidate = dict(source_candidate)
        pts = int(candidate["pts"])
        record_id = f"pts:{pts}"
        image_path = _safe_output_path(folder, f"curation-inputs/frame_pts_{pts}.png")
        if image_path is None:
            raise RuntimeError(f"Curation image path escapes its private run folder for PTS {pts}")
        if not _valid_image(image_path):
            for previous_run in prior_run_folders:
                previous_image = previous_run / "curation-inputs" / f"frame_pts_{pts}.png"
                if not _valid_image(previous_image):
                    continue
                try:
                    os.link(previous_image, image_path)
                except OSError:
                    if not image_path.exists():
                        shutil.copy2(previous_image, image_path)
                if _valid_image(image_path):
                    curation_input_cache_hits += 1
                    break
            if not _valid_image(image_path):
                frame = reader.get_frame(pts, max_side=960)
                image = frame.image if hasattr(frame, "image") else frame
                _atomic_save_image(image, image_path)
                curation_input_decodes += 1
        record = {
            "id": record_id,
            "pts": pts,
            "time": float(candidate["time"]),
            "image": str(image_path),
            "tags": {},
            "quality": float(candidate.get("quality", candidate.get("score", 0.0))),
            "technical_quality": float(candidate.get("technical_quality", candidate.get("quality", 0.0))),
            "source_score": float(candidate.get("source_score", 0.5)),
            "reasons": (
                [candidate.get("reasons")]
                if isinstance(candidate.get("reasons"), str)
                else list(candidate.get("reasons") or ())
            ),
        }
        candidates_by_id[record_id] = candidate
        records.append(record)

    detector_model_revision = str(
        getattr(model, "identity_revision", None)
        or getattr(model, "manifest_revision", None)
        or getattr(model, "revision", "unknown")
    )
    from anime_picker.batch_analysis import IDENTITY_PREPROCESS_REVISION
    from anime_picker.models import HEAD_REVISION, PERSON_REVISION

    detector_recipe = {
        "model_revision": detector_model_revision,
        "person_detector_revision": PERSON_REVISION,
        "head_detector_revision": HEAD_REVISION,
        "preprocess_revision": IDENTITY_PREPROCESS_REVISION,
        "candidate_max_side": 960,
    }
    detector_revision = hashlib.sha256(
        json.dumps(detector_recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    detector_cache_root = cache_root / "detections" / detector_revision[:16]
    detector_cache_root.mkdir(parents=True, exist_ok=True)
    detector_cache_hits = 0
    detector_cache_misses = 0
    detector_calls = 0
    detector_seconds = 0.0
    prepare_runtime = getattr(model, "_prepare_runtime", None)
    if callable(prepare_runtime):
        prepare_runtime()
        if os.environ.get("ONNX_MODE") != "gpu":
            raise RuntimeError("cute mode requires ONNX_MODE=gpu before person/head detection")
    for index, record in enumerate(records, 1):
        pts = int(record["pts"])
        image_path = Path(str(record["image"]))
        image_digest = _file_digest(image_path)
        cache_path = detector_cache_root / f"pts_{pts}.json"
        detections: list[dict[str, Any]] | None = None
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError, TypeError):
                cached = None
            if (
                isinstance(cached, Mapping)
                and cached.get("source_fingerprint") == source_fingerprint
                and cached.get("detector_revision") == detector_revision
                and type(cached.get("pts")) is int
                and cached.get("pts") == pts
                and cached.get("image_sha256") == image_digest
                and isinstance(cached.get("detections"), list)
                and all(isinstance(item, Mapping) for item in cached["detections"])
            ):
                detections = [dict(item) for item in cached["detections"]]
                detector_cache_hits += 1
        if detections is None:
            detect = getattr(model, "detect", None)
            if not callable(detect):
                raise RuntimeError("cute mode requires the GPU-backed person/head detector")
            with Image.open(image_path) as opened:
                detect_started = time.perf_counter()
                detections = detect(opened.convert("RGB"))
                detector_seconds += time.perf_counter() - detect_started
            if not isinstance(detections, list) or any(not isinstance(item, Mapping) for item in detections):
                raise RuntimeError(f"person/head detector returned malformed evidence for PTS {pts}")
            detections = [dict(item) for item in detections]
            _write_json(
                cache_path,
                {
                    "source_fingerprint": source_fingerprint,
                    "detector_revision": detector_revision,
                    "pts": pts,
                    "image_sha256": image_digest,
                    "detections": detections,
                },
            )
            detector_cache_misses += 1
            detector_calls += 1
        record["detections"] = detections
        if index == len(records) or index % 128 == 0:
            print(
                f"  [detect] candidates {index}/{len(records)} "
                f"(cache hits {detector_cache_hits}/{index})",
                flush=True,
            )

    tagging_started = time.perf_counter()
    tag_stats = _tag_curated_candidates(model, records, cache_root, tagger_revision, batch_size=8)
    tagging_seconds = time.perf_counter() - tagging_started
    curation_started = time.perf_counter()

    def curation_progress(done: int, total: int) -> None:
        if done == 0 or done == total or done % 128 == 0:
            print(f"  [curation] candidates {done}/{total}", flush=True)

    report = curate_candidates(
        records,
        presence_threshold=0.17,
        time_window=4.0,
        minimum_cut_quality=minimum_cut_quality,
        include_full_tags=False,
        progress=curation_progress,
    )
    curation_seconds = time.perf_counter() - curation_started
    for key in ("kept", "excluded", "groups"):
        if key not in report:
            raise RuntimeError(f"curate_candidates result is missing {key!r}")
    summary = report.get("summary", report.get("counts", {}))
    if not isinstance(summary, Mapping):
        raise RuntimeError("curate_candidates summary/counts must be a mapping")
    kept: list[dict[str, Any]] = []
    for result in report["kept"]:
        if not isinstance(result, Mapping):
            raise RuntimeError("curate_candidates returned a malformed kept record")
        record_id = str(result.get("id", ""))
        candidate = candidates_by_id.get(record_id)
        if candidate is None:
            raise RuntimeError(f"curate_candidates returned unknown candidate {record_id!r}")
        merged = dict(candidate)
        for key in ("character_presence", "presence_score", "presence_evidence", "cut_quality", "quality_components", "visual_metrics", "group_id", "tags", "detections"):
            if key in result:
                merged[key] = result[key]
        kept.append(merged)
    backend = {
        "framework": "Transformers/PyTorch PixAI tagger",
        "require_gpu": bool(getattr(model, "require_gpu", False)),
        "requested_device": 0 if bool(getattr(model, "require_gpu", False)) else -1,
        "pipeline_device": None,
        "execution": "GPU-backed tag_many inference" if int(tag_stats.get("cache_misses", 0)) else "cached tags only; no model inference",
        "precision": diagnostics.get("tag_precision", getattr(model, "_tag_precision", None)),
        "runtime_cuda_available": diagnostics.get("torch_cuda_available"),
        "device_evidence": (
            "pipeline-reported" if getattr(getattr(model, "_tagger", None), "device", None) is not None
            else "no model inference occurred" if not int(tag_stats.get("cache_misses", 0))
            else "GPU-required pipeline configuration; runtime device not exposed"
        ),
    }
    tagger = getattr(model, "_tagger", None)
    device = getattr(tagger, "device", None)
    if device is not None:
        backend["pipeline_device"] = str(device)
    audit = {
        "revision": CUTE_CURATION_REVISION,
        "tagger_revision": tagger_revision,
        "tagger_backend": backend,
        "detector_used": True,
        "detector_backend": {
            "models": {"person": PERSON_REVISION, "head": HEAD_REVISION},
            "model_revision": detector_model_revision,
            "execution": "GPU-required ONNX person/head detection",
            "require_gpu": bool(getattr(model, "require_gpu", False)),
            "onnx_mode": os.environ.get("ONNX_MODE"),
            "cache_revision": detector_revision,
            "cache_directory": str(detector_cache_root),
        },
        "detection": {
            "candidates": len(records),
            "calls": detector_calls,
            "seconds": detector_seconds,
            "cache_hits": detector_cache_hits,
            "cache_misses": detector_cache_misses,
        },
        "curation_input_cache_hits": curation_input_cache_hits,
        "curation_input_decodes": curation_input_decodes,
        "identity_model_used": False,
        "ccip_used": False,
        "identity_id_emitted": False,
        "tagging": {**tag_stats, "seconds": tagging_seconds},
        "detection_seconds": detector_seconds,
        "curation_seconds": curation_seconds,
        "report": report,
    }
    return kept, audit, {
        "tagged": int(tag_stats.get("tagged", 0)),
        "tag_cache_hits": int(tag_stats.get("cache_hits", 0)),
        "tag_batches": int(tag_stats.get("batches", 0)),
        "tagged_candidates": len(records),
        "detector_used": True,
        "character_candidates": len(kept),
        "excluded_candidates": len(report.get("excluded", [])),
        "near_duplicate_groups": int(summary.get("near_duplicate_groups", summary.get("duplicate_groups", 0))),
        "detector_seconds": detector_seconds,
        "detector_calls": detector_calls,
        "detector_cache_hits": detector_cache_hits,
        "detector_cache_misses": detector_cache_misses,
        "curation_input_cache_hits": curation_input_cache_hits,
        "curation_input_decodes": curation_input_decodes,
        "tag_seconds": tagging_seconds,
        "curation_seconds": curation_seconds,
    }


def _filter_tags(
    model: Any,
    reader: Any,
    groups: Sequence[dict[str, Any]],
    wanted: Sequence[Mapping[str, str]],
    avoided: Sequence[Mapping[str, str]],
    cache_root: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    from anime_picker.scoring import compact_tag_evidence, evaluate_tags, subject_tag_ambiguity

    if not wanted and not avoided:
        return {"tagged": 0, "tag_seconds": 0.0, "cache_hits": 0}
    wants_subject = any(item["scope"] != "frame" for item in [*wanted, *avoided])
    wants_frame = any(item["scope"] == "frame" for item in [*wanted, *avoided])
    tagged = 0
    elapsed = 0.0
    cache_hits = 0
    tag_revision = str(getattr(model, "tag_revision", model.revision))
    normalized_wanted = [{"scope": item["scope"], "tag": item["name"]} for item in wanted]
    normalized_avoided = [{"scope": item["scope"], "tag": item["name"]} for item in avoided]
    for group in groups:
        kept = []
        for candidate in group["candidates"]:
            frame = reader.get_frame(int(candidate["pts"]), max_side=960).image
            subject_tags: Mapping[str, Any] = {}
            frame_tags: Mapping[str, Any] = {}
            started = time.perf_counter()
            if wants_subject and mode != "all":
                bbox = candidate.get("bbox")
                if not isinstance(bbox, Sequence) or len(bbox) != 4:
                    continue
                box = [float(bbox[0]) * frame.width, float(bbox[1]) * frame.height, float(bbox[2]) * frame.width, float(bbox[3]) * frame.height]
                crop = _crop(frame, box)
                subject_tags, cache_hit = _read_or_compute_tags(model, crop, cache_root, tag_revision)
                cache_hits += int(cache_hit)
                tagged += 1
            if wants_frame:
                frame_tags, cache_hit = _read_or_compute_tags(model, frame, cache_root, tag_revision)
                cache_hits += int(cache_hit)
                tagged += 1
            elapsed += time.perf_counter() - started
            if wants_subject and subject_tag_ambiguity(subject_tags, 0.8):
                candidate["tag_reasons"] = ["subject_ambiguous_multiple"]
                continue
            result = evaluate_tags(subject_tags, frame_tags, normalized_wanted, normalized_avoided)
            candidate["tags"] = {
                "subject": compact_tag_evidence(subject_tags, normalized_wanted + normalized_avoided) if wants_subject else {},
                "frame": compact_tag_evidence(frame_tags, normalized_wanted + normalized_avoided) if wants_frame else {},
            }
            candidate["tag_score"] = float(result["score"])
            candidate["tag_reasons"] = list(result["reasons"])
            if result["passed"]:
                candidate["score"] = float(candidate.get("score", 0.0)) * float(result["score"])
                kept.append(candidate)
        group["candidates"] = kept
    return {"tagged": tagged, "tag_seconds": elapsed, "cache_hits": cache_hits, "revision": tag_revision}


def _run_jobs(args: argparse.Namespace, videos: Sequence[Path], references_by_video: Sequence[tuple[Path, list[Path]]], wanted: Sequence[Mapping[str, str]], avoided: Sequence[Mapping[str, str]], model: Any | None, revision: str, diagnostics: Mapping[str, Any], initial_gpu: Mapping[str, Any]) -> tuple[dict[str, Any], Path, bool]:
    # Whatever ends the loop (error, interrupt), no video lease outlives it.
    lease = contextlib.ExitStack()
    try:
        return _run_jobs_inner(args, videos, references_by_video, wanted, avoided, model, revision, diagnostics, initial_gpu, lease)
    finally:
        lease.close()


def _run_jobs_inner(args: argparse.Namespace, videos: Sequence[Path], references_by_video: Sequence[tuple[Path, list[Path]]], wanted: Sequence[Mapping[str, str]], avoided: Sequence[Mapping[str, str]], model: Any | None, revision: str, diagnostics: Mapping[str, Any], initial_gpu: Mapping[str, Any], lease: contextlib.ExitStack) -> tuple[dict[str, Any], Path, bool]:
    from anime_picker.batch_analysis import IDENTITY_PREPROCESS_REVISION, SCAN_REVISION
    from anime_picker.video import VideoReader

    source = args.source.resolve()
    output_root = args.output_root.resolve()
    state_root = (args.state_root or (default_state_root(output_root, ROOT) / "batch")).resolve()
    if state_root == output_root or state_root.is_relative_to(output_root) or output_root.is_relative_to(state_root):
        raise ValueError("--state-root and --output-root must be separate, non-nested directories")
    cache_root = args.cache_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    if os.stat(output_root).st_dev != os.stat(state_root).st_dev:
        raise ValueError("--output-root and --state-root must be on the same volume for atomic publication")
    cache_root.mkdir(parents=True, exist_ok=True)
    mode = args.mode
    tag_revision = str(getattr(model, "tag_revision", revision)) if model is not None else None
    params: dict[str, Any] = {
        "mode": mode,
        "selection_semantics": {
            "cute": "character presence, cut-quality ranking, and visual near-duplicate grouping; no identity grouping or top-n cap",
            "all": "scene, coverage, and local-change candidates across the video",
            "character": "matches against explicitly supplied reference crops",
            "auto-topn": "frequency-ranked identity groups limited to the requested top-n; not comprehensive identity coverage",
        }[mode],
        "candidate_spacing": args.candidate_spacing,
        "chunk_seconds": args.chunk_seconds,
        "top_n": args.characters_per_video if mode == "auto-topn" else None,
        "wanted": list(wanted),
        "avoided": list(avoided),
        "logo_removal": not args.no_logo_removal,
        "scan_revision": SCAN_REVISION,
        "identity_preprocess_revision": IDENTITY_PREPROCESS_REVISION if mode in {"character", "auto-topn"} else None,
        "model_revision": revision,
        "tagger_revision": tag_revision if mode == "cute" else None,
        "selection_revision": CUTE_SELECTION_REVISION if mode == "cute" else SELECTION_REVISION,
        "curation_revision": CUTE_CURATION_REVISION if mode == "cute" else None,
        "dedup_revision": CUTE_CURATION_REVISION if mode == "cute" else DEDUP_REVISION,
        "output_manifest_revision": OUTPUT_MANIFEST_REVISION,
    }
    if mode == "cute":
        # The cut-quality floor changes which curated cuts are exported, so a
        # completed run for another floor must never be reused as complete.
        # Only cute consumes it; other modes keep their existing hashes.
        params["cut_quality_min"] = float(args.cut_quality_min)
    outcomes: list[dict[str, Any]] = []
    batch_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    report_path = state_root / f"batch_run_{batch_id}.json"
    batch_telemetry: dict[str, Any] = {
        "gpu_before": dict(initial_gpu),
        "model_diagnostics": dict(diagnostics),
        "stages": {
            "scan_seconds": 0.0,
            "detection_seconds": 0.0,
            "embedding_seconds": 0.0,
            "distance_seconds": 0.0,
            "dedup_seconds": 0.0,
            "dedup_wall_seconds": 0.0,
            "dedup_decode_seconds": 0.0,
            "dedup_rgb_seconds": 0.0,
            "dedup_lookup_seconds": 0.0,
            "dedup_validation_seconds": 0.0,
            "dedup_prefilter_seconds": 0.0,
            "dedup_aborted_pass_seconds": 0.0,
            "tag_seconds": 0.0,
            "curation_seconds": 0.0,
            "export_seconds": 0.0,
        },
        "dedup": _empty_dedup_stats(),
    }
    for count, (video, manual_paths) in enumerate(references_by_video, 1):
        print(f"[{count}/{len(videos)}] {video.name} ({mode})", flush=True)
        job: dict[str, Any] = {"video": str(video), "mode": mode, "status": "error", "characters": [], "warnings": []}
        folder: Path | None = None
        try:
            reader = VideoReader(video, decode_threads=args.decode_threads)
            fingerprint = reader.fingerprint
            try:
                lease.enter_context(exclusion.video_lease(ROOT, fingerprint, f"batch:{os.getpid()}"))
            except exclusion.ExclusionBusy as busy:
                # Another process owns this video; other videos still run.
                job.update({"status": "busy", "reason": str(busy)})
                print(f"  [busy] {busy}", file=sys.stderr, flush=True)
                outcomes.append(job)
                lease.close()
                continue
            manual_digests = [{"name": path.name, "sha256": _file_digest(path)} for path in manual_paths]
            force_nonce = uuid.uuid4().hex if args.force else None
            config_raw = json.dumps({"params": params, "references": manual_digests, "force_nonce": force_nonce}, sort_keys=True, ensure_ascii=False)
            config_hash = hashlib.sha256(config_raw.encode("utf-8")).hexdigest()[:12]
            source_hash = fingerprint.rsplit(":", 1)[-1][:12]
            folder = state_root / "runs" / f"{_clean(video.stem)}_{source_hash}_{config_hash}"
            marker = folder / "batch-result.json"
            if not args.force and marker.is_file():
                try:
                    old = json.loads(marker.read_text(encoding="utf-8"))
                    old_files = old.get("saved_files", [])
                    if _verified_completed_outputs(folder, old, fingerprint, config_hash):
                        try:
                            images, public_files, active = _publish_batch_images(
                                folder,
                                old,
                                output_root=output_root,
                                layout_state_root=publication_state_root(output_root),
                                video=video,
                                fingerprint=fingerprint,
                                receipt_paths=[marker, report_path],
                            )
                            job.update({"status": "skipped", "reason": "verified completed output and exact-PTS manifest", "output": str(folder), "images_directory": str(images), "files": len(old_files), "png_count": len(public_files), "saved_files": old_files, "outputs": old["outputs"], "public_files": public_files, "public_publication_id": active.get("publication_id"), "fingerprint": fingerprint, "config_hash": config_hash})
                            _write_json(marker, old)
                            register_public_receipts(
                                output_root,
                                video.name,
                                fingerprint,
                                state_root=publication_state_root(output_root),
                                receipt_paths=[marker],
                                publication_id=active.get("publication_id"),
                            )
                        except (OSError, ValueError, KeyError, TypeError) as exc:
                            job.update({"status": "error", "error": f"Public output could not be verified: {type(exc).__name__}: {exc}", "output": str(folder), "fingerprint": fingerprint, "config_hash": config_hash})
                        outcomes.append(job)
                        print(f"  [{job['status']}] {job.get('images_directory', 'public images unavailable')}", flush=True)
                        continue
                    if old.get("status") == "completed" and old.get("fingerprint") == fingerprint and old.get("config_hash") == config_hash:
                        job["warnings"].append("Completed output manifest did not verify; unverified PTS outputs will be regenerated and per-group verified outputs may be reused")
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            folder.mkdir(parents=True, exist_ok=True)
            job.update({"fingerprint": fingerprint, "config_hash": config_hash, "output": str(folder), "output_root": str(output_root), "state_root": str(state_root), "model_revision": revision, "output_manifest_revision": OUTPUT_MANIFEST_REVISION})

            scan_started = time.perf_counter()
            scanned, scan_stats, scan_key = scan_video(
                reader,
                cache_root / source_hash,
                fingerprint,
                chunk_seconds=args.chunk_seconds,
                progress=lambda done, total: print(f"  [scan] chunks {done}/{total}", flush=True) if done == total or done % 12 == 0 else None,
            )
            batch_telemetry["stages"]["scan_seconds"] += time.perf_counter() - scan_started
            candidates, selection_stats = select_candidates(
                scanned,
                spacing_seconds=args.candidate_spacing,
                video_path=video,
            )
            quality_rank = _quality_ranker(scanned)
            candidate_by_pts = {int(item["pts"]): dict(item) for item in candidates}
            job["analysis"] = {"scan": scan_stats, "selection": selection_stats, "scan_key": scan_key}
            print(f"  [candidates] decoded={scan_stats['frames_decoded']} selected={len(candidates)}", flush=True)

            groups: list[dict[str, Any]] = []
            dedup_reports: list[dict[str, Any]] = []
            dedup_aggregate = _empty_dedup_stats()

            def deduplicate_group(
                items: Sequence[Mapping[str, Any]],
                label: str,
                *,
                preserve_scan_events: bool = False,
                pixel_hash_distance: int = 5,
                rgb_similarity_threshold: float = 0.02,
            ) -> list[dict[str, Any]]:
                progress = lambda done, total: print(
                    f"  [dedup:{label}] candidates {done}/{total}", flush=True
                ) if done > 0 and (done == total or done % 256 == 0) else None
                result, stats = _deduplicate_with_reader(
                    items,
                    reader,
                    progress=progress,
                    preserve_scan_events=preserve_scan_events,
                    pixel_hash_distance=pixel_hash_distance,
                    rgb_similarity_threshold=rgb_similarity_threshold,
                )
                dedup_reports.append({"group": label, **stats})
                for key in (
                    "candidates_completed", "candidates_total", "pixel_comparisons", "frame_lookups",
                    "reader_opens", "decoded_frames", "seek_fallbacks", "decode_seconds", "rgb_seconds",
                    "lookup_seconds", "validation_seconds", "prefilter_seconds",
                    "aborted_pass_seconds", "dedup_pass_seconds", "dedup_seconds",
                    "dedup_wall_seconds", "wall_seconds",
                ):
                    dedup_aggregate[key] += stats[key]
                    batch_telemetry["dedup"][key] += stats[key]
                dedup_aggregate["optimized_attempted"] = bool(
                    dedup_aggregate["optimized_attempted"] or stats["optimized_attempted"]
                )
                dedup_aggregate["optimized_used"] = bool(
                    dedup_aggregate["optimized_used"] or stats["optimized_used"]
                )
                dedup_aggregate["sequential_fast_path"] = bool(
                    dedup_aggregate["sequential_fast_path"] or stats["sequential_fast_path"]
                )
                dedup_aggregate["strict_fallback"] = bool(
                    dedup_aggregate["strict_fallback"] or stats["strict_fallback"]
                )
                dedup_aggregate["max_cached_images"] = max(
                    int(dedup_aggregate["max_cached_images"]), int(stats["max_cached_images"])
                )
                if stats["fallback_reason"]:
                    dedup_aggregate.setdefault("fallback_reasons", []).append(
                        {"group": label, "reason": stats["fallback_reason"]}
                    )
                dedup_aggregate["fallback_reason"] = stats["fallback_reason"]
                batch_telemetry["dedup"]["optimized_attempted"] = bool(
                    batch_telemetry["dedup"]["optimized_attempted"] or stats["optimized_attempted"]
                )
                batch_telemetry["dedup"]["optimized_used"] = bool(
                    batch_telemetry["dedup"]["optimized_used"] or stats["optimized_used"]
                )
                batch_telemetry["dedup"]["sequential_fast_path"] = bool(
                    batch_telemetry["dedup"]["sequential_fast_path"] or stats["sequential_fast_path"]
                )
                batch_telemetry["dedup"]["strict_fallback"] = bool(
                    batch_telemetry["dedup"]["strict_fallback"] or stats["strict_fallback"]
                )
                batch_telemetry["dedup"]["max_cached_images"] = max(
                    int(batch_telemetry["dedup"]["max_cached_images"]), int(stats["max_cached_images"])
                )
                if stats["fallback_reason"]:
                    batch_telemetry["dedup"].setdefault("fallback_reasons", []).append(
                        {"video": video.name, "group": label, "reason": stats["fallback_reason"]}
                    )
                batch_telemetry["dedup"]["fallback_reason"] = stats["fallback_reason"]
                if stats["strict_fallback"]:
                    batch_telemetry["dedup"].setdefault("fallbacks", []).append(
                        {"video": video.name, "group": label, "reason": stats["fallback_reason"]}
                    )
                batch_telemetry["stages"]["dedup_seconds"] += float(stats["dedup_seconds"])
                batch_telemetry["stages"]["dedup_wall_seconds"] += float(stats["dedup_wall_seconds"])
                batch_telemetry["stages"]["dedup_decode_seconds"] += float(stats["decode_seconds"])
                batch_telemetry["stages"]["dedup_rgb_seconds"] += float(stats["rgb_seconds"])
                batch_telemetry["stages"]["dedup_lookup_seconds"] += float(stats["lookup_seconds"])
                batch_telemetry["stages"]["dedup_validation_seconds"] += float(stats["validation_seconds"])
                batch_telemetry["stages"]["dedup_prefilter_seconds"] += float(stats["prefilter_seconds"])
                batch_telemetry["stages"]["dedup_aborted_pass_seconds"] += float(stats["aborted_pass_seconds"])
                print(
                    f"  [dedup:{label}] comparisons={stats['pixel_comparisons']} lookups={stats['frame_lookups']} "
                    f"opens={stats['reader_opens']} decoded={stats['decoded_frames']} "
                    f"seek_fallbacks={stats['seek_fallbacks']} decode={stats['decode_seconds']:.2f}s "
                    f"rgb={stats['rgb_seconds']:.2f}s lookup={stats['lookup_seconds']:.2f}s "
                    f"validation={stats['validation_seconds']:.2f}s prefilter={stats['prefilter_seconds']:.2f}s "
                    f"dedup_wall={stats['dedup_wall_seconds']:.2f}s "
                    f"optimized_used={stats['optimized_used']} fallback={stats['fallback_reason']}",
                    flush=True,
                )
                return result

            identity_records: list[dict[str, Any]] = []
            identity_stats: dict[str, Any] = {}
            curation_audit: dict[str, Any] | None = None
            if mode == "cute":
                if model is None:
                    raise RuntimeError("cute character curation model was not initialized")
                cute_candidates: list[dict[str, Any]] = []
                for scan_item in candidates:
                    cute_candidates.append(
                        _cute_candidate_for_curation(scan_item, quality_rank(scan_item))
                    )
                curation_audit_path = folder / "curation-report.json"
                curated, curation_audit, curation_stats = _curate_character_candidates(
                    model,
                    reader,
                    cute_candidates,
                    folder=folder,
                    cache_root=cache_root / source_hash,
                    source_fingerprint=fingerprint,
                    tagger_revision=str(getattr(model, "tag_revision", revision)),
                    diagnostics=diagnostics,
                    minimum_cut_quality=float(args.cut_quality_min),
                )
                _write_json(curation_audit_path, curation_audit)
                job["analysis"]["curation"] = {
                    **curation_stats,
                    "revision": CUTE_CURATION_REVISION,
                    "tagger_revision": curation_audit["tagger_revision"],
                    "tagger_backend": curation_audit["tagger_backend"],
                    "identity_model_used": False,
                    "ccip_used": False,
                    "identity_id_emitted": False,
                    "report": str(curation_audit_path),
                    "summary": curation_audit["report"].get("summary", curation_audit["report"].get("counts", {})),
                }
                batch_telemetry["stages"]["tag_seconds"] += float(curation_stats["tag_seconds"])
                batch_telemetry["stages"]["curation_seconds"] += float(curation_stats["curation_seconds"])
                groups.append({
                    "reference_group": None,
                    "index": 1,
                    "mode": "cute",
                    "support_frames": None,
                    "evidence": [],
                    "candidates": curated,
                    "distance_stats": {},
                    "curation": job["analysis"]["curation"],
                })
            elif mode in {"character", "auto-topn"}:
                if model is None:
                    raise RuntimeError("Identity model was not initialized")
                identity_started = time.perf_counter()
                identity_records, identity_stats = analyze_identity(
                    reader,
                    model,
                    candidates,
                    cache_root / source_hash,
                    scan_key,
                    str(getattr(model, "identity_revision", revision)),
                    chunk_seconds=args.chunk_seconds,
                    # The analysis helper currently passes its historical
                    # preview limit (3); comprehensive character modes must
                    # retain every valid, non-overlapping detector result.
                    character_boxes=lambda detections, size, _limit: character_boxes(detections, size),
                    crop_image=_crop,
                    progress=lambda done, total: print(f"  [detect+CCIP] chunks {done}/{total}", flush=True) if done == total or done % 12 == 0 else None,
                )
                batch_telemetry["stages"]["detection_seconds"] += float(identity_stats.get("detect_seconds", 0.0))
                batch_telemetry["stages"]["embedding_seconds"] += float(identity_stats.get("embed_seconds", 0.0))
                identity_stats["total_seconds"] = time.perf_counter() - identity_started
                job["analysis"]["identity_cache"] = {key: value for key, value in identity_stats.items() if key in {"chunks_cache_hit", "chunks_rebuilt", "preprocess_revision", "detect_calls", "detect_seconds", "embed_calls", "embed_seconds", "embedded_crops", "total_seconds"}}

                selected_groups: list[ReferenceGroup]
                if mode == "character":
                    selected_groups = manual_references(manual_paths)
                else:
                    selected_groups, auto_stats = _automatic_reference_groups(model, identity_records, args.characters_per_video)
                    auto_stats["requested_top_n"] = args.characters_per_video
                    auto_stats["selection_semantics"] = "frequency-ranked top-N identity subset; not comprehensive identity coverage"
                    job["analysis"]["frequency_selection"] = auto_stats
                    batch_telemetry["stages"]["distance_seconds"] += float(auto_stats.get("distance_seconds", 0.0))
                for group_index, group in enumerate(selected_groups, 1):
                    if group.mode == "auto-topn":
                        for evidence in group.evidence:
                            frame = reader.get_frame(int(evidence["source_pts"]), max_side=960).image
                            bbox = evidence["bbox"]
                            group.images.append(_crop(frame, [float(bbox[0]) * frame.width, float(bbox[1]) * frame.height, float(bbox[2]) * frame.width, float(bbox[3]) * frame.height]))
                        # These crops are exactly the ones already embedded in
                        # the analysis cache; keep the cached vectors as refs.
                        evidence_pts = {int(item["source_pts"]) for item in group.evidence}
                        group_records = [item for item in identity_records if int(item["pts"]) in evidence_pts and item.get("feature") is not None]
                        if group_records:
                            # Preserve evidence order and avoid another CCIP call.
                            feature_by_pair = {(int(item["pts"]), tuple(item["bbox"])): np.asarray(item["feature"], dtype=np.float32) for item in group_records}
                            ordered_vectors = []
                            for evidence in group.evidence:
                                key = (int(evidence["source_pts"]), tuple(evidence["bbox"]))
                                if key in feature_by_pair:
                                    ordered_vectors.append(feature_by_pair[key])
                            ref_features = np.stack(ordered_vectors).astype(np.float32, copy=False)
                        else:
                            ref_features = np.asarray(model.embed(group.images), dtype=np.float32)
                    else:
                        ref_features = np.asarray(model.embed(group.images), dtype=np.float32)
                    identity_candidates, distance_stats = _identity_scores(model, ref_features, identity_records)
                    batch_telemetry["stages"]["distance_seconds"] += float(distance_stats.get("distance_seconds", 0.0))
                    merged: list[dict[str, Any]] = []
                    for record in identity_candidates:
                        scan_item = candidate_by_pts.get(int(record["pts"]))
                        if scan_item is None:
                            continue
                        item = {**scan_item, **record}
                        item["quality"] = quality_rank(scan_item)
                        item["score"] = 0.72 * float(item.get("identity", 0.0)) + 0.25 * item["quality"] + 0.03 * float(item.get("detector_confidence", 0.0))
                        merged.append(item)
                    merged = deduplicate_group(merged, f"character-{group_index:02d}")
                    groups.append({"reference_group": group, "index": group_index, "mode": group.mode, "support_frames": group.support_frames, "evidence": group.evidence, "candidates": merged, "distance_stats": distance_stats})
            else:
                merged = []
                for scan_item in candidates:
                    item = dict(scan_item)
                    item["quality"] = quality_rank(scan_item)
                    item["identity"] = None
                    item["distance"] = None
                    item["score"] = item["quality"]
                    merged.append(item)
                merged = deduplicate_group(
                    merged,
                    "all",
                    preserve_scan_events=True,
                    # Coverage anchors can carry small but meaningful face
                    # changes that are invisible to a whole-frame similarity
                    # threshold. In all mode collapse only effectively exact
                    # repeats; keep the full scene/coverage sampling otherwise.
                    pixel_hash_distance=0,
                    rgb_similarity_threshold=0.000001,
                )
                groups.append({"reference_group": None, "index": 1, "mode": "all", "support_frames": None, "evidence": [], "candidates": merged, "distance_stats": {}})

            dedup_aggregate["groups"] = dedup_reports
            job["analysis"]["dedup"] = dedup_aggregate

            if (wanted or avoided) and mode == "all" and any(item["scope"] != "frame" for item in [*wanted, *avoided]):
                raise ValueError("all mode supports frame: PixAI filters only; subject: filters require --mode character or --mode auto-topn")
            tag_stats = _filter_tags(model, reader, groups, wanted, avoided, cache_root / source_hash, mode=mode) if wanted or avoided else {"tagged": 0, "tag_seconds": 0.0, "cache_hits": 0}
            batch_telemetry["stages"]["tag_seconds"] += float(tag_stats.get("tag_seconds", 0.0))
            job["analysis"]["tagging"] = tag_stats

            export_started = time.perf_counter()
            export_jobs: list[dict[str, Any]] = []
            saved_files: list[str] = []
            output_manifest: list[dict[str, Any]] = []
            for group in groups:
                group_index = int(group["index"])
                group_dir_name = "frames" if mode == "all" else f"character_{group_index:02d}"
                group_dir = _safe_output_path(folder, group_dir_name)
                if group_dir is None:
                    raise RuntimeError(f"Output group path escapes its output folder: {group_dir_name}")
                group_dir.mkdir(parents=True, exist_ok=True)
                reference_paths: list[str] = []
                reference_group = group["reference_group"]
                if reference_group is not None:
                    for ref_index, image in enumerate(reference_group.images, 1):
                        ref_path = group_dir / f"reference_{ref_index:02d}.png"
                        if not _valid_image(ref_path):
                            _atomic_save_image(image, ref_path)
                        reference_paths.append(ref_path.relative_to(folder).as_posix())
                exports: list[dict[str, Any]] = []
                warnings: list[str] = []
                for candidate in group["candidates"]:
                    pts = int(candidate["pts"])
                    name = f"frame_pts_{pts}.png"
                    destination = _safe_output_path(folder, f"{group_dir_name}/{name}")
                    original = _safe_output_path(folder, f"{group_dir_name}/originals/{name}") if not args.no_logo_removal else None
                    if destination is None or (not args.no_logo_removal and original is None):
                        raise RuntimeError(f"Output path escapes its per-video folder for PTS {pts}")
                    partial = _matching_partial_export(folder, group_dir_name, pts, fingerprint, config_hash)
                    partial_group_dir, partial_output_path, partial_export = partial if partial is not None else (None, None, None)
                    output_verified = (
                        partial_export is not None
                        and _verified_partial_png(partial_output_path, partial_export.get("sha256"))
                    )
                    original_verified = (
                        partial_export is not None
                        and original is not None
                        and _verified_partial_original(partial_group_dir, partial_export, pts)
                    )
                    if args.no_logo_removal:
                        if not output_verified:
                            export_jobs.append({"pts": pts, "destination": destination, "original": None, "output_verified": False, "group_index": group_index, "candidate": candidate, "group": group, "group_dir": group_dir})
                    elif not output_verified or not original_verified:
                        export_jobs.append({
                            "pts": pts,
                            "destination": destination,
                            "original": original,
                            "output_verified": output_verified,
                            "group_index": group_index,
                            "candidate": candidate,
                            "reuse_original": original_verified,
                            "group": group,
                            "group_dir": group_dir,
                        })
                # Group metadata is assembled after the shared ordered export pass.
                group["group_dir"] = group_dir
                group["group_dir_name"] = group_dir_name
                group["reference_paths"] = reference_paths
                group["exports"] = exports
                group["warnings"] = warnings

            output_map: dict[int, list[Path]] = {}
            for entry in export_jobs:
                if entry.get("reuse_original"):
                    continue
                output_map.setdefault(entry["pts"], []).append(entry["original"] or entry["destination"])
            missing_pts: set[int] = set()
            if output_map:
                missing_pts = reader.export_pngs(output_map)
            if missing_pts:
                raise RuntimeError(f"Exact PTS export could not find {len(missing_pts)} selected frames: {sorted(missing_pts)[:8]}")
            for entry in export_jobs:
                destination: Path = entry["destination"]
                original: Path | None = entry["original"]
                if args.no_logo_removal or entry.get("output_verified"):
                    continue
                if original is None or not _valid_image(original):
                    raise RuntimeError(f"Native original export missing for PTS {entry['pts']}")
                from anime_picker.server import _process_logo_frame
                original_relative = original.relative_to(entry["group_dir"]).as_posix()
                with Image.open(original) as image:
                    processed, logo_metadata, frame_warnings = _process_logo_frame(image.convert("RGB"), [], original_relative_path=original_relative)
                entry["logo_metadata"] = logo_metadata
                entry["warnings"] = frame_warnings
                entry["group"]["warnings"].extend(frame_warnings)
                if logo_metadata.get("applied"):
                    _atomic_save_image(processed, destination)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temp = destination.with_name(destination.name + ".tmp")
                    try:
                        shutil.copyfile(original, temp)
                        os.replace(temp, destination)
                    finally:
                        temp.unlink(missing_ok=True)

            for group in groups:
                group_dir: Path = group["group_dir"]
                group_dir_name: str = group["group_dir_name"]
                metadata_relative = (group_dir / "metadata.json").relative_to(folder).as_posix()
                exports: list[dict[str, Any]] = []
                group_candidates = group["candidates"]
                for candidate in group_candidates:
                    pts = int(candidate["pts"])
                    name = f"frame_pts_{pts}.png"
                    destination = group_dir / name
                    if not _valid_image(destination):
                        raise RuntimeError(f"Output PNG is missing or invalid: {destination}")
                    original = group_dir / "originals" / name if not args.no_logo_removal else None
                    if original is not None and not _valid_image(original):
                        raise RuntimeError(f"Original PTS PNG is missing or invalid: {original}")
                    logo_metadata = next((item.get("logo_metadata") for item in export_jobs if item["pts"] == pts and item["destination"] == destination), None)
                    output_digest = _file_digest(destination)
                    original_digest = _file_digest(original) if original is not None else None
                    export_record = {
                        "file": name,
                        "pts": pts,
                        "source_pts": pts,
                        "time": float(candidate["time"]),
                        "bbox": candidate.get("bbox"),
                        "score": float(
                            candidate.get("cut_quality", candidate.get("score", 0.0))
                            if group["mode"] == "cute"
                            else candidate.get("score", 0.0)
                        ),
                        "sha256": output_digest,
                        "quality": float(candidate.get("quality", 0.0)),
                        "quality_metrics": {key: candidate.get(key) for key in ("scan_sharpness", "brightness", "contrast", "local_change", "scene_change")},
                        "selection_reasons": candidate.get("reasons", []),
                        "tags": candidate.get("tags", {}),
                        "tag_reasons": candidate.get("tag_reasons", []),
                        "original_relative_path": original.relative_to(group_dir).as_posix() if original is not None else None,
                        "original_sha256": original_digest,
                        "logo_cleanup": logo_metadata or ({"requested": False, "applied": False, "status": "disabled"} if args.no_logo_removal else {"requested": True, "applied": None, "status": "reused_existing_output"}),
                    }
                    if group["mode"] != "cute":
                        export_record.update({
                            "detector": {"kind": candidate.get("detector_kind"), "confidence": candidate.get("detector_confidence")},
                            "identity": candidate.get("identity"),
                            "identity_corroboration": candidate.get("identity_corroboration"),
                            "distance": candidate.get("distance"),
                            "identity_group": {"mode": group["mode"], "index": group["index"]},
                        })
                    elif group["mode"] == "cute":
                        export_record.update({
                            "character_presence": candidate.get("character_presence"),
                            "presence_score": candidate.get("presence_score"),
                            "cut_quality": candidate.get("cut_quality"),
                            "quality_components": candidate.get("quality_components"),
                            "presence_evidence": candidate.get("presence_evidence"),
                            "curation_group_id": candidate.get("group_id"),
                            "curation_revision": CUTE_CURATION_REVISION,
                        })
                    exports.append(export_record)
                    saved_files.append(f"{group_dir_name}/{name}")
                    manifest_entry = {
                        "file": f"{group_dir_name}/{name}",
                        "metadata": metadata_relative,
                        "pts": pts,
                        "sha256": output_digest,
                        "original_file": original.relative_to(folder).as_posix() if original is not None else None,
                        "original_sha256": original_digest,
                    }
                    if group["mode"] == "cute":
                        manifest_entry["curation_group_id"] = candidate.get("group_id")
                    else:
                        manifest_entry["identity_group"] = {"mode": group["mode"], "index": group["index"]}
                    output_manifest.append(manifest_entry)
                reference_group = group["reference_group"]
                char_metadata = {
                    "mode": group["mode"],
                    "selection_revision": CUTE_SELECTION_REVISION if group["mode"] == "cute" else SELECTION_REVISION,
                    "dedup_revision": CUTE_CURATION_REVISION if group["mode"] == "cute" else DEDUP_REVISION,
                    "output_manifest_revision": OUTPUT_MANIFEST_REVISION,
                    "support_frames": group["support_frames"],
                    "references": group["evidence"],
                    "reference_files": group["reference_paths"],
                    "source": str(video),
                    "source_timebase": str(reader.info.time_base),
                    "source_fingerprint": fingerprint,
                    "config_hash": config_hash,
                    "model_revision": revision,
                    "wanted": list(wanted), "avoided": list(avoided),
                    "analysis": job["analysis"],
                    "stats": (
                        {"selected": len(group_candidates), "curation": group.get("curation", {})}
                        if group["mode"] == "cute"
                        else {"selected": len(group_candidates), "identity": group["distance_stats"]}
                    ),
                    "warnings": group["warnings"],
                    "exports": exports,
                }
                if group["mode"] == "cute":
                    char_metadata["curation"] = group.get("curation")
                _write_json(group_dir / "metadata.json", char_metadata)
                job["characters"].append({"index": group["index"], "mode": group["mode"], "status": "completed" if exports else "no_matches", "references": group["reference_paths"], "exported": len(exports), "warnings": group["warnings"], "directory": str(group_dir)})
                print(f"    [export] {len(exports)} native PTS PNGs in {group_dir_name}", flush=True)
            batch_telemetry["stages"]["export_seconds"] += time.perf_counter() - export_started

            if reader.fingerprint != fingerprint:
                raise RuntimeError("Video changed while being processed; outputs must be rechecked")
            job["saved_files"] = saved_files
            job["outputs"] = output_manifest
            job["status"] = "completed" if saved_files and all(item["status"] == "completed" for item in job["characters"]) else "incomplete"
        except KeyboardInterrupt:
            job.update({"status": "interrupted", "error": "User interrupted the batch"})
            outcomes.append(job)
            if folder:
                _write_json(folder / "batch-result.json", job)
            lease.close()
            break
        except Exception as exc:
            job.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
        if folder:
            _write_json(folder / "batch-result.json", job)
        if folder and job.get("status") == "completed":
            try:
                marker_value = dict(job)
                images, public_files, active = _publish_batch_images(
                    folder,
                    marker_value,
                    output_root=output_root,
                    layout_state_root=publication_state_root(output_root),
                    video=video,
                    fingerprint=str(job["fingerprint"]),
                    receipt_paths=[folder / "batch-result.json", report_path],
                )
                job["images_directory"] = str(images)
                job["public_files"] = public_files
                job["public_publication_id"] = active.get("publication_id")
                job["png_count"] = len(public_files)
                job["images_directory_relative"] = marker_value["images_directory"]
                _write_json(folder / "batch-result.json", marker_value)
                register_public_receipts(
                    output_root,
                    video.name,
                    str(job["fingerprint"]),
                    state_root=publication_state_root(output_root),
                    receipt_paths=[folder / "batch-result.json"],
                    publication_id=active.get("publication_id"),
                )
                print(f"  [images] {images} ({len(public_files)} PNGs)", flush=True)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                job["publish_error"] = f"{type(exc).__name__}: {exc}"
                job["status"] = "error"
                # Keep the completed internal marker intact so the next run
                # can retry public publication without rescanning/redecoding.
                print(f"  [publish:error] {job['publish_error']}", file=sys.stderr, flush=True)
        lease.close()
        outcomes.append(job)
        print(f"  [result] {job['status']}", flush=True)

    finished = len(outcomes) == len(videos) and all(job["status"] in {"completed", "skipped"} for job in outcomes)
    return {
        "batch_id": batch_id,
        "source": str(source),
        "output_root": str(output_root),
        "state_root": str(state_root),
        "params": params,
        "videos_total": len(videos),
        "videos_attempted": len(outcomes),
        "status": "completed" if finished else "incomplete",
        "jobs": outcomes,
        "telemetry": batch_telemetry,
    }, report_path, finished


def run(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    source = args.source.resolve()
    videos = discover_videos(source)
    if not videos:
        print(f"No MP4/MKV files in {source}", file=sys.stderr)
        return 2
    shared = args.reference.resolve() if args.reference else None
    if shared is not None and args.mode != "character":
        raise ValueError("--reference requires --mode character; it is never silently ignored")
    references_by_video = [
        (video, reference_files(video, args.reference_dir.resolve(), shared) if args.mode == "character" else [])
        for video in videos
    ]
    if args.mode == "character":
        missing = [video.name for video, refs in references_by_video if not refs]
        if missing:
            raise ValueError(f"character mode requires a reference crop for every video; missing: {', '.join(missing)}")
    print(f"[batch] {len(videos)} videos found in {source}; mode={args.mode}", flush=True)
    for video, refs in references_by_video:
        detail = (
            f"explicit references ({len(refs)})" if args.mode == "character"
            else f"frequency top-{args.characters_per_video} identity subset (not all identities)" if args.mode == "auto-topn"
            else "character-present cuts ranked and visually deduplicated" if args.mode == "cute"
            else "all good scene candidates"
        )
        print(f"  - {video.name}: {detail}", flush=True)
    if args.dry_run:
        print("[dry-run] No model inference, frame decoding, or file export performed.")
        return 0

    wanted, avoided = _conditions(args.want), _conditions(args.avoid)
    if args.mode == "all" and any(item["scope"] != "frame" for item in [*wanted, *avoided]):
        raise ValueError("all mode supports frame: PixAI filters only; subject: filters require --mode character or --mode auto-topn")
    if args.mode == "cute" and any(item["scope"] == "subject" for item in [*wanted, *avoided]):
        raise ValueError("cute mode ranks full-frame character tags; subject: filters require --mode character or --mode auto-topn")
    needs_model = args.mode in {"cute", "character", "auto-topn"} or bool(wanted or avoided)
    # Registered for the process lifetime so the UI and other lock-aware
    # processes can tell this batch from one that predates the locks.
    registration = exclusion.register_batch_process(ROOT)
    try:
        # A model run contends with any lock-less batch for the GPU; a
        # model-free run only with one publishing to the same output root.
        exclusion.ensure_no_legacy_batch(ROOT, None if needs_model else args.output_root.resolve())
        return _run_locked(args, source, videos, references_by_video, wanted, avoided, needs_model)
    except exclusion.ExclusionBusy as error:
        print(f"[batch:busy] {error}", file=sys.stderr, flush=True)
        return 3
    finally:
        registration.release()


def _run_locked(args: argparse.Namespace, source: Path, videos: Sequence[Path], references_by_video: Sequence[tuple[Path, list[Path]]], wanted: Sequence[Mapping[str, str]], avoided: Sequence[Mapping[str, str]], needs_model: bool) -> int:
    from anime_picker.models import ModelService

    model = (
        ModelService(
            model_root=args.model_root.resolve(),
            require_gpu=True,
            runtime_cache_root=args.model_cache_root.resolve() if args.model_cache_root else None,
        )
        if needs_model
        else None
    )
    if model is not None:
        model = exclusion.guard_model(model, ROOT)
    revision = "none (low-cost all-frame mode; no GPU model loaded)"
    diagnostics: Mapping[str, Any] = {"ready": True, "models_loaded": False}
    gpu_before = _gpu_memory_snapshot() if needs_model else {"available": False, "reason": "no GPU model stage"}
    report_path: Path | None = None
    report: dict[str, Any] | None = None
    finished = False
    cleanup: Mapping[str, Any] = {"performed": False}
    try:
        if model is not None:
            _validate_conditions(model, wanted, avoided)
            diagnostics = model.diagnostics()
            if not diagnostics.get("ready", False):
                raise RuntimeError(f"Model runtime not ready: {diagnostics}")
            revision = str(model.revision)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            except Exception:
                pass
        report, report_path, finished = _run_jobs(args, videos, references_by_video, wanted, avoided, model, revision, diagnostics, gpu_before)
        if model is not None and report is not None:
            telemetry = report.setdefault("telemetry", {})
            telemetry["model_diagnostics_before_inference"] = telemetry.get(
                "model_diagnostics", dict(diagnostics)
            )
            try:
                after_diagnostics = model.diagnostics()
            except Exception as error:
                telemetry["model_diagnostics_refresh_error"] = f"{type(error).__name__}: {error}"
            else:
                telemetry["model_diagnostics_after_inference"] = dict(after_diagnostics)
                telemetry["model_diagnostics"] = dict(after_diagnostics)
    finally:
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                cleanup = close()
            else:
                cleanup = {"performed": False, "reason": "ModelService.close unavailable"}
    if report is None or report_path is None:
        return 2
    report.setdefault("telemetry", {})["gpu_after_cleanup"] = _gpu_memory_snapshot() if needs_model else {"available": False, "reason": "no GPU model stage"}
    report["telemetry"]["cleanup"] = dict(cleanup)
    report["telemetry"]["model_loaded"] = bool(model is not None)
    _write_json(report_path, report)
    if not args.no_archive:
        from scripts.archive_completed import archive_report
        try:
            archive = archive_report(report, source_dir=source, completed_dir=source.parent / "completed", output_root=args.output_root.resolve())
            report["archive"] = archive
            if archive.get("errors"):
                finished = False
        except (OSError, ValueError, KeyError, TypeError) as error:
            finished = False
            report["archive_error"] = f"{type(error).__name__}: {error}"
            print(f"[archive:error] {report['archive_error']}", file=sys.stderr, flush=True)
        report["status"] = "completed" if finished else "incomplete"
        _write_json(report_path, report)
    reference_errors: list[str] = []
    for job in report.get("jobs", []):
        if not isinstance(job, Mapping) or job.get("status") not in {"completed", "skipped"}:
            continue
        publication_id = job.get("public_publication_id")
        video_value = job.get("video")
        fingerprint = job.get("fingerprint")
        if not isinstance(publication_id, str) or not isinstance(video_value, str) or not isinstance(fingerprint, str):
            continue
        try:
            register_public_receipts(
                args.output_root.resolve(),
                Path(video_value).name,
                fingerprint,
                state_root=publication_state_root(args.output_root.resolve()),
                receipt_paths=[report_path],
                publication_id=publication_id,
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            reference_errors.append(f"{video_value}: {type(error).__name__}: {error}")
    if reference_errors:
        finished = False
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["delivery_reference_errors"] = reference_errors
        report["status"] = "incomplete"
        _write_json(report_path, report)
    else:
        # Receipt registration may have reconciled this report to an archived
        # publication after a concurrent UI or batch run changed the live set.
        report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"[batch] {report['status']}; report: {report_path}", flush=True)
    return 0 if finished else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"[batch:error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
