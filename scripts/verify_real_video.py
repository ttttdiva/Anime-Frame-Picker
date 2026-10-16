#!/usr/bin/env python3
"""Run a bounded, local real-video smoke check for Anime Frame Picker.

The check intentionally uses one short, known fixture from Episode 1.  It is
evidence that the runtime wiring works on a real file; it is not a precision
benchmark and it never uploads the source video.  Model loading is explicit
and GPU-required so a CPU fallback cannot look like a successful run.

The default fixture uses the blue-haired, yellow-eyed character around 70 s as
the reference and searches 55--85 s.  The interval contains a blonde knight
scene around 60 s and a red-haired character around 80 s for a small negative
embedding comparison.  All generated artifacts live below ``.cache/qa``.
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_VIDEO = ROOT / "source" / "The World's Strongest Witch Episode 1.mp4"
DEFAULT_OUT = ROOT / ".cache" / "qa"

# Normalized boxes are deliberately conservative.  They are only negative
# fixtures; callers can replace them with --negative when a source cut changes.
DEFAULT_NEGATIVES = (
    ("blonde-knight-60s", 60.0, (0.14, 0.06, 0.38, 0.99)),
    ("red-haired-80s", 80.0, (0.00, 0.00, 0.47, 0.96)),
)


# Set ANIME_PICKER_QA_TRACE_TIMEOUT=60 for a bounded traceback if CUDA/model
# startup stalls.  The timer is opt-in so ordinary --help and local checks stay
# quiet.  faulthandler writes to stderr, which the runner captures in the QA
# log alongside normal progress output.
_TRACE_TIMEOUT = float(os.environ.get("ANIME_PICKER_QA_TRACE_TIMEOUT", "0") or 0.0)
if _TRACE_TIMEOUT > 0:
    faulthandler.enable()
    faulthandler.dump_traceback_later(_TRACE_TIMEOUT, repeat=False)


def _safe(value: Any) -> Any:
    """Convert common model/runtime values into JSON-safe values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    return str(value)


def _image_size(image: Any) -> tuple[int, int]:
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        return int(size[0]), int(size[1])
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("model input is not image-shaped")
    return int(array.shape[1]), int(array.shape[0])


def _as_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if hasattr(image, "convert"):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image).astype(np.uint8), mode="RGB")


def _parse_box(value: Sequence[Any], width: int, height: int) -> list[float]:
    values = [float(item) for item in value]
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise ValueError("box must contain four finite numbers")
    if max(abs(item) for item in values) <= 1.5:
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    x0, x1 = sorted((max(0.0, min(float(width), values[0])), max(0.0, min(float(width), values[2]))))
    y0, y1 = sorted((max(0.0, min(float(height), values[1])), max(0.0, min(float(height), values[3]))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("box is too small after clipping")
    return [x0, y0, x1, y1]


def _detection_box(raw: Any, width: int, height: int) -> tuple[list[float], float, str] | None:
    if isinstance(raw, Mapping):
        box = raw.get("box", raw.get("bbox"))
        score = raw.get("score", raw.get("confidence", 1.0))
        kind = str(raw.get("kind", "person")).lower()
    else:
        box, score, kind = raw, 1.0, "person"
    if box is None:
        return None
    try:
        return _parse_box(box, width, height), float(score), kind
    except (TypeError, ValueError, OverflowError):
        return None


def _crop(image: Image.Image, box: Sequence[float]) -> Image.Image:
    width, height = image.size
    x0, y0, x1, y1 = [int(round(item)) for item in box]
    x0, y0 = max(0, min(width - 1, x0)), max(0, min(height - 1, y0))
    x1, y1 = max(x0 + 1, min(width, x1)), max(y0 + 1, min(height, y1))
    return image.crop((x0, y0, x1, y1)).convert("RGB")


def _frame_at(reader: Any, requested_time: float) -> Any:
    info = getattr(reader, "info", None)
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    start = max(0.0, float(requested_time))
    end = min(duration, start + 0.20) if duration > 0 else start + 0.20
    if end <= start:
        end = start + 0.04
    frames = iter(reader.sample(start, end, 0.04, max_side=None))
    frame = next(frames, None)
    if frame is None:
        raise RuntimeError(f"VideoReader returned no frame near {requested_time:.3f}s")
    return frame


def _load_model_class(spec: str | None) -> tuple[Any, str]:
    candidates = [item for item in [spec, os.environ.get("ANIME_PICKER_MODEL_SERVICE")] if item]
    candidates.extend(
        [
            "anime_picker.models:ModelService",
            "anime_picker.model_service:ModelService",
            "anime_picker.app:ModelService",
        ]
    )
    errors: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        module_name, separator, attribute = candidate.partition(":")
        if not separator:
            module_name, _, attribute = candidate.rpartition(".")
        if not module_name or not attribute:
            errors.append(f"{candidate}: expected module:Class")
            continue
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attribute), candidate
        except (ImportError, AttributeError) as error:
            errors.append(f"{candidate}: {error}")
    raise RuntimeError(
        "ModelService was not importable. Tried: " + "; ".join(errors)
    )


class RecordingModel:
    """Proxy that records the image shapes reaching each model boundary."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.detect_calls: list[tuple[int, int]] = []
        self.embed_calls: list[list[tuple[int, int]]] = []
        self.tag_calls: list[tuple[int, int]] = []
        self.distance_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def detect(self, image: Any) -> Any:
        self.detect_calls.append(_image_size(image))
        return self.inner.detect(image)

    def embed(self, images: Iterable[Any]) -> Any:
        batch = list(images)
        self.embed_calls.append([_image_size(item) for item in batch])
        return self.inner.embed(batch)

    def distance(self, references: Any, features: Any) -> Any:
        refs = np.asarray(references)
        items = np.asarray(features)
        self.distance_calls.append((tuple(refs.shape), tuple(items.shape)))
        return self.inner.distance(references, features)

    def tag(self, image: Any) -> Any:
        self.tag_calls.append(_image_size(image))
        return self.inner.tag(image)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _parse_scope_items(values: Sequence[str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for raw in values:
        scope, separator, tag = raw.partition(":")
        if not separator:
            scope, tag = "subject", raw
        scope, tag = scope.strip().lower(), tag.strip()
        if scope not in {"subject", "frame"} or not tag:
            raise ValueError(f"scope item must be subject:tag or frame:tag: {raw!r}")
        result.append({"scope": scope, "tag": tag})
    return result


def _parse_negative(raw: str) -> tuple[str, float, tuple[float, float, float, float]]:
    # label@time:x0,y0,x1,y1
    try:
        label_time, box_text = raw.split(":", 1)
        label, time_text = label_time.rsplit("@", 1)
        values = tuple(float(item) for item in box_text.split(","))
    except (TypeError, ValueError) as error:
        raise ValueError(f"negative must be label@time:x0,y0,x1,y1: {raw!r}") from error
    if len(values) != 4:
        raise ValueError(f"negative box must contain four values: {raw!r}")
    return label, float(time_text), values  # type: ignore[return-value]


def _select_reference_box(model: RecordingModel, frame: Any, explicit: str | None) -> list[float]:
    image = _as_rgb(frame.image)
    width, height = image.size
    if explicit:
        return _parse_box([float(item) for item in explicit.split(",")], width, height)
    detections = []
    for raw in model.detect(image) or []:
        parsed = _detection_box(raw, width, height)
        if parsed is None:
            continue
        box, score, kind = parsed
        area = (box[2] - box[0]) * (box[3] - box[1]) / max(1.0, width * height)
        detections.append((kind == "person", score, area, box))
    if not detections:
        raise RuntimeError("reference frame has no valid person detection; pass --reference-box")
    # Prefer a person box over a head box, then model confidence, then area.
    detections.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return detections[0][3]


def _progress_printer() -> Any:
    last_value = -1.0
    last_message = ""

    def progress(value: float, message: str) -> None:
        nonlocal last_value, last_message
        value = float(value)
        if message != last_message or value - last_value >= 0.05 or value >= 1.0:
            print(f"[search] {value:6.1%} {message}", flush=True)
            last_value, last_message = value, message

    return progress


def _verify_native_export(reader: Any, candidate: Mapping[str, Any], out_dir: Path, index: int) -> dict[str, Any]:
    pts = int(candidate["pts"])
    exact = reader.get_frame(pts, max_side=None)
    destination = out_dir / f"candidate-{index:02d}-pts-{pts}-native.png"
    exported = reader.export_png(pts, destination)
    with Image.open(destination) as image:
        exported_pixels = np.asarray(image.convert("RGB"))
    exact_pixels = np.asarray(_as_rgb(exact.image))
    pixel_equal = bool(np.array_equal(exact_pixels, exported_pixels))
    expected_time = float(exact.time)
    candidate_time = float(candidate.get("time", float("nan")))
    return {
        "pts": pts,
        "candidate_time": candidate_time,
        "source_time": expected_time,
        "time_delta": abs(candidate_time - expected_time),
        "time_matches": bool(math.isfinite(candidate_time) and abs(candidate_time - expected_time) <= 1e-6),
        "export_pts": int(exported.pts),
        "native_size": list(exact_pixels.shape[1::-1]),
        "export_size": list(exported_pixels.shape[1::-1]),
        "pixel_equal_rgb": pixel_equal,
        "export_path": str(destination.resolve()),
        "preview_path": str(candidate.get("preview_path", "")),
        "preview_exists": Path(str(candidate.get("preview_path", ""))).is_file(),
    }


def _hash_duplicates(candidates: Sequence[Mapping[str, Any]], window: float = 0.55, limit: int = 6) -> list[dict[str, Any]]:
    from anime_picker.scoring import dhash, hamming_distance, phash

    hashes: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            with Image.open(str(candidate["preview_path"])) as image:
                preview = image.convert("RGB")
            with Image.open(str(candidate["crop_path"])) as image:
                crop = image.convert("RGB")
            hashes.append(
                {
                    "candidate": candidate,
                    "full_phash": phash(preview),
                    "full_dhash": dhash(preview),
                    "crop_phash": phash(crop),
                    "crop_dhash": dhash(crop),
                }
            )
        except (OSError, KeyError, TypeError, ValueError):
            continue
    duplicates: list[dict[str, Any]] = []
    for left_index, left in enumerate(hashes):
        for right in hashes[left_index + 1 :]:
            left_candidate, right_candidate = left["candidate"], right["candidate"]
            if abs(float(left_candidate.get("time", 0.0)) - float(right_candidate.get("time", 0.0))) > window:
                continue
            full_distance = max(
                hamming_distance(left["full_phash"], right["full_phash"]),
                hamming_distance(left["full_dhash"], right["full_dhash"]),
            )
            crop_distance = max(
                hamming_distance(left["crop_phash"], right["crop_phash"]),
                hamming_distance(left["crop_dhash"], right["crop_dhash"]),
            )
            if full_distance <= limit and crop_distance <= limit:
                duplicates.append(
                    {
                        "left_id": left_candidate.get("id"),
                        "right_id": right_candidate.get("id"),
                        "left_time": left_candidate.get("time"),
                        "right_time": right_candidate.get("time"),
                        "full_hash_distance": full_distance,
                        "crop_hash_distance": crop_distance,
                    }
                )
    return duplicates


def _nearby_candidate_pairs(candidates: Sequence[Mapping[str, Any]], window: float = 0.55) -> list[dict[str, Any]]:
    """List close-in-time candidate pairs that need human visual review.

    Hashes are useful for an automated hint, but anime cuts can change crop
    boxes and overlays enough to evade a small Hamming threshold while still
    looking like the same pose.  Every close pair is therefore surfaced for
    contact-sheet review instead of being silently called distinct.
    """

    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(candidates):
        try:
            left_time = float(left.get("time", 0.0))
        except (TypeError, ValueError):
            continue
        for right in candidates[left_index + 1 :]:
            try:
                right_time = float(right.get("time", 0.0))
            except (TypeError, ValueError):
                continue
            delta = abs(left_time - right_time)
            if delta <= window:
                pairs.append(
                    {
                        "left_id": left.get("id"),
                        "right_id": right.get("id"),
                        "left_time": left_time,
                        "right_time": right_time,
                        "delta_seconds": delta,
                    }
                )
    return pairs


def _contact_sheet(candidates: Sequence[Mapping[str, Any]], path: Path) -> str | None:
    if not candidates:
        return None
    tile_w, tile_h, header_h = 480, 270, 52
    cols = 3
    rows = (len(candidates) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + header_h)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for index, candidate in enumerate(candidates):
        image_path = Path(str(candidate.get("preview_path", "")))
        if not image_path.is_file():
            image_path = Path(str(candidate.get("crop_path", "")))
        if not image_path.is_file():
            continue
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = (index % cols) * tile_w
        y = (index // cols) * (tile_h + header_h)
        draw.rectangle((x, y, x + tile_w, y + header_h), fill=(8, 8, 8))
        label = (
            f"{float(candidate.get('time', 0.0)):7.3f}s  "
            f"d={float(candidate.get('distance', 0.0)):.4f}  "
            f"s={float(candidate.get('score', 0.0)):.3f}"
        )
        draw.text((x + 8, y + 8), label, fill=(255, 255, 255), font=font)
        draw.text((x + 8, y + 30), str(candidate.get("id", ""))[:54], fill=(210, 210, 210), font=font)
        px = x + (tile_w - image.width) // 2
        py = y + header_h + (tile_h - image.height) // 2
        sheet.paste(image, (px, py))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92, optimize=True)
    return str(path.resolve())


def _scope_audit(
    model: RecordingModel,
    reader: Any,
    candidates: Sequence[Mapping[str, Any]],
    wanted: Sequence[Mapping[str, Any]],
    avoided: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    info = getattr(reader, "info", None)
    native_size = (int(getattr(info, "width", 0) or 0), int(getattr(info, "height", 0) or 0))
    embed_inputs = [size for batch in model.embed_calls for size in batch]
    analysis_frame_sizes = set(model.detect_calls)
    native_full_frame_inputs = [size for size in embed_inputs if size == native_size]
    analysis_full_frame_inputs = [size for size in embed_inputs if size in analysis_frame_sizes]
    full_frame_embed_inputs = sorted(set(native_full_frame_inputs + analysis_full_frame_inputs))
    boxes = []
    for candidate in candidates:
        bbox = candidate.get("bbox")
        if isinstance(bbox, Sequence) and len(bbox) == 4:
            try:
                area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
                boxes.append({"id": candidate.get("id"), "area_fraction": area, "bbox": list(bbox)})
            except (TypeError, ValueError):
                pass
    whole_frame_candidates = [item for item in boxes if item["area_fraction"] >= 0.985]
    output_scope_keys = all(
        isinstance(candidate.get("tags"), Mapping)
        and "subject" in candidate["tags"]
        and "frame" in candidate["tags"]
        for candidate in candidates
    ) if (wanted or avoided) and candidates else None
    return {
        "native_size": list(native_size),
        "detect_inputs": [list(item) for item in model.detect_calls],
        "identity_embed_inputs": [list(item) for item in embed_inputs],
        "analysis_frame_sizes": [list(item) for item in sorted(analysis_frame_sizes)],
        "identity_native_full_frame_inputs": [list(item) for item in sorted(set(native_full_frame_inputs))],
        "identity_analysis_full_frame_inputs": [list(item) for item in sorted(set(analysis_full_frame_inputs))],
        "identity_full_frame_inputs": [list(item) for item in full_frame_embed_inputs],
        "no_full_frame_identity_inputs": bool(embed_inputs) and not full_frame_embed_inputs,
        "candidate_boxes": boxes,
        "whole_frame_candidate_boxes": whole_frame_candidates,
        "no_whole_frame_candidate": not whole_frame_candidates,
        "wanted": _safe(list(wanted)),
        "avoided": _safe(list(avoided)),
        "candidate_tag_scopes_present": output_scope_keys,
        "tag_calls_before_direct_audit": [list(item) for item in model.tag_calls],
    }


def _direct_tag_scope_audit(model: RecordingModel, reader: Any, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Call the tagger once on a candidate crop and its complete source frame.

    This keeps the scope evidence explicit even when the primary run has no
    tag conditions.  The returned tag values are diagnostic only; they never
    change the primary candidate ranking.
    """

    if not callable(getattr(model, "tag", None)):
        return {"available": False, "reason": "ModelService.tag is unavailable"}
    if not candidates:
        return {"available": False, "reason": "no candidates"}
    candidate = candidates[0]
    crop_path = Path(str(candidate.get("crop_path", "")))
    if not crop_path.is_file():
        return {"available": False, "reason": "candidate crop is unavailable"}
    try:
        with Image.open(crop_path) as image:
            crop = image.convert("RGB")
        full_frame = _as_rgb(reader.get_frame(int(candidate["pts"]), max_side=None).image)
        subject_tags = model.tag(crop)
        frame_tags = model.tag(full_frame)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        return {"available": False, "reason": f"direct tag audit failed: {error}"}
    crop_size = list(crop.size)
    frame_size = list(full_frame.size)
    return {
        "available": True,
        "candidate_id": candidate.get("id"),
        "subject_input_size": crop_size,
        "frame_input_size": frame_size,
        "distinct_input_sizes": crop_size != frame_size,
        "subject_tags": _safe(subject_tags),
        "frame_tags": _safe(frame_tags),
    }


def _negative_distances(model: RecordingModel, reference: Image.Image, reader: Any, specs: Sequence[tuple[str, float, Sequence[float]]]) -> list[dict[str, Any]]:
    ref_features = np.asarray(model.embed([reference]))
    if ref_features.ndim == 1:
        ref_features = ref_features.reshape(1, -1)
    result: list[dict[str, Any]] = []
    for label, requested_time, raw_box in specs:
        frame = _frame_at(reader, requested_time)
        image = _as_rgb(frame.image)
        box = _parse_box(raw_box, image.width, image.height)
        crop = _crop(image, box)
        features = np.asarray(model.embed([crop]))
        if features.ndim == 1:
            features = features.reshape(1, -1)
        distances = np.asarray(model.distance(ref_features, features)).reshape(-1)
        distance = float(distances[0]) if distances.size else float("nan")
        result.append(
            {
                "label": label,
                "requested_time": requested_time,
                "actual_time": float(frame.time),
                "pts": int(frame.pts),
                "box": box,
                "crop_size": list(crop.size),
                "distance": distance,
            }
        )
    return result


def _compact_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the small stdout/CI summary; full model tags stay in the JSON file."""

    primary = report.get("primary") if isinstance(report.get("primary"), Mapping) else {}
    scope = report.get("scope_pass") if isinstance(report.get("scope_pass"), Mapping) else {}
    stats = primary.get("stats") if isinstance(primary.get("stats"), Mapping) else {}
    scope_stats = scope.get("stats") if isinstance(scope.get("stats"), Mapping) else {}
    negatives = primary.get("negative_embeddings") if isinstance(primary.get("negative_embeddings"), list) else []
    candidates = primary.get("candidates") if isinstance(primary.get("candidates"), list) else []
    results_dir = None
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("preview_path"):
            results_dir = str(Path(str(candidate["preview_path"])).parent)
            break
    return {
        "status": report.get("status"),
        "report_path": report.get("artifacts", {}).get("report_path") if isinstance(report.get("artifacts"), Mapping) else None,
        "candidate_contact_sheet": primary.get("contact_sheet"),
        "candidate_results_dir": results_dir,
        "primary_candidates": primary.get("candidate_count", 0),
        "scope_candidates": scope.get("candidate_count", 0) if scope else 0,
        "native_exports_rgb_equal": primary.get("all_native_exports_equal_rgb"),
        "candidate_times_match_source": primary.get("all_candidate_times_match_source"),
        "negative_distances": [
            {"label": item.get("label"), "distance": item.get("distance")}
            for item in negatives
            if isinstance(item, Mapping)
        ],
        "primary_cache_hits": stats.get("cache_hits"),
        "scope_cache_hits": scope_stats.get("cache_hits"),
        "scope_cache_misses": scope_stats.get("cache_misses"),
        "multi_person_full_frame_rejections": (
            report.get("scope_audit", {}).get("multi_person_full_frame_rejections")
            if isinstance(report.get("scope_audit"), Mapping)
            else None
        ),
        "visual_review_pairs": primary.get("nearby_candidate_pairs", []),
        "warnings": report.get("warnings", []),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--model-service", default=None, help="module:ModelService (default: anime_picker.models:ModelService or env)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=None, help="pipeline cache root (default: output-dir/real-video)")
    parser.add_argument("--reference-time", type=float, default=70.0)
    parser.add_argument("--reference-box", default=None, help="optional x0,y0,x1,y1 in pixels or normalized coordinates")
    parser.add_argument("--start", type=float, default=55.0)
    parser.add_argument("--end", type=float, default=85.0)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--coarse-step", type=float, default=1.0)
    parser.add_argument("--dense-step", type=float, default=0.25)
    parser.add_argument("--near-step", type=float, default=0.08)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-tagged", type=int, default=None)
    parser.add_argument("--wanted", action="append", default=[], help="scope:tag, e.g. subject:smile")
    parser.add_argument("--avoided", action="append", default=[], help="scope:tag, e.g. frame:multiple_people")
    parser.add_argument("--scope-tag", default="smile", help="run a second subject-scope pass using this tag; empty disables it")
    parser.add_argument("--no-scope-pass", action="store_true")
    parser.add_argument("--negative", action="append", default=[], help="label@time:x0,y0,x1,y1; replaces the two fixture negatives")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    video = args.video.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = (args.cache_dir or output_dir / "real-video").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "real-video-verification.json"
    report: dict[str, Any] = {
        "status": "not_run",
        "fixture": {
            "video": str(video),
            "reference_requested_time": args.reference_time,
            "search_start": args.start,
            "search_end": args.end,
            "source_is_local_only": True,
        },
        "artifacts": {"report_path": str(report_path.resolve())},
        "warnings": [],
    }

    if args.model_root is None:
        model_root_text = os.environ.get("ANIME_PICKER_MODEL_ROOT", "").strip()
        args.model_root = Path(model_root_text) if model_root_text else None
    if args.model_root is None:
        report["status"] = "blocked_model_root_missing"
        report["warnings"].append("pass --model-root or set ANIME_PICKER_MODEL_ROOT")
        report_path.write_text(json.dumps(_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(_safe(report), indent=2, ensure_ascii=False))
        return 2
    model_root = args.model_root.resolve()
    if not video.is_file():
        report["status"] = "blocked_video_missing"
        report["warnings"].append(f"video not found: {video}")
        report_path.write_text(json.dumps(_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(_safe(report), indent=2, ensure_ascii=False))
        return 2

    try:
        model_class, model_spec = _load_model_class(args.model_service)
        print(f"[model] loading {model_spec} from {model_root} with require_gpu=True", flush=True)
        model = RecordingModel(model_class(model_root=model_root, require_gpu=True))
        from anime_picker.pipeline import run_search
        from anime_picker.video import VideoReader

        reader = VideoReader(video)
        ref_frame = _frame_at(reader, args.reference_time)
        ref_image = _as_rgb(ref_frame.image)
        reference_box = _select_reference_box(model, ref_frame, args.reference_box)
        reference_crop = _crop(ref_image, reference_box)
        reference_frame_path = output_dir / "reference-frame-native.png"
        reference_crop_path = output_dir / "reference-crop.png"
        reference_export = output_dir / "reference-export-native.png"
        ref_image.save(reference_frame_path, format="PNG")
        reference_crop.save(reference_crop_path, format="PNG")
        exported_reference = reader.export_png(int(ref_frame.pts), reference_export)
        with Image.open(reference_export) as image:
            reference_export_pixels = np.asarray(image.convert("RGB"))
        reference_exact = np.asarray(ref_image)
        reference_export_equal = bool(np.array_equal(reference_exact, reference_export_pixels))
        report["model"] = {"service": model_spec, "model_root": str(model_root), "require_gpu": True}
        report["reference"] = {
            "requested_time": args.reference_time,
            "actual_time": float(ref_frame.time),
            "pts": int(ref_frame.pts),
            "native_size": list(ref_image.size),
            "box_pixels": reference_box,
            "crop_size": list(reference_crop.size),
            "frame_path": str(reference_frame_path.resolve()),
            "crop_path": str(reference_crop_path.resolve()),
            "export_path": str(reference_export.resolve()),
            "export_pixel_equal_rgb": reference_export_equal,
            "export_pts": int(exported_reference.pts),
        }

        wanted = _parse_scope_items(args.wanted)
        avoided = _parse_scope_items(args.avoided)
        settings = {
            "start": args.start,
            "end": args.end,
            "candidate_count": max(1, args.candidate_count),
            "coarse_step": args.coarse_step,
            "dense_step": args.dense_step,
            "near_step": args.near_step,
            "rescue": False,
            "max_samples": max(1, args.max_samples),
            "max_tagged": (
                max(1, args.max_tagged)
                if args.max_tagged is not None
                else max(24, args.candidate_count * 8)
            ),
            "_feature_cache_dir": str((cache_dir / "features").resolve()),
        }
        primary_cache = cache_dir / "primary"
        primary = run_search(
            str(video),
            [reference_crop],
            wanted,
            avoided,
            settings,
            {"detector": model, "identity": model, "tagger": model},
            primary_cache,
            progress=_progress_printer(),
            cancelled=lambda: False,
        )
        candidates = primary.get("candidates", []) if isinstance(primary, Mapping) else []
        candidate_audits = [_verify_native_export(reader, candidate, output_dir, index) for index, candidate in enumerate(candidates)]
        duplicates = _hash_duplicates(candidates)
        nearby_pairs = _nearby_candidate_pairs(candidates)
        contact_path = _contact_sheet(candidates, output_dir / "candidates-contact.jpg")
        scope_audit = _scope_audit(model, reader, candidates, wanted, avoided)
        scope_audit["direct_tag_scope"] = _direct_tag_scope_audit(model, reader, candidates)
        primary_rejected = primary.get("stats", {}).get("rejected", {}) if isinstance(primary, Mapping) else {}
        scope_audit["multi_person_full_frame_rejections"] = int(primary_rejected.get("multi_person_full_frame", 0) or 0)
        direct_tag_scope = scope_audit["direct_tag_scope"]
        identity_scope_ok = bool(scope_audit.get("no_full_frame_identity_inputs"))
        box_scope_ok = bool(scope_audit.get("no_whole_frame_candidate"))
        tag_scope_ok = not direct_tag_scope.get("available") or bool(direct_tag_scope.get("distinct_input_sizes"))
        scope_audit["identity_scope_ok"] = identity_scope_ok
        scope_audit["box_scope_ok"] = box_scope_ok
        scope_audit["tag_scope_ok"] = tag_scope_ok
        negative_specs = [_parse_negative(item) for item in args.negative] if args.negative else list(DEFAULT_NEGATIVES)
        negatives = _negative_distances(model, reference_crop, reader, negative_specs)
        positive_distances = [float(item.get("distance")) for item in candidates if item.get("distance") is not None]
        best_positive = min(positive_distances) if positive_distances else float("nan")
        negative_ordering = bool(negatives) and math.isfinite(best_positive) and all(
            math.isfinite(float(item.get("distance", float("nan")))) and float(item["distance"]) > best_positive
            for item in negatives
        )
        primary_report = {
            "settings": settings,
            "wanted": wanted,
            "avoided": avoided,
            "stats": primary.get("stats", {}) if isinstance(primary, Mapping) else {},
            "warnings": primary.get("warnings", []) if isinstance(primary, Mapping) else ["pipeline did not return a mapping"],
            "candidate_count": len(candidates),
            "candidates": candidates,
            "native_export_checks": candidate_audits,
            "all_native_exports_equal_rgb": bool(candidate_audits) and all(item["pixel_equal_rgb"] for item in candidate_audits),
            "all_candidate_times_match_source": bool(candidate_audits) and all(item["time_matches"] for item in candidate_audits),
            "duplicates_by_pixel_hash": duplicates,
            "nearby_candidate_pairs": nearby_pairs,
            "visual_duplicate_review_required": bool(nearby_pairs),
            "pixel_hash_duplicate_check": "heuristic_only; inspect contact sheet",
            "contact_sheet": contact_path,
            "negative_embeddings": negatives,
            "best_positive_distance": best_positive if math.isfinite(best_positive) else None,
            "positive_beats_all_fixture_negatives": negative_ordering,
        }

        scope_report: dict[str, Any] | None = None
        if not args.no_scope_pass and args.scope_tag.strip():
            scope_wanted = [{"scope": "subject", "tag": args.scope_tag.strip()}]
            print(f"[scope] running subject:{args.scope_tag.strip()} pass with the same cache", flush=True)
            scoped = run_search(
                str(video),
                [reference_crop],
                scope_wanted,
                [],
                settings,
                {"detector": model, "identity": model, "tagger": model},
                primary_cache,
                progress=_progress_printer(),
                cancelled=lambda: False,
            )
            scope_candidates = scoped.get("candidates", []) if isinstance(scoped, Mapping) else []
            scope_report = {
                "wanted": scope_wanted,
                "candidate_count": len(scope_candidates),
                "stats": scoped.get("stats", {}) if isinstance(scoped, Mapping) else {},
                "warnings": scoped.get("warnings", []) if isinstance(scoped, Mapping) else [],
                "candidates": scope_candidates,
                "candidate_tag_scopes_present": all(
                    isinstance(item.get("tags"), Mapping)
                    and "subject" in item["tags"]
                    and "frame" in item["tags"]
                    for item in scope_candidates
                ) if scope_candidates else None,
            }

        report.update(
            {
                "status": "pass" if (
                    candidates
                    and reference_export_equal
                    and all(item["pixel_equal_rgb"] and item["time_matches"] for item in candidate_audits)
                    and identity_scope_ok
                    and box_scope_ok
                    and tag_scope_ok
                ) else "fail",
                "video_info": _safe(getattr(reader, "info", None).__dict__ if hasattr(getattr(reader, "info", None), "__dict__") else getattr(reader, "info", None)),
                "cache_dir": str(cache_dir),
                "primary": primary_report,
                "scope_pass": scope_report,
                "scope_audit": scope_audit,
                "model_calls": {
                    "detect_count": len(model.detect_calls),
                    "embed_batch_shapes": model.embed_calls,
                    "distance_shapes": _safe(model.distance_calls),
                    "tag_input_shapes": model.tag_calls,
                },
                "limitations": [
                    "This is one local fixture and a bounded 30-second interval; it does not establish universal precision or recall.",
                    "The fixture negative boxes are visual QA aids and should be reviewed if the source cut changes.",
                    "Candidate contact sheets require a human visual check for duplicates and wrong-character leakage.",
                ],
                "artifacts": {
                    "report_path": str(report_path.resolve()),
                    "candidate_contact_sheet": contact_path,
                    "reference_frame": str(reference_frame_path.resolve()),
                    "reference_crop": str(reference_crop_path.resolve()),
                },
            }
        )
        if duplicates:
            report["warnings"].append(f"{len(duplicates)} visually duplicate candidate pair(s) need review")
        if nearby_pairs:
            report["warnings"].append(f"{len(nearby_pairs)} close-in-time candidate pair(s) require contact-sheet review")
        if not scope_report:
            report["warnings"].append("subject-scope pass was disabled")
    except Exception as error:  # noqa: BLE001 - preserve a machine-readable QA report
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"

    report_path.write_text(json.dumps(_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    if _TRACE_TIMEOUT > 0:
        faulthandler.cancel_dump_traceback_later()
    print(json.dumps(_safe(_compact_summary(report)), indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
