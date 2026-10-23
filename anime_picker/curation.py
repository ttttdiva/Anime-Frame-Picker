"""Character-presence curation and conservative near-duplicate grouping.

This module consumes already-tagged/detected image records.  It does not load
models, infer identity, limit the number of characters, or mutate input images.
The duplicate check deliberately combines whole-frame and local evidence so a
small but meaningful face/expression change is not hidden by a similar scene.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .scoring import dhash, hamming_distance, phash


_PRESENCE_TAGS = frozenset(
    {
        "solo",
        "person",
        "people",
        "character",
        "portrait",
        "multiple_girls",
        "multiple_boys",
        "multiple_people",
        "multiple_persons",
    }
)
_EXPRESSION_TAGS = frozenset(
    {
        "smile",
        "smiling",
        "grin",
        "laughing",
        "happy",
        "joyful",
        "giggling",
        "playful",
        "cute",
        "heart_eyes",
        "frown",
        "frowning",
        "sad",
        "angry",
        "surprised",
        "embarrassed",
        "blush",
        "wink",
        "closed_eyes",
        "open_eyes",
        "open_mouth",
        "closed_mouth",
        "teeth",
    }
)
_CUTE_EXPRESSION_POSITIVE_WEIGHTS = {
    "smile": 1.0,
    "smiling": 1.0,
    "grin": 0.95,
    "laughing": 0.98,
    "happy": 1.0,
    "joyful": 0.98,
    "giggling": 0.95,
    "blush": 0.88,
    "wink": 0.82,
    "playful": 0.90,
    "cute": 0.90,
    "embarrassed": 0.78,
    "heart_eyes": 0.90,
    "open_mouth": 0.30,
    "surprised": 0.30,
    "closed_eyes": 0.22,
    "teeth": 0.15,
}
_CUTE_EXPRESSION_NEGATIVE_WEIGHTS = {
    "frown": 1.0,
    "frowning": 1.0,
    "sad": 1.0,
    "angry": 1.0,
}
_FACE_VISIBILITY_TAG_WEIGHTS = {
    "portrait": 0.95,
    "close_up": 0.95,
    "closeup": 0.95,
    "headshot": 0.98,
    "face": 0.90,
    "face_visible": 0.98,
    "face_focus": 0.98,
    "upper_body": 0.78,
    "bust": 0.78,
    "cowboy_shot": 0.62,
    "looking_at_viewer": 0.76,
    "facing_viewer": 0.76,
    "looking_straight_at_viewer": 0.80,
}
_FACE_OBSCURATION_TAG_WEIGHTS = {
    "from_behind": 0.85,
    "back_view": 0.85,
    "face_obscured": 0.95,
    "face_hidden": 0.95,
    "covered_face": 0.90,
    "silhouette": 0.85,
    "looking_away": 0.35,
}
_COMPOSITION_TAG_WEIGHTS = {
    "centered": 0.95,
    "center_focus": 0.95,
    "subject_center": 0.95,
    "center": 0.86,
    "centered_composition": 0.95,
    "symmetrical": 0.82,
    "balanced_composition": 0.88,
    "rule_of_thirds": 0.80,
    "portrait": 0.82,
    "close_up": 0.82,
    "closeup": 0.82,
    "upper_body": 0.76,
    "bust": 0.76,
    "cowboy_shot": 0.60,
    "full_body": 0.46,
    "wide_shot": 0.26,
    "long_shot": 0.24,
    "far_away": 0.20,
    "looking_at_viewer": 0.72,
    "facing_viewer": 0.72,
    "looking_straight_at_viewer": 0.76,
    "off_center": 0.40,
}
_TAG_NORMALIZE_RE = re.compile(r"[\s-]+")
_SAFE_IMAGE_RESAMPLE = Image.Resampling.LANCZOS
_GLOBAL_PHASH_CANDIDATE_DISTANCE = 22
_GLOBAL_DHASH_CANDIDATE_DISTANCE = 26
_CV2_THREAD_CONFIG_LOCK = threading.Lock()
_CV2_SINGLE_THREADED = False


def curate_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    presence_threshold: float = 0.17,
    time_window: float = 4.0,
    minimum_cut_quality: float = 0.0,
    include_full_tags: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Keep character-present frames and collapse only strongly evidenced twins.

    Each record requires a unique stable ``id`` and may carry ``pts``, ``time``,
    ``image`` (a Pillow image or path), ``tags`` (the nested mapping returned by
    ``ModelService.tag_many``), ``detections``, and a numeric ``quality`` or
    ``source_score``.  General PixAI presence labels use the model-card 0.17
    threshold.  A detector-only fallback requires a high-confidence, sizeable
    person/head box; tiny crowd/background boxes do not count.

    The report has ``kept``, ``excluded``, ``groups``, and ``counts``.  Every
    input belongs to exactly one visual group.  A near-duplicate exclusion
    names its selected representative via ``duplicate_of`` and the
    ``near_duplicate_lower_rank`` exclusion reason; a frame without evidence is
    reported as ``no_character``.  No result-count cap is applied.
    """

    try:
        threshold = float(presence_threshold)
        window = float(time_window)
        quality_floor = float(minimum_cut_quality)
    except (TypeError, ValueError) as error:
        raise ValueError("presence_threshold, time_window, and minimum_cut_quality must be numeric") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("presence_threshold must be in [0, 1]")
    if not math.isfinite(window) or window < 0:
        raise ValueError("time_window must be finite and nonnegative")
    if not math.isfinite(quality_floor) or not 0.0 <= quality_floor <= 1.0:
        raise ValueError("minimum_cut_quality must be in [0, 1]")

    prepared: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise TypeError("each candidate record must be a mapping")
        if raw.get("id") is None or not str(raw.get("id")):
            raise ValueError("each candidate record requires a nonempty stable id")
        candidate_id = str(raw["id"])
        if candidate_id in seen_ids:
            raise ValueError(f"candidate ids must be unique: {candidate_id}")
        seen_ids.add(candidate_id)
        candidate_time = _finite_number(raw.get("time"))
        if candidate_time is None:
            raise ValueError(f"candidate {candidate_id} requires a finite time in seconds")
        image = _load_rgb(raw.get("image"))
        tags = _normalise_tags(raw.get("tags"))
        detections = _normalise_detections(raw.get("detections"), image.size)
        features = _image_features(image, detections)
        presence = _character_presence(tags, detections, threshold, features)
        prepared.append(
            {
                "record": dict(raw),
                "id": candidate_id,
                "pts": raw.get("pts"),
                "time": candidate_time,
                "tags": tags,
                "detections": detections,
                "presence": presence,
                "features": features,
                "quality_components": _quality_components(raw, tags, detections, presence),
                "rank_score": 0.0,
            }
        )
        prepared[-1]["rank_score"] = _combine_quality(prepared[-1]["quality_components"])

    total = len(prepared)
    if progress is not None:
        progress(0, total)
    visual_candidates = [item for item in prepared if item["presence"]["present"]]
    no_character_candidates = [item for item in prepared if not item["presence"]["present"]]
    # No-character records are excluded regardless of visual grouping. Keep
    # them as explicit singleton groups in the report, while spending global
    # near-duplicate comparisons only on candidates eligible for export.
    groups = _visual_groups(visual_candidates, window, progress=progress, progress_total=total)
    groups.extend([[item] for item in no_character_candidates])
    if progress is not None:
        completed = len(visual_candidates)
        for _ in no_character_candidates:
            completed += 1
            progress(completed, total)
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    group_reports: list[dict[str, Any]] = []

    for members in groups:
        group_id = _stable_group_id(member["id"] for member in members)
        positive = [member for member in members if member["presence"]["present"]]
        ranked = sorted(
            positive or members,
            key=lambda member: (-member["rank_score"], _time_sort(member), member["id"]),
        )
        representative = ranked[0] if positive and ranked[0]["rank_score"] >= quality_floor else None
        member_ids = [member["id"] for member in sorted(members, key=_time_sort)]
        pair_metrics = list(members[0].get("_group_pair_evidence", []))
        pair_checks = dict(
            members[0].get(
                "_group_pair_checks",
                {"evaluated": 0, "confirmed": 0, "admitted_frames": 0, "full_proof_count": 0},
            )
        )
        pair_checks.setdefault("admitted_frames", 0)
        pair_checks.setdefault("full_proof_count", 0)
        pair_checks["one_full_proof_per_admitted_frame"] = (
            pair_checks["full_proof_count"] == pair_checks["admitted_frames"]
        )

        for member in members:
            enriched = _public_candidate(
                member, group_id, include_full_tags=include_full_tags
            )
            if representative is None:
                if member["presence"]["present"]:
                    enriched["exclusion_reason"] = "below_minimum_cut_quality"
                else:
                    enriched["exclusion_reason"] = "no_character"
                excluded.append(enriched)
            elif member["id"] == representative["id"]:
                kept.append(enriched)
            else:
                enriched["exclusion_reason"] = "near_duplicate_lower_rank"
                enriched["duplicate_of"] = representative["id"]
                excluded.append(enriched)

        group_reports.append(
            {
                "group_id": group_id,
                "representative_id": representative["id"] if representative else None,
                "members": member_ids,
                "pair_checks": pair_checks,
                "pair_evidence": pair_metrics,
            }
        )

    kept.sort(key=_public_time_sort)
    excluded.sort(key=_public_time_sort)
    group_reports.sort(key=lambda item: item["members"])
    duplicate_groups = sum(1 for group in group_reports if len(group["members"]) > 1)
    return {
        "kept": kept,
        "excluded": excluded,
        "groups": group_reports,
        "summary": {
            "input": len(prepared),
            "character_present": sum(1 for item in prepared if item["presence"]["present"]),
            "character_groups": sum(
                1
                for group in groups
                if any(item["presence"]["present"] for item in group)
            ),
            "near_duplicate_groups": duplicate_groups,
            "kept": len(kept),
            "excluded": len(excluded),
            "excluded_no_character": sum(
                1 for item in excluded if item["exclusion_reason"] == "no_character"
            ),
            "excluded_near_duplicate": sum(
                1 for item in excluded if item["exclusion_reason"] == "near_duplicate_lower_rank"
            ),
            "excluded_below_minimum_cut_quality": sum(
                1 for item in excluded if item["exclusion_reason"] == "below_minimum_cut_quality"
            ),
        },
        "settings": {
            "presence_threshold": threshold,
            "time_window_seconds": window,
            "time_window_applied_to_visual_duplicates": False,
            "visual_grouping_scope": "character-present candidates only; no-character records remain excluded individually",
            "registration_solver": "single-threaded-full-resolution-ECC+pixel-verification",
            "minimum_cut_quality": quality_floor,
            "global_candidate_hash_radius": {
                "phash": _GLOBAL_PHASH_CANDIDATE_DISTANCE,
                "dhash": _GLOBAL_DHASH_CANDIDATE_DISTANCE,
                "operator": "phash_distance <= radius AND dhash_distance <= radius",
            },
            "global_candidate_hash_nomination": "phash_distance <= 22 AND dhash_distance <= 26",
            "duplicate_method": "global-phash-dhash-candidates+complete-link+whole-frame-and-local-hash+pixel-ssim+translation-registration+face-expression-protection",
        },
    }


def _load_rgb(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB").copy()
    if isinstance(value, (str, Path)):
        with Image.open(value) as opened:
            return opened.convert("RGB").copy()
    raise ValueError("candidate image must be a Pillow image or a path")


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalise_tags(raw: Any) -> dict[str, dict[str, float]]:
    if not isinstance(raw, Mapping):
        return {}
    groups: dict[str, dict[str, float]] = {}
    nested = any(isinstance(value, Mapping) for value in raw.values())
    source = raw if nested else {"general": raw}
    for category, values in source.items():
        if not isinstance(values, Mapping):
            continue
        cleaned: dict[str, float] = {}
        for tag, value in values.items():
            score = _finite_number(value)
            if score is None:
                continue
            normalized = _TAG_NORMALIZE_RE.sub("_", str(tag).strip().lower()).strip("_")
            if normalized:
                cleaned[normalized] = max(0.0, min(1.0, score))
        groups[str(category).strip().lower()] = cleaned
    return groups


def _normalise_detections(raw: Any, size: tuple[int, int]) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    width, height = size
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        box = item.get("box", item.get("bbox", item.get("xyxy")))
        if not isinstance(box, Sequence) or len(box) != 4:
            continue
        values = [_finite_number(part) for part in box]
        if any(value is None for value in values):
            continue
        x0, y0, x1, y1 = (float(value) for value in values)
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
            x0, x1 = x0 * width, x1 * width
            y0, y1 = y0 * height, y1 * height
        x0, x1 = sorted((max(0.0, min(float(width), x0)), max(0.0, min(float(width), x1))))
        y0, y1 = sorted((max(0.0, min(float(height), y0)), max(0.0, min(float(height), y1))))
        if x1 <= x0 or y1 <= y0:
            continue
        score = _finite_number(item.get("score", item.get("confidence", item.get("conf", 1.0))))
        kind = str(item.get("kind", item.get("label", "person"))).strip().lower()
        result.append(
            {
                "box": [x0 / width, y0 / height, x1 / width, y1 / height],
                "score": max(0.0, min(1.0, score if score is not None else 0.0)),
                "kind": "head" if kind in {"head", "face"} else "person",
            }
        )
    return result


def _general_scores(tags: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    general = tags.get("general", {})
    return {str(key).lower(): float(value) for key, value in general.items()}


def _is_person_tag(tag: str) -> bool:
    if tag in _PRESENCE_TAGS:
        return True
    if re.fullmatch(r"\d+(?:girls?|boys?|persons?|people)", tag):
        return True
    return False


def _character_presence(
    tags: Mapping[str, Mapping[str, float]],
    detections: Sequence[Mapping[str, Any]],
    threshold: float,
    features: Mapping[str, Any],
) -> dict[str, Any]:
    general = _general_scores(tags)
    tag_evidence = [
        {"tag": tag, "score": score}
        for tag, score in sorted(general.items(), key=lambda item: (-item[1], item[0]))
        if _is_person_tag(tag) and score >= threshold
    ]
    detector_evidence: list[dict[str, Any]] = []
    for detection in detections:
        x0, y0, x1, y1 = detection["box"]
        area = (x1 - x0) * (y1 - y0)
        if detection["kind"] == "head":
            strong_enough = detection["score"] >= 0.78 and area >= 0.004
        else:
            strong_enough = detection["score"] >= 0.72 and area >= 0.025
        if strong_enough:
            detector_evidence.append(
                {
                    "kind": detection["kind"],
                    "score": detection["score"],
                    "area_fraction": round(area, 6),
                    "box": list(detection["box"]),
                }
            )
    content_gate = bool(
        float(features.get("luminance_std", 0.0)) >= 0.025
        and float(features.get("visual_entropy", 0.0)) >= 0.10
        and int(features.get("informative_tile_count", 0)) >= 2
        and float(features.get("visible_pixel_fraction", 0.0)) >= 0.05
    )
    present = bool(content_gate and (tag_evidence or detector_evidence))
    confidence = max(
        [item["score"] for item in tag_evidence]
        + [item["score"] for item in detector_evidence]
        + [0.0]
    )
    return {
        "present": present,
        "confidence": confidence,
        "method": "pixai_general_tag" if tag_evidence else "strong_large_detector" if detector_evidence else None,
        "tag_evidence": tag_evidence,
        "detector_evidence": detector_evidence,
        "content_gate": {
            "passed": content_gate,
            "luminance_std": round(float(features.get("luminance_std", 0.0)), 6),
            "visual_entropy": round(float(features.get("visual_entropy", 0.0)), 6),
            "informative_tile_count": int(features.get("informative_tile_count", 0)),
            "visible_pixel_fraction": round(float(features.get("visible_pixel_fraction", 0.0)), 6),
        },
    }


def _quality_parts(raw: Mapping[str, Any]) -> tuple[float, float]:
    quality = raw.get("quality")
    quality_map = quality if isinstance(quality, Mapping) else {}
    numeric_quality = _finite_number(quality)
    raw_technical = raw.get(
        "technical_quality",
        quality_map.get("technical", quality_map.get("quality_score", raw.get("quality_score", numeric_quality))),
    )
    raw_source = raw.get("source_score", raw.get("score", quality_map.get("source_score")))
    if raw_source is None:
        raw_source = numeric_quality
    technical = _finite_number(raw_technical)
    source = _finite_number(raw_source)
    sharpness = _finite_number(raw.get("sharpness", quality_map.get("sharpness")))
    if technical is None:
        technical = max(0.0, min(1.0, sharpness / (sharpness + 100.0))) if sharpness and sharpness > 0 else 0.5
    if source is None:
        source = 0.5
    return max(0.0, min(1.0, technical)), max(0.0, min(1.0, source))


def _quality_components(
    raw: Mapping[str, Any],
    tags: Mapping[str, Mapping[str, float]],
    detections: Sequence[Mapping[str, Any]],
    presence: Mapping[str, Any],
) -> dict[str, float]:
    technical, source = _quality_parts(raw)
    general = _general_scores(tags)
    positive_expression = max(
        (general.get(tag, 0.0) * weight for tag, weight in _CUTE_EXPRESSION_POSITIVE_WEIGHTS.items()),
        default=0.0,
    )
    negative_expression = max(
        (general.get(tag, 0.0) * weight for tag, weight in _CUTE_EXPRESSION_NEGATIVE_WEIGHTS.items()),
        default=0.0,
    )
    # A modest neutral baseline lets negative expressions lower the ranking
    # below an otherwise equivalent neutral frame while positive cues still
    # lift attractive expressions clearly above it.
    expression = max(0.0, min(1.0, max(0.18, positive_expression) - 0.75 * negative_expression))
    heads = [item for item in detections if item["kind"] == "head" and item["score"] >= 0.3]
    people = [item for item in detections if item["kind"] == "person" and item["score"] >= 0.3]
    face_visibility = 0.25
    if heads:
        face_visibility = max(
            min(1.0, item["score"] * (0.45 + min(1.0, math.sqrt(_box_area(item["box"]) / 0.06)) * 0.55))
            for item in heads
        )
    elif people:
        face_visibility = max(min(0.72, item["score"] * 0.65) for item in people)
    else:
        face_visibility = _tag_face_visibility(general)
    composition = _composition_score(heads or people, general)
    presence_score = float(presence["confidence"])
    overlay_score = max(
        general.get("text_focus", 0.0),
        general.get("credits", 0.0),
        general.get("foreground_text", 0.0) * 0.85,
        general.get("english_text", 0.0) * 0.45,
        general.get("japanese_text", 0.0) * 0.45,
    )
    return {
        "technical": round(technical, 6),
        "source": round(source, 6),
        "face_visibility": round(face_visibility, 6),
        "composition": round(composition, 6),
        "expression": round(expression, 6),
        "presence": round(presence_score, 6),
        "text_cleanliness": round(1.0 - max(0.0, min(1.0, overlay_score)), 6),
    }


def _combine_quality(components: Mapping[str, float]) -> float:
    return round(
        0.22 * components["technical"]
        + 0.22 * components["face_visibility"]
        + 0.18 * components["composition"]
        + 0.28 * components["expression"]
        + 0.05 * components["presence"]
        + 0.05 * components["text_cleanliness"],
        6,
    )


def _tagged_max(general: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return max((general.get(tag, 0.0) * weight for tag, weight in weights.items()), default=0.0)


def _tag_face_visibility(general: Mapping[str, float]) -> float:
    visibility = max(0.25, _tagged_max(general, _FACE_VISIBILITY_TAG_WEIGHTS))
    obscuration = _tagged_max(general, _FACE_OBSCURATION_TAG_WEIGHTS)
    return max(0.08, visibility * (1.0 - 0.78 * obscuration))


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _composition_score(
    boxes: Sequence[Mapping[str, Any]], general: Mapping[str, float] | None = None
) -> float:
    if not boxes:
        return max(0.25, _tagged_max(general or {}, _COMPOSITION_TAG_WEIGHTS))
    scores: list[float] = []
    for item in boxes:
        x0, y0, x1, y1 = item["box"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        center = max(0.0, 1.0 - math.sqrt((cx - 0.5) ** 2 + (cy - 0.48) ** 2) / 0.72)
        area = _box_area(item["box"])
        size_fit = max(0.0, 1.0 - abs(math.log(max(area, 1e-5) / 0.10)) / 4.0)
        scores.append(0.55 * center + 0.45 * size_fit)
    return sum(scores) / len(scores)


def _image_features(image: Image.Image, detections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gray = image.convert("L")
    phash_value = phash(gray)
    # phash/dhash use different transforms; keep their signatures separately.
    dhash_value = dhash(gray)
    grid_hashes: list[str] = []
    cols, rows = 4, 3
    for row in range(rows):
        for column in range(cols):
            left = round(image.width * column / cols)
            right = round(image.width * (column + 1) / cols)
            top = round(image.height * row / rows)
            bottom = round(image.height * (row + 1) / rows)
            grid_hashes.append(phash(gray.crop((left, top, right, bottom))))
    resized = image.resize((64, 64), _SAFE_IMAGE_RESAMPLE)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    luminance = np.asarray(resized.convert("L"), dtype=np.float32) / 255.0
    gray_histogram = np.histogram(luminance, bins=32, range=(0.0, 1.0))[0].astype(np.float64)
    gray_probabilities = gray_histogram / max(1.0, float(gray_histogram.sum()))
    nonzero_probabilities = gray_probabilities[gray_probabilities > 0]
    visual_entropy = float(-np.sum(nonzero_probabilities * np.log2(nonzero_probabilities)))
    informative_tile_count = 0
    for row in range(3):
        for column in range(4):
            tile = luminance[
                round(64 * row / 3) : round(64 * (row + 1) / 3),
                round(64 * column / 4) : round(64 * (column + 1) / 4),
            ]
            informative_tile_count += int(float(tile.std()) >= 0.025)
    visible_pixel_fraction = float(np.mean(luminance >= 0.08))
    alignment_gray = np.asarray(gray.resize((256, 144), _SAFE_IMAGE_RESAMPLE), dtype=np.uint8)
    alignment_rgb = np.asarray(image.resize((128, 72), _SAFE_IMAGE_RESAMPLE), dtype=np.uint8)
    faces = [item for item in detections if item["kind"] == "head"]
    face_crops: list[dict[str, Any]] = []
    for item in faces:
        x0, y0, x1, y1 = item["box"]
        crop = image.crop(
            (
                max(0, int(x0 * image.width)),
                max(0, int(y0 * image.height)),
                min(image.width, max(1, int(math.ceil(x1 * image.width)))),
                min(image.height, max(1, int(math.ceil(y1 * image.height)))),
            )
        ).resize((48, 48), _SAFE_IMAGE_RESAMPLE)
        face_crops.append(
            {
                "box": item["box"],
                "score": item["score"],
                "rgb": np.asarray(crop, dtype=np.float32) / 255.0,
            }
        )
    return {
        "phash": phash_value,
        "dhash": dhash_value,
        "grid_hashes": grid_hashes,
        "rgb": rgb,
        "gray": luminance,
        "luminance_std": float(luminance.std()),
        "visual_entropy": visual_entropy,
        "informative_tile_count": informative_tile_count,
        "visible_pixel_fraction": visible_pixel_fraction,
        "alignment_gray": alignment_gray,
        "alignment_rgb": alignment_rgb,
        "faces": face_crops,
        "person_count": sum(1 for item in detections if item["kind"] == "person" and item["score"] >= 0.45 and _box_area(item["box"]) >= 0.02),
        "head_count": sum(1 for item in detections if item["kind"] == "head" and item["score"] >= 0.45 and _box_area(item["box"]) >= 0.002),
    }


def _expression_evidence(tags: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {tag: score for tag, score in _general_scores(tags).items() if tag in _EXPRESSION_TAGS}


def _expression_change(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _tag_expression_change(left, right) or _raw_face_crop_change(left, right)


def _tag_expression_change(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_tags = _expression_evidence(left["tags"])
    right_tags = _expression_evidence(right["tags"])
    # Localized expression pairs use a lower delta than broad tag changes, but
    # require both credible confidence and evidence moving in opposite
    # directions. This catches a small mouth/eye transition without treating a
    # lone noisy score fluctuation as a different expression.
    opposed_states = (
        ("open_mouth", "closed_mouth", 0.18, 0.40, 0.08),
        ("open_eyes", "closed_eyes", 0.18, 0.40, 0.08),
        ("smile", "frown", 0.22, 0.45, 0.10),
        ("smiling", "frowning", 0.22, 0.45, 0.10),
        ("happy", "sad", 0.22, 0.45, 0.10),
    )
    for first, second, contrast_threshold, confidence_floor, directional_floor in opposed_states:
        if _opposed_tag_transition(
            left_tags,
            right_tags,
            first,
            second,
            contrast_threshold=contrast_threshold,
            confidence_floor=confidence_floor,
            directional_floor=directional_floor,
        ):
            return True
    wink_delta = abs(left_tags.get("wink", 0.0) - right_tags.get("wink", 0.0))
    if max(left_tags.get("wink", 0.0), right_tags.get("wink", 0.0)) >= 0.55 and wink_delta >= 0.24:
        return True
    # Other expression labels remain protected for large, high-confidence
    # shifts. Paired localized tags above use more specific evidence.
    for tag in _EXPRESSION_TAGS:
        delta = abs(left_tags.get(tag, 0.0) - right_tags.get(tag, 0.0))
        if delta >= 0.35 and max(left_tags.get(tag, 0.0), right_tags.get(tag, 0.0)) >= 0.65:
            return True
    return False


def _raw_face_crop_change(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    try:
        left_faces = left["features"]["faces"]
        right_faces = right["features"]["faces"]
    except (KeyError, TypeError):
        return False
    return _face_crop_change(left_faces, right_faces)


def _opposed_tag_transition(
    left: Mapping[str, float],
    right: Mapping[str, float],
    first: str,
    second: str,
    *,
    contrast_threshold: float,
    confidence_floor: float,
    directional_floor: float,
) -> bool:
    left_first, right_first = left.get(first, 0.0), right.get(first, 0.0)
    left_second, right_second = left.get(second, 0.0), right.get(second, 0.0)
    first_delta = right_first - left_first
    second_delta = right_second - left_second
    contrast_delta = abs((left_first - left_second) - (right_first - right_second))
    peak_confidence = max(left_first, right_first, left_second, right_second)
    opposite_directions = first_delta * second_delta < 0
    both_move = min(abs(first_delta), abs(second_delta)) >= directional_floor
    return bool(
        peak_confidence >= confidence_floor
        and contrast_delta >= contrast_threshold
        and opposite_directions
        and both_move
    )


def _face_crop_change(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> bool:
    if not left or not right:
        return False
    best = 1.0
    for first in left:
        for second in right:
            try:
                first_box = np.asarray(first["box"], dtype=np.float64)
                second_box = np.asarray(second["box"], dtype=np.float64)
                first_rgb = np.asarray(first["rgb"], dtype=np.float32)
                second_rgb = np.asarray(second["rgb"], dtype=np.float32)
                if (
                    first_box.shape != (4,)
                    or second_box.shape != (4,)
                    or not np.isfinite(first_box).all()
                    or not np.isfinite(second_box).all()
                    or first_box[2] <= first_box[0]
                    or first_box[3] <= first_box[1]
                    or second_box[2] <= second_box[0]
                    or second_box[3] <= second_box[1]
                    or first_rgb.shape != (48, 48, 3)
                    or second_rgb.shape != (48, 48, 3)
                    or not np.isfinite(first_rgb).all()
                    or not np.isfinite(second_rgb).all()
                    or np.min(first_rgb) < 0.0
                    or np.max(first_rgb) > 1.0
                    or np.min(second_rgb) < 0.0
                    or np.max(second_rgb) > 1.0
                ):
                    return True
                # Raw-crop comparison intentionally keeps the existing IoU
                # and MAE thresholds. Registered face evidence handles crop
                # coordinates after ECC, rather than relaxing this test.
                if _box_iou(first_box, second_box) < 0.45:
                    continue
                difference = float(np.mean(np.abs(first_rgb - second_rgb)))
            except (KeyError, TypeError, ValueError, OverflowError):
                return True
            best = min(best, difference)
    return best > 0.065


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix0, iy0 = max(left[0], right[0]), max(left[1], right[1])
    ix1, iy1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = _box_area(left) + _box_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _tag_person_count(tags: Mapping[str, Mapping[str, float]]) -> int | None:
    general = _general_scores(tags)
    for tag, score in general.items():
        match = re.fullmatch(r"(\d+)(?:girls?|boys?|persons?|people)", tag)
        if match and score >= 0.45:
            return int(match.group(1))
    if any(general.get(tag, 0.0) >= 0.45 for tag in ("multiple_girls", "multiple_boys", "multiple_people", "multiple_persons")):
        return 2
    return None


def _material_people_change(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_features, right_features = left["features"], right["features"]
    for key in ("person_count", "head_count"):
        if left_features[key] != right_features[key] and (left_features[key] or right_features[key]):
            return True
    left_count = _tag_person_count(left["tags"])
    right_count = _tag_person_count(right["tags"])
    return left_count is not None and right_count is not None and left_count != right_count


def _similarity_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    first, second = left["features"], right["features"]
    local_distances = [hamming_distance(a, b) for a, b in zip(first["grid_hashes"], second["grid_hashes"])]
    rgb_mae = float(np.mean(np.abs(first["rgb"] - second["rgb"])))
    gray_a, gray_b = first["gray"], second["gray"]
    mean_a, mean_b = float(gray_a.mean()), float(gray_b.mean())
    var_a, var_b = float(gray_a.var()), float(gray_b.var())
    covariance = float(((gray_a - mean_a) * (gray_b - mean_b)).mean())
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mean_a * mean_b + c1) * (2 * covariance + c2)) / (
        (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
    )
    tag_expression_change = _tag_expression_change(left, right)
    raw_face_change = _raw_face_crop_change(left, right)
    return {
        "phash_distance": hamming_distance(first["phash"], second["phash"]),
        "dhash_distance": hamming_distance(first["dhash"], second["dhash"]),
        "grid_local_hash_max_distance": max(local_distances, default=0),
        "grid_local_hash_mean_distance": round(float(np.mean(local_distances)) if local_distances else 0.0, 6),
        "resized_rgb_mean_absolute_difference": round(rgb_mae, 8),
        "global_gray_ssim": round(float(ssim), 8),
        "expression_change": tag_expression_change or raw_face_change,
        "tag_expression_change": tag_expression_change,
        "raw_face_change": raw_face_change,
        "material_people_change": _material_people_change(left, right),
        "alignment_checked": False,
        "alignment_correlation_before": None,
        "alignment_correlation": None,
        "alignment_gain": None,
        "alignment_mae": None,
        "alignment_rgb_mae": None,
        "alignment_translation_fraction": None,
        "alignment_dx_pixels": None,
        "alignment_dy_pixels": None,
        "aligned_face_check_performed": False,
        "all_face_matches_verified": False,
        "face_matching_count": 0,
        "face_match_ious": [],
        "minimum_face_support": None,
        "maximum_aligned_face_mae": None,
        "face_check_failure": None,
    }


def _correlation_crop_is_defined(crop: np.ndarray, reference_shape: tuple[int, ...]) -> bool:
    """Report whether a crop can produce a defined Pearson correlation.

    ``np.corrcoef`` divides by each input's standard deviation, so an empty,
    non-finite or exactly constant crop yields NaN/Infinity instead of evidence.
    Constant crops are rejected on an exact zero variance: an epsilon tolerance
    would only hide the same undefined correlation behind a fabricated number.
    """

    if crop.shape != reference_shape or crop.size == 0:
        return False
    if not bool(np.isfinite(crop).all()):
        return False
    # A constant crop has exactly zero variance. Compare the value extent rather
    # than np.var, whose float32 mean rounding can leave a constant crop with a
    # tiny nonzero variance that would slip past an exact-zero comparison.
    return float(np.ptp(crop)) > 0.0


def _registered_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any] | None:
    """Compare a plausible near-neighbor after a small global translation.

    Estimate translation with ECC at full 256x144 resolution. OpenCV's default
    thread pool is excessive for these small crops, so configure it once per
    process. Tag-level expression and person-count protections are checked
    separately and always veto duplicate removal.
    """

    first = left["features"]["alignment_gray"].astype(np.float32) / 255.0
    second = right["features"]["alignment_gray"].astype(np.float32) / 255.0
    try:
        import cv2

        global _CV2_SINGLE_THREADED
        if not _CV2_SINGLE_THREADED:
            with _CV2_THREAD_CONFIG_LOCK:
                if not _CV2_SINGLE_THREADED:
                    cv2.setNumThreads(1)
                    _CV2_SINGLE_THREADED = True
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-6)
        _, warp = cv2.findTransformECC(
            first,
            second,
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            None,
            3,
        )
        dx, dy = float(warp[0, 2]), float(warp[1, 2])
        height, width = first.shape
        # A non-finite estimate must never reach warpAffine or the evidence
        # dict: report the existing conservative None instead of numbers.
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return None
        translation_fraction = max(abs(dx) / width, abs(dy) / height)
        if not math.isfinite(translation_fraction):
            return None
        aligned = cv2.warpAffine(
            second,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # The permitted motion is under 4% of the frame, so this fixed central
        # crop avoids edge interpolation artifacts without discarding the face.
        crop = (slice(8, height - 8), slice(12, width - 12))
        first_crop, aligned_crop = first[crop], aligned[crop]
        second_crop = second[crop]
        reference_shape = first_crop.shape
        # Reject undefined crops before correlating: an empty, non-finite or
        # exactly constant crop would fabricate NaN/Infinity correlations.
        if not (
            _correlation_crop_is_defined(first_crop, reference_shape)
            and _correlation_crop_is_defined(aligned_crop, reference_shape)
            and _correlation_crop_is_defined(second_crop, reference_shape)
        ):
            return None
        aligned_mae = float(np.mean(np.abs(first_crop - aligned_crop)))
        aligned_corr = float(np.corrcoef(first_crop.reshape(-1), aligned_crop.reshape(-1))[0, 1])
        before_corr = float(np.corrcoef(first_crop.reshape(-1), second_crop.reshape(-1))[0, 1])
        first_rgb = left["features"]["alignment_rgb"].astype(np.float32) / 255.0
        second_rgb = right["features"]["alignment_rgb"].astype(np.float32) / 255.0
        rgb_warp = warp.copy()
        rgb_warp[:, 2] *= 0.5
        rgb_height, rgb_width = first_rgb.shape[:2]
        aligned_rgb = cv2.warpAffine(
            second_rgb,
            rgb_warp,
            (rgb_width, rgb_height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        rgb_crop = (slice(4, rgb_height - 4), slice(6, rgb_width - 6))
        first_rgb_crop, aligned_rgb_crop = first_rgb[rgb_crop], aligned_rgb[rgb_crop]
        if not (
            first_rgb_crop.size > 0
            and bool(np.isfinite(first_rgb_crop).all())
            and bool(np.isfinite(aligned_rgb_crop).all())
        ):
            return None
        aligned_rgb_mae = float(np.mean(np.abs(first_rgb_crop - aligned_rgb_crop)))
        numeric = {
            "alignment_correlation_before": before_corr,
            "alignment_correlation": aligned_corr,
            "alignment_gain": aligned_corr - before_corr,
            "alignment_mae": aligned_mae,
            "alignment_rgb_mae": aligned_rgb_mae,
            "alignment_translation_fraction": translation_fraction,
            "alignment_dx_pixels": dx,
            "alignment_dy_pixels": dy,
        }
        # Strict JSON evidence stays NaN/Infinity-free: an undefined metric is
        # reported as an invalid registration (None), never as a number.
        if not all(math.isfinite(value) for value in numeric.values()):
            return None
        return {
            "alignment_checked": True,
            **{name: round(value, 8) for name, value in numeric.items()},
        }
    except Exception:
        return None


def _aligned_face_similarity(
    left: Mapping[str, Any], right: Mapping[str, Any], registration: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify every detected face after the already-computed global translation."""

    result: dict[str, Any] = {
        "aligned_face_check_performed": True,
        "all_face_matches_verified": False,
        "face_matching_count": 0,
        "face_match_ious": [],
        "minimum_face_support": None,
        "maximum_aligned_face_mae": None,
        "face_check_failure": None,
    }
    try:
        left_faces = left["features"]["faces"]
        right_faces = right["features"]["faces"]
        if not left_faces or not right_faces:
            result["face_check_failure"] = "missing_faces"
            return result
        if len(left_faces) != len(right_faces):
            result["face_check_failure"] = "face_count_mismatch"
            return result

        def valid_face(face: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
            try:
                box = np.asarray(face["box"], dtype=np.float64)
                rgb = np.asarray(face["rgb"], dtype=np.float32)
            except (KeyError, TypeError, ValueError, OverflowError):
                return None
            if (
                box.shape != (4,)
                or rgb.shape != (48, 48, 3)
                or not np.isfinite(box).all()
                or not np.isfinite(rgb).all()
                or np.min(rgb) < 0.0
                or np.max(rgb) > 1.0
                or box[0] < 0.0
                or box[1] < 0.0
                or box[2] > 1.0
                or box[3] > 1.0
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                return None
            return box, rgb

        normalized_left = [valid_face(face) for face in left_faces]
        normalized_right = [valid_face(face) for face in right_faces]
        if any(face is None for face in (*normalized_left, *normalized_right)):
            result["face_check_failure"] = "invalid_face_data"
            return result

        dx = float(registration["alignment_dx_pixels"])
        dy = float(registration["alignment_dy_pixels"])
        tx, ty = dx / 256.0, dy / 144.0
        if not math.isfinite(tx) or not math.isfinite(ty):
            result["face_check_failure"] = "nonfinite_translation"
            return result

        edges: list[tuple[float, int, int]] = []
        for left_index, left_face in enumerate(normalized_left):
            assert left_face is not None
            left_box, _ = left_face
            for right_index, right_face in enumerate(normalized_right):
                assert right_face is not None
                right_box, _ = right_face
                shifted_right_box = right_box + np.asarray([-tx, -ty, -tx, -ty], dtype=np.float64)
                overlap = _box_iou(left_box, shifted_right_box)
                if overlap >= 0.45:
                    edges.append((overlap, left_index, right_index))
        edges.sort(key=lambda edge: (-edge[0], edge[1], edge[2]))

        matched_left: set[int] = set()
        matched_right: set[int] = set()
        matches: list[tuple[float, int, int]] = []
        for overlap, left_index, right_index in edges:
            if left_index in matched_left or right_index in matched_right:
                continue
            matched_left.add(left_index)
            matched_right.add(right_index)
            matches.append((overlap, left_index, right_index))
        result["face_matching_count"] = len(matches)
        result["face_match_ious"] = [round(float(overlap), 8) for overlap, _, _ in matches]
        if len(matches) != len(left_faces) or len(matched_right) != len(right_faces):
            result["face_check_failure"] = "unmatched_faces"
            return result

        import cv2

        ones = np.ones((48, 48), dtype=np.float32)
        supports: list[float] = []
        maes: list[float] = []
        all_passed = True
        for _, left_index, right_index in matches:
            left_value = normalized_left[left_index]
            right_value = normalized_right[right_index]
            assert left_value is not None and right_value is not None
            left_box, left_rgb = left_value
            right_box, right_rgb = right_value
            lx0, ly0, lx1, ly1 = (float(value) for value in left_box)
            rx0, ry0, rx1, ry1 = (float(value) for value in right_box)
            matrix = np.asarray(
                [
                    [
                        (lx1 - lx0) / (rx1 - rx0),
                        0.0,
                        (lx0 + tx - rx0) * 48.0 / (rx1 - rx0),
                    ],
                    [
                        0.0,
                        (ly1 - ly0) / (ry1 - ry0),
                        (ly0 + ty - ry0) * 48.0 / (ry1 - ry0),
                    ],
                ],
                dtype=np.float32,
            )
            aligned = cv2.warpAffine(
                right_rgb,
                matrix,
                (48, 48),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            sampled_mask = cv2.warpAffine(
                ones,
                matrix,
                (48, 48),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            valid = sampled_mask >= 0.999999
            support = float(np.mean(valid))
            supports.append(support)
            if not np.isfinite(aligned).all() or not math.isfinite(support):
                all_passed = False
                continue
            if support < 0.80:
                all_passed = False
                continue
            mae = float(np.mean(np.abs(left_rgb[valid] - aligned[valid])))
            if not math.isfinite(mae):
                all_passed = False
                continue
            maes.append(mae)
            if mae > 0.065:
                all_passed = False

        result["minimum_face_support"] = round(min(supports), 8) if supports else None
        result["maximum_aligned_face_mae"] = round(max(maes), 8) if maes else None
        result["all_face_matches_verified"] = bool(all_passed and len(maes) == len(matches))
        if not result["all_face_matches_verified"]:
            result["face_check_failure"] = "support_or_aligned_mae_failed"
        return result
    except Exception as error:
        result["face_check_failure"] = f"alignment_error:{type(error).__name__}"
        return result


def _pair_evaluation(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    time_window: float,
    rejected_distance_floor: tuple[float, float, int] | None = None,
) -> dict[str, Any]:
    # Keep time_window in the private signature for existing callers, but time
    # distance is diagnostic only: a cut can recur anywhere in the video.
    del time_window
    metrics = _similarity_metrics(left, right)
    # Retain conservative behavior for older mocked/private evidence dictionaries
    # that only report the legacy combined expression field.
    legacy_expression_change = bool(metrics.get("expression_change", False))
    tag_expression_change = bool(metrics.get("tag_expression_change", legacy_expression_change))
    raw_face_change = bool(metrics.get("raw_face_change", False))
    if legacy_expression_change and not tag_expression_change and not raw_face_change:
        # Conflicting or partial evidence is conservative: an older combined
        # veto must never disappear because a newer field is absent/false.
        tag_expression_change = True
    metrics["tag_expression_change"] = tag_expression_change
    metrics["raw_face_change"] = raw_face_change
    metrics["expression_change"] = tag_expression_change or raw_face_change
    direct_duplicate = bool(
        not tag_expression_change
        and not raw_face_change
        and not metrics["material_people_change"]
        and metrics["phash_distance"] <= 3
        and metrics["dhash_distance"] <= 3
        and metrics["grid_local_hash_max_distance"] <= 4
        and metrics["resized_rgb_mean_absolute_difference"] <= 0.006
        and metrics["global_gray_ssim"] >= 0.985
    )
    registered_duplicate = False
    registration = None
    registered_candidate = bool(
        not direct_duplicate
        and not tag_expression_change
        and not metrics["material_people_change"]
        and metrics["phash_distance"] <= 22
        and metrics["dhash_distance"] <= 26
        and metrics["resized_rgb_mean_absolute_difference"] <= 0.15
        and metrics["global_gray_ssim"] >= 0.25
    )
    if registered_candidate:
        raw_distance = _evidence_distance(metrics)
        if rejected_distance_floor is not None and raw_distance >= rejected_distance_floor:
            metrics["registration_skipped_by_rejected_floor"] = True
        else:
            metrics["registration_skipped_by_rejected_floor"] = False
            registration = _registered_similarity(left, right)
    else:
        metrics["registration_skipped_by_rejected_floor"] = False
    if registered_candidate and registration is not None:
        metrics.update(registration)
        registration_global_gate = bool(
            registration["alignment_translation_fraction"] <= 0.11
            and registration["alignment_correlation"] >= 0.93
            and (
                registration["alignment_gain"] >= 0.01
                or registration["alignment_correlation_before"] >= 0.985
            )
            and registration["alignment_mae"] <= 0.025
            and registration["alignment_rgb_mae"] <= 0.03
        )
        if raw_face_change and registration_global_gate:
            face_evidence = _aligned_face_similarity(left, right, registration)
            metrics.update(face_evidence)
        face_verified = not raw_face_change or bool(metrics.get("all_face_matches_verified"))
        metrics["expression_change"] = tag_expression_change or (raw_face_change and not face_verified)
        registered_duplicate = registration_global_gate and face_verified
    if tag_expression_change:
        metrics["expression_change"] = True
    is_duplicate = direct_duplicate or registered_duplicate
    if is_duplicate:
        reason = "registered_structure_agrees" if registered_duplicate else "whole_frame_and_local_evidence_agree"
    elif metrics["expression_change"]:
        reason = "expression_or_face_change"
    elif metrics["material_people_change"]:
        reason = "person_count_change"
    elif metrics["grid_local_hash_max_distance"] > 4:
        reason = "local_structure_differs"
    elif metrics["resized_rgb_mean_absolute_difference"] > 0.006:
        reason = "pixel_difference_too_large"
    else:
        reason = "global_similarity_threshold_not_met"
    return {
        "left_id": left["id"],
        "right_id": right["id"],
        "near_duplicate": is_duplicate,
        "decision_reason": reason,
        "time_delta_seconds": round(abs(float(left["time"]) - float(right["time"])), 6),
        **metrics,
    }


def _hash_segment_keys(value: str, segment_count: int) -> list[tuple[int, int]]:
    """Split a fixed-width perceptual hash into disjoint LSH segments.

    With ``max_distance + 1`` segments, any pair within that Hamming distance
    must share at least one segment. These buckets are a recall-safe candidate
    index; pixel, local, registration, expression, and person-count checks
    still decide whether a pair is a duplicate.
    """

    hash_value = int(value, 16)
    bit_count = len(value) * 4
    segment_count = max(1, min(int(segment_count), bit_count))
    base_width, extra_bits = divmod(bit_count, segment_count)
    result: list[tuple[int, int]] = []
    offset = 0
    for segment in range(segment_count):
        width = base_width + (1 if segment < extra_bits else 0)
        result.append((segment, (hash_value >> offset) & ((1 << width) - 1)))
        offset += width
    return result


def _visual_hash_index(
    records: Sequence[dict[str, Any]], feature: str, max_distance: int
) -> tuple[dict[tuple[int, int], list[int]], list[list[tuple[int, int]]]]:
    buckets: dict[tuple[int, int], list[int]] = {}
    segments_by_record: list[list[tuple[int, int]]] = []
    for index, record in enumerate(records):
        segments = _hash_segment_keys(str(record["features"][feature]), max_distance + 1)
        segments_by_record.append(segments)
        for key in segments:
            buckets.setdefault(key, []).append(index)
    return buckets, segments_by_record


def _hash_neighbor_indexes(
    index: int,
    records: Sequence[dict[str, Any]],
    buckets: Mapping[tuple[int, int], Sequence[int]],
    segments_by_record: Sequence[Sequence[tuple[int, int]]],
    feature: str,
    max_distance: int,
) -> set[int]:
    possible: set[int] = set()
    for key in segments_by_record[index]:
        possible.update(buckets.get(key, ()))
    value = str(records[index]["features"][feature])
    possible.discard(index)
    return {
        other
        for other in possible
        if hamming_distance(value, str(records[other]["features"][feature])) <= max_distance
    }


def _pairwise_visual_hash_neighbors(
    records: Sequence[dict[str, Any]],
) -> list[set[int]]:
    """Build exact global hash neighbors without low-information LSH buckets.

    At the larger registered-duplicate radii, disjoint LSH segments become so
    short that their buckets nominate most of the video. For ordinary episode
    candidate sets, scanning the compact 64-bit hashes directly is faster and
    produces exactly the same neighbor relation. No timestamp or result-count
    limit is involved.
    """

    phashes = [int(str(record["features"]["phash"]), 16) for record in records]
    dhashes = [int(str(record["features"]["dhash"]), 16) for record in records]
    neighbors: list[set[int]] = [set() for _ in records]
    for left_index in range(len(records)):
        left_phash, left_dhash = phashes[left_index], dhashes[left_index]
        for right_index in range(left_index + 1, len(records)):
            if (
                (left_phash ^ phashes[right_index]).bit_count() <= _GLOBAL_PHASH_CANDIDATE_DISTANCE
                and (left_dhash ^ dhashes[right_index]).bit_count() <= _GLOBAL_DHASH_CANDIDATE_DISTANCE
            ):
                neighbors[left_index].add(right_index)
                neighbors[right_index].add(left_index)
    return neighbors


def _indexed_visual_hash_neighbors(
    index: int,
    records: Sequence[dict[str, Any]],
    phash_buckets: Mapping[tuple[int, int], Sequence[int]],
    phash_segments: Sequence[Sequence[tuple[int, int]]],
    dhash_buckets: Mapping[tuple[int, int], Sequence[int]],
    dhash_segments: Sequence[Sequence[tuple[int, int]]],
) -> set[int]:
    phash_neighbors = _hash_neighbor_indexes(
        index,
        records,
        phash_buckets,
        phash_segments,
        "phash",
        _GLOBAL_PHASH_CANDIDATE_DISTANCE,
    )
    dhash_neighbors = _hash_neighbor_indexes(
        index,
        records,
        dhash_buckets,
        dhash_segments,
        "dhash",
        _GLOBAL_DHASH_CANDIDATE_DISTANCE,
    )
    return phash_neighbors.intersection(dhash_neighbors)


def _coarse_pair_evaluation(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Explain a pair rejected by necessary global-hash limits without alignment."""

    left_features, right_features = left["features"], right["features"]
    return {
        "left_id": left["id"],
        "right_id": right["id"],
        "near_duplicate": False,
        "decision_reason": "global_hash_distance_exceeded",
        "phash_distance": hamming_distance(left_features["phash"], right_features["phash"]),
        "dhash_distance": hamming_distance(left_features["dhash"], right_features["dhash"]),
        "time_delta_seconds": round(abs(float(left["time"]) - float(right["time"])), 6),
    }


def _visual_groups(
    records: Sequence[dict[str, Any]],
    time_window: float,
    *,
    progress: Callable[[int, int], None] | None = None,
    progress_total: int | None = None,
) -> list[list[dict[str, Any]]]:
    # Search globally with 64-bit Hamming neighbors. Time is diagnostic only.
    # Use direct pair scanning at the supported episode candidate scale; fall
    # back to recall-safe LSH buckets for unusually large candidate sets.
    pairwise_neighbors: list[set[int]] | None = None
    phash_buckets: dict[tuple[int, int], list[int]] | None = None
    phash_segments: list[list[tuple[int, int]]] | None = None
    dhash_buckets: dict[tuple[int, int], list[int]] | None = None
    dhash_segments: list[list[tuple[int, int]]] | None = None
    if len(records) <= 5000:
        pairwise_neighbors = _pairwise_visual_hash_neighbors(records)
    else:
        phash_buckets, phash_segments = _visual_hash_index(
            records, "phash", _GLOBAL_PHASH_CANDIDATE_DISTANCE
        )
        dhash_buckets, dhash_segments = _visual_hash_index(
            records, "dhash", _GLOBAL_DHASH_CANDIDATE_DISTANCE
        )
    record_index = {record["id"]: index for index, record in enumerate(records)}

    # Greedy complete-link groups avoid a chain of tiny changes collapsing two
    # frames that are materially different from each other.
    ordered = sorted(records, key=_time_sort)
    total = len(records) if progress_total is None else int(progress_total)
    completed = 0
    groups: list[list[dict[str, Any]]] = []
    record_group_index: dict[str, int] = {}
    for candidate in ordered:
        candidate_index = record_index[candidate["id"]]
        if pairwise_neighbors is not None:
            neighbor_indexes = pairwise_neighbors[candidate_index]
        else:
            assert phash_buckets is not None and phash_segments is not None
            assert dhash_buckets is not None and dhash_segments is not None
            neighbor_indexes = _indexed_visual_hash_neighbors(
                candidate_index,
                records,
                phash_buckets,
                phash_segments,
                dhash_buckets,
                dhash_segments,
            )
        # Both global hashes must nominate the pair. Hashes only find eligible
        # pairs; whole-frame, local, pixel, registration, and protection checks
        # still decide whether frames can collapse.
        visual_neighbors = {records[index]["id"] for index in neighbor_indexes}
        possible_groups = sorted(
            {
                record_group_index[neighbor_id]
                for neighbor_id in visual_neighbors
                if neighbor_id in record_group_index
            }
        )
        best_rejected: dict[str, Any] | None = None
        admitted = False
        for group_index in possible_groups:
            group = groups[group_index]
            evidence: list[dict[str, Any]] = []
            group_rejected = False
            for member in group:
                if member["id"] in visual_neighbors:
                    rejection_floor = (
                        _evidence_distance(best_rejected)
                        if group_rejected and best_rejected is not None
                        else None
                    )
                    item = _pair_evaluation(
                        candidate,
                        member,
                        time_window,
                        rejected_distance_floor=rejection_floor,
                    )
                else:
                    item = _coarse_pair_evaluation(candidate, member)
                evidence.append(item)
                if not item["near_duplicate"]:
                    group_rejected = True
                    if best_rejected is None or _evidence_distance(item) < _evidence_distance(best_rejected):
                        best_rejected = item
            duplicate_pairs = [item for item in evidence if item["near_duplicate"]]
            if len(duplicate_pairs) == len(evidence):
                group.append(candidate)
                record_group_index[candidate["id"]] = group_index
                admitted = True
                checks = group[0].setdefault(
                    "_group_pair_checks",
                    {"evaluated": 0, "confirmed": 0, "admitted_frames": 0, "full_proof_count": 0},
                )
                checks.setdefault("admitted_frames", 0)
                checks.setdefault("full_proof_count", 0)
                checks["evaluated"] += len(evidence)
                checks["confirmed"] += len(duplicate_pairs)
                checks["admitted_frames"] += 1
                # Preserve one full visual proof per admitted frame. Complete-
                # link still checks every pair, while compact evidence avoids
                # storing a quadratic number of duplicate metrics per group.
                if duplicate_pairs:
                    group[0].setdefault("_group_pair_evidence", []).append(duplicate_pairs[0])
                    checks["full_proof_count"] += 1
                break
        if not admitted:
            candidate["_group_pair_evidence"] = [best_rejected] if best_rejected is not None else []
            groups.append([candidate])
            record_group_index[candidate["id"]] = len(groups) - 1
        completed += 1
        if progress is not None:
            progress(completed, total)
    return groups


def _evidence_distance(evidence: Mapping[str, Any]) -> tuple[float, float, int]:
    if evidence.get("decision_reason") == "global_hash_distance_exceeded":
        return (
            max(
                float(evidence.get("phash_distance", 64)) / 64.0,
                float(evidence.get("dhash_distance", 64)) / 64.0,
            ),
            1.0,
            64,
        )
    return (
        float(evidence.get("resized_rgb_mean_absolute_difference", 1.0)),
        1.0 - float(evidence.get("global_gray_ssim", 0.0)),
        int(evidence.get("grid_local_hash_max_distance", 64)),
    )


def _pair_summary(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return _pair_evaluation(left, right, math.inf)


def _stable_group_id(ids: Sequence[str] | Any) -> str:
    payload = "\0".join(sorted(str(value) for value in ids)).encode("utf-8")
    return "cg_" + hashlib.sha256(payload).hexdigest()[:12]


def _public_candidate(
    item: Mapping[str, Any], group_id: str, *, include_full_tags: bool = True
) -> dict[str, Any]:
    record = dict(item["record"])
    if not include_full_tags:
        record["tags"] = _compact_tags(item["tags"])
        record["tags_compacted"] = True
    record.update(
        {
            "id": item["id"],
            "group_id": group_id,
            "character_presence": bool(item["presence"]["present"]),
            "presence_score": item["presence"]["confidence"],
            "presence_evidence": {
                "method": item["presence"]["method"],
                "tag_evidence": list(item["presence"]["tag_evidence"]),
                "detector_evidence": list(item["presence"]["detector_evidence"]),
                "content_gate": dict(item["presence"]["content_gate"]),
            },
            "cut_quality": item["rank_score"],
            "quality_components": dict(item["quality_components"]),
            "visual_metrics": {
                "phash": item["features"]["phash"],
                "dhash": item["features"]["dhash"],
                "local_hashes_4x3": list(item["features"]["grid_hashes"]),
            },
        }
    )
    return record


def _compact_tags(tags: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Keep only person, expression, framing, composition, and text cues."""

    relevant = (
        _EXPRESSION_TAGS
        | _PRESENCE_TAGS
        | frozenset(_FACE_VISIBILITY_TAG_WEIGHTS)
        | frozenset(_FACE_OBSCURATION_TAG_WEIGHTS)
        | frozenset(_COMPOSITION_TAG_WEIGHTS)
        | frozenset({"text_focus", "credits", "foreground_text", "english_text", "japanese_text"})
    )
    result: dict[str, dict[str, float]] = {}
    for category, values in tags.items():
        selected = {
            str(tag): float(score)
            for tag, score in values.items()
            if str(tag) in relevant or (category == "general" and _is_person_tag(str(tag)))
        }
        if selected:
            result[str(category)] = selected
    return result


def _time_sort(item: Mapping[str, Any]) -> tuple[float, str, str]:
    time_value = item.get("time")
    try:
        numeric_time = float(time_value) if time_value is not None else math.inf
    except (TypeError, ValueError):
        numeric_time = math.inf
    pts = item.get("pts")
    return numeric_time, str(pts if pts is not None else ""), str(item.get("id", ""))


def _public_time_sort(item: Mapping[str, Any]) -> tuple[float, str, str]:
    return _time_sort(item)


__all__ = ["curate_candidates"]
