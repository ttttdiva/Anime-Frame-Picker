"""Pure scoring, image-quality, tag-gating, and pixel deduplication helpers.

The search pipeline deliberately keeps identity matching in the model service.  The
functions in this module only transform the service's official distances for display,
measure the target crop, and rank/deduplicate already matched candidates.  In
particular, no perceptual hash is used as an identity signal.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # Pillow is a runtime dependency of the local frame picker.
    from PIL import Image, ImageFilter
except Exception:  # pragma: no cover - makes the pure helpers importable in tooling
    Image = Any  # type: ignore[assignment,misc]
    ImageFilter = None  # type: ignore[assignment]


DEFAULT_WEIGHTS = {"identity": 0.35, "condition": 0.30, "quality": 0.35}


def clamp01(value: Any, default: float = 0.0) -> float:
    """Return *value* clamped to ``[0, 1]``.

    Model and tag services are external to this module, so malformed values are
    treated as a neutral low score instead of poisoning an entire result set.
    """

    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return max(0.0, min(1.0, number))


def normalize_weights(weights: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Normalize identity/condition/quality weights without changing their ratios."""

    source = dict(DEFAULT_WEIGHTS)
    if weights:
        for key in source:
            value = weights.get(key)
            if value is not None:
                try:
                    source[key] = max(0.0, float(value))
                except (TypeError, ValueError):
                    pass
    total = sum(source.values())
    if total <= 0:
        return {key: 1.0 / len(source) for key in source}
    return {key: value / total for key, value in source.items()}


def identity_display_score(distance: Any, threshold: Any) -> float:
    """Map an official distance to a display score in ``[0, 1]``.

    Acceptance remains a strict distance comparison in :func:`identity_match`;
    this value is only the user-facing score component.
    """

    try:
        distance_value = float(distance)
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(distance_value) or not math.isfinite(threshold_value):
        return 0.0
    if threshold_value <= 0:
        return 1.0 if distance_value <= 0 else 0.0
    return clamp01(1.0 - distance_value / threshold_value)


# Short aliases are useful to callers that use the term "identity score".
distance_to_identity_score = identity_display_score
identity_score = identity_display_score


def identity_match(distance: Any, threshold: Any) -> bool:
    """Return whether an official model distance passes its configured threshold."""

    try:
        distance_value = float(distance)
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        return False
    return math.isfinite(distance_value) and math.isfinite(threshold_value) and distance_value <= threshold_value


def aggregate_reference_distances(
    distances: Sequence[Any] | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Aggregate distances for one candidate against several references.

    The minimum official distance is the acceptance signal.  Other references
    contribute a soft corroboration value so a single close pose can still match,
    while a candidate close to several reference poses receives a more stable
    display score.  This never turns a failing minimum distance into a match.
    """

    values: list[float] = []
    for value in np.asarray(distances).reshape(-1).tolist():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    if not values:
        return {
            "distance": float("inf"),
            "corroboration": 0.0,
            "matched_references": 0,
            "accepted": False,
            "identity": 0.0,
        }
    minimum = min(values)
    threshold_value = float(threshold)
    accepted = identity_match(minimum, threshold_value)
    # A small tolerance allows different expressions/crops to corroborate a
    # minimum without making the acceptance boundary more permissive.
    tolerance = max(0.01, abs(threshold_value) * 0.20)
    matched = sum(1 for value in values if value <= threshold_value + tolerance)
    corroboration = matched / max(1, len(values))
    identity = identity_display_score(minimum, threshold_value)
    if len(values) > 1 and accepted:
        # Keep the minimum as the dominant signal; corroboration adds at most 15%.
        identity = clamp01(identity * (0.85 + 0.15 * corroboration))
    return {
        "distance": minimum,
        "corroboration": corroboration,
        "matched_references": matched,
        "accepted": accepted,
        "identity": identity,
    }


# A discoverable name for code that calls this operation a multi-reference match.
multi_reference_match = aggregate_reference_distances


def combine_score(
    identity: Any,
    condition: Any,
    quality: Any,
    weights: Mapping[str, Any] | None = None,
) -> float:
    """Combine independent identity, condition, and image-quality scores."""

    normalized = normalize_weights(weights)
    return clamp01(
        normalized["identity"] * clamp01(identity)
        + normalized["condition"] * clamp01(condition)
        + normalized["quality"] * clamp01(quality)
    )


score_candidate = combine_score


def _rgb_array(image: Any) -> np.ndarray:
    """Convert a PIL image or array to a finite RGB uint8 array."""

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        elif array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        elif array.ndim == 3 and array.shape[2] > 3:
            array = array[..., :3]
    else:
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        elif array.ndim == 3 and array.shape[2] > 3:
            array = array[..., :3]
    if array.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


def _gray_array(image: Any) -> np.ndarray:
    rgb = _rgb_array(image).astype(np.float32)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _resize_gray(image: Any, size: tuple[int, int]) -> np.ndarray:
    if hasattr(image, "convert") and Image is not Any:
        pil = image.convert("L")
        return np.asarray(pil.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    gray = _gray_array(image)
    # Pillow is normally installed, but nearest-neighbour indexing keeps hashes
    # available in a minimal test environment too.
    ys = np.linspace(0, max(0, gray.shape[0] - 1), size[1]).round().astype(int)
    xs = np.linspace(0, max(0, gray.shape[1] - 1), size[0]).round().astype(int)
    return gray[np.ix_(ys, xs)]


def _dct_2d(values: np.ndarray) -> np.ndarray:
    """Small dependency-free DCT-II used by the perceptual hash."""

    height, width = values.shape
    x = np.arange(height, dtype=np.float64)[:, None]
    u = np.arange(height, dtype=np.float64)[None, :]
    basis_y = np.cos(np.pi * (2 * x + 1) * u / (2 * height))
    x = np.arange(width, dtype=np.float64)[:, None]
    u = np.arange(width, dtype=np.float64)[None, :]
    basis_x = np.cos(np.pi * (2 * x + 1) * u / (2 * width))
    return basis_y.T @ values @ basis_x


def phash(image: Any, hash_size: int = 8, high_frequency_size: int | None = None) -> str:
    """Return a compact pixel perceptual hash as hexadecimal text."""

    hash_size = max(2, int(hash_size))
    high_frequency_size = int(high_frequency_size or hash_size * 4)
    high_frequency_size = max(hash_size, min(64, high_frequency_size))
    gray = _resize_gray(image, (high_frequency_size, high_frequency_size))
    coefficients = _dct_2d(gray)[:hash_size, :hash_size]
    low = coefficients[1:, 1:]
    median = float(np.median(low)) if low.size else float(np.median(coefficients))
    bits = coefficients > median
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    width = (bits.size + 3) // 4
    return f"{value:0{width}x}"


def dhash(image: Any, hash_size: int = 8) -> str:
    """Return a horizontal difference hash as hexadecimal text."""

    hash_size = max(2, int(hash_size))
    gray = _resize_gray(image, (hash_size + 1, hash_size))
    bits = gray[:, 1:] > gray[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    width = (bits.size + 3) // 4
    return f"{value:0{width}x}"


def hamming_distance(left: str, right: str) -> int:
    """Return the bit distance between two hexadecimal hashes."""

    try:
        return (int(str(left), 16) ^ int(str(right), 16)).bit_count()
    except (TypeError, ValueError):
        # Different hash widths are still comparable as strings of bits.
        left_bits = bin(int.from_bytes(str(left).encode(), "utf-8")).count("1")
        right_bits = bin(int.from_bytes(str(right).encode(), "utf-8")).count("1")
        return left_bits + right_bits


def _crop_from_box(image: Any, bbox: Sequence[float] | None) -> Any:
    if bbox is None:
        return image
    if len(bbox) != 4:
        return image
    if hasattr(image, "size") and not isinstance(image, np.ndarray):
        width, height = image.size
    else:
        array = _rgb_array(image)
        height, width = array.shape[:2]
    x0, y0, x1, y1 = [float(value) for value in bbox]
    # Accept either normalized or pixel coordinates.
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    x0 = max(0, min(width - 1, int(round(x0))))
    y0 = max(0, min(height - 1, int(round(y0))))
    x1 = max(x0 + 1, min(width, int(round(x1))))
    y1 = max(y0 + 1, min(height, int(round(y1))))
    if hasattr(image, "crop"):
        return image.crop((x0, y0, x1, y1))
    return _rgb_array(image)[y0:y1, x0:x1]


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1] * 4.0
    neighbors = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(np.var(center - neighbors))


def quality_metrics(
    image: Any,
    bbox: Sequence[float] | None = None,
    previous: Any | None = None,
    following: Any | None = None,
    settings: Mapping[str, Any] | None = None,
    *,
    previous_time: float | None = None,
    following_time: float | None = None,
    current_time: float | None = None,
) -> dict[str, Any]:
    """Measure target-region sharpness, contrast, clipping, size, and transitions.

    These are deliberately image-quality heuristics.  There is no anatomy or
    universal blink detector here; optional blink penalties are only applied when
    the caller explicitly requests them.
    """

    settings = settings or {}
    target = _crop_from_box(image, bbox)
    array = _rgb_array(target)
    gray = _gray_array(target)
    height, width = gray.shape[:2]
    blur_raw = _laplacian_variance(gray)
    # A 180 variance is sharp enough for a 960px anime crop; this saturates rather
    # than pretending that sharpness has a universal scale.
    blur_score = clamp01(blur_raw / float(settings.get("blur_reference", 180.0)))
    contrast_raw = float(np.std(gray))
    contrast_score = clamp01(contrast_raw / float(settings.get("contrast_reference", 55.0)))
    clipping_fraction = float(np.mean((array <= 2) | (array >= 253)))
    clipping_score = clamp01(1.0 - clipping_fraction * 2.2)
    frame_width, frame_height = _rgb_array(image).shape[1], _rgb_array(image).shape[0]
    area_fraction = (width * height) / max(1.0, frame_width * frame_height)
    # Very small target crops are poor evidence even when visually crisp.
    size_score = clamp01(math.sqrt(max(0.0, area_fraction)) * 3.2)

    temporal_differences: list[float] = []
    current_small = _resize_gray(target, (32, 32))
    require_temporal_times = bool(settings.get("temporal_require_times", False))
    temporal_max_gap = settings.get("temporal_max_gap")
    try:
        temporal_max_gap_value = float(temporal_max_gap) if temporal_max_gap is not None else math.inf
    except (TypeError, ValueError):
        temporal_max_gap_value = math.inf
    for neighbor, neighbor_time in ((previous, previous_time), (following, following_time)):
        if neighbor is None:
            continue
        # Sparse coarse/dense neighbors can be different shots.  Only compare
        # frames whose timestamps are actually close; the pipeline requires
        # timestamps, while direct helper callers retain the historical behavior
        # when no temporal window is requested.
        if require_temporal_times and (current_time is None or neighbor_time is None):
            continue
        if current_time is not None and neighbor_time is not None:
            try:
                if abs(float(current_time) - float(neighbor_time)) > temporal_max_gap_value:
                    continue
            except (TypeError, ValueError):
                if require_temporal_times:
                    continue
        neighbor_target = _crop_from_box(neighbor, bbox)
        neighbor_small = _resize_gray(neighbor_target, (32, 32))
        temporal_differences.append(float(np.mean(np.abs(current_small - neighbor_small))) / 255.0)
    temporal_delta = float(np.mean(temporal_differences)) if temporal_differences else 0.0
    # A single frame can legitimately change expression; only extreme local
    # differences are treated as a transition/fade warning.
    transition_penalty = clamp01((temporal_delta - 0.34) / 0.40)
    condition = clamp01(1.0 - transition_penalty)

    blink_penalty = 0.0
    if bool(settings.get("penalize_blink", settings.get("blink_penalty", False))):
        # No anatomy model is claimed.  This optional soft penalty only reacts to
        # a caller-provided scalar, if one exists from a dedicated detector.
        blink_penalty = clamp01(settings.get("blink_penalty_value", 0.0))
        condition *= 1.0 - 0.35 * blink_penalty

    quality = clamp01(
        0.35 * blur_score + 0.25 * contrast_score + 0.20 * clipping_score + 0.20 * size_score
    )
    return {
        "quality": quality,
        "condition": condition,
        "blur": blur_score,
        "blur_raw": blur_raw,
        "contrast": contrast_score,
        "contrast_raw": contrast_raw,
        "clipping": clipping_score,
        "clipping_fraction": clipping_fraction,
        "size": size_score,
        "area_fraction": area_fraction,
        "temporal_delta": temporal_delta,
        "transition_penalty": transition_penalty,
        "blink_penalty": blink_penalty,
        "reasons": [
            reason
            for reason, flagged in (
                ("blur", blur_score < 0.18),
                ("low_contrast", contrast_score < 0.18),
                ("clipped", clipping_fraction > 0.40),
                ("small_target", size_score < 0.12),
                ("transition", transition_penalty > 0.5),
                ("blink", blink_penalty > 0),
            )
            if flagged
        ],
    }


compute_quality = quality_metrics


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    """Intersection over union for normalized or pixel boxes."""

    if len(left) != 4 or len(right) != 4:
        return 0.0
    lx0, ly0, lx1, ly1 = map(float, left)
    rx0, ry0, rx1, ry1 = map(float, right)
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def overlap_penalty(target_box: Sequence[float], other_boxes: Iterable[Sequence[float]]) -> float:
    """Return a severe-overlap penalty in ``[0, 1]`` for a target crop."""

    return clamp01(max((box_iou(target_box, other) for other in other_boxes), default=0.0))


def _tag_maps(tags: Mapping[str, Any] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    if not isinstance(tags, Mapping):
        return result
    for group in ("general", "character", "tags"):
        values = tags.get(group)
        if isinstance(values, Mapping):
            for key, value in values.items():
                try:
                    result[str(key).strip().lower()] = max(result.get(str(key).strip().lower(), 0.0), clamp01(value))
                except Exception:
                    continue
    # A flat tag map is also accepted for lightweight local model adapters.
    if not any(isinstance(tags.get(group), Mapping) for group in ("general", "character", "tags")):
        for key, value in tags.items():
            if isinstance(value, (float, int)):
                result[str(key).strip().lower()] = clamp01(value)
    return result


def evaluate_tags(
    subject_tags: Mapping[str, Any] | None,
    frame_tags: Mapping[str, Any] | None,
    wanted: Sequence[Mapping[str, Any]] | None = None,
    avoided: Sequence[Mapping[str, Any]] | None = None,
    threshold: float = 0.17,
) -> dict[str, Any]:
    """Gate and score subject/frame tags without mixing their meanings.

    Subject tags come from the matched person crop; frame tags come from the full
    frame.  A requested subject expression therefore cannot be satisfied by a
    different character elsewhere in the frame.
    """

    subject = _tag_maps(subject_tags)
    frame = _tag_maps(frame_tags)
    threshold_value = clamp01(threshold)

    def requirement_value(requirement: Mapping[str, Any]) -> float:
        tag = str(requirement.get("tag", "")).strip().lower()
        scope = str(requirement.get("scope", "subject")).strip().lower()
        values = frame if scope == "frame" else subject
        return clamp01(values.get(tag, 0.0))

    wanted_values = [requirement_value(item) for item in (wanted or [])]
    avoided_values = [requirement_value(item) for item in (avoided or [])]
    wanted_pass = all(value >= threshold_value for value in wanted_values)
    avoided_pass = all(value < threshold_value for value in avoided_values)
    wanted_score = float(np.mean(wanted_values)) if wanted_values else 1.0
    avoided_score = 1.0 - (max(avoided_values) if avoided_values else 0.0)
    score = clamp01(wanted_score * avoided_score)
    reasons: list[str] = []
    for item, value in zip(wanted or [], wanted_values):
        if value < threshold_value:
            reasons.append(f"wanted:{item.get('tag', '')}")
    for item, value in zip(avoided or [], avoided_values):
        if value >= threshold_value:
            reasons.append(f"avoided:{item.get('tag', '')}")
    return {
        "passed": wanted_pass and avoided_pass,
        "score": score,
        "wanted": wanted_values,
        "avoided": avoided_values,
        "reasons": reasons,
    }


tag_gate = evaluate_tags


SUBJECT_AMBIGUOUS_GENERAL_TAGS = frozenset(
    {
        "2girls",
        "2boys",
        "multiple_girls",
        "multiple_boys",
        "multiple girls",
        "multiple boys",
    }
)


def subject_tag_ambiguity(
    tags: Mapping[str, Any] | None,
    threshold: float = 0.8,
) -> dict[str, float]:
    """Detect strong general multi-person signals for subject attribution.

    This is a conservative tag heuristic, not an anatomy truth claim.  Only
    exact known general labels are considered; character/frame scopes are kept
    separate by the caller.
    """

    if not isinstance(tags, Mapping):
        return {}
    general = tags.get("general")
    if not isinstance(general, Mapping):
        return {}
    threshold_value = clamp01(threshold)
    result: dict[str, float] = {}
    for name, value in general.items():
        if str(name).strip().lower() not in SUBJECT_AMBIGUOUS_GENERAL_TAGS:
            continue
        score = clamp01(value)
        if score >= threshold_value:
            result[str(name)] = score
    return result


subject_ambiguity = subject_tag_ambiguity


def compact_tag_evidence(
    tags: Mapping[str, Any] | None,
    explicit_tags: Sequence[str | Mapping[str, Any]] | None = None,
    threshold: float = 0.17,
    *,
    general_limit: int = 20,
    category_limit: int = 3,
) -> dict[str, dict[str, float]]:
    """Return bounded public tag evidence while retaining full maps internally.

    General tags are limited to the strongest values at/above the official
    threshold.  Auxiliary categories are limited to a few strong entries.  Tags
    explicitly requested by the user are retained even when their confidence is
    below the threshold so the UI can explain a failed gate without serializing
    the model's entire vocabulary output.
    """

    if not isinstance(tags, Mapping):
        return {}
    groups: dict[str, Mapping[str, Any]] = {}
    has_nested = False
    for group, values in tags.items():
        if isinstance(values, Mapping):
            groups[str(group)] = values
            has_nested = True
    if not has_nested:
        groups = {"general": tags}

    requested: set[str] = set()
    for item in explicit_tags or []:
        if isinstance(item, Mapping):
            value = item.get("tag")
        else:
            value = item
        if value is not None:
            requested.add(str(value).strip().lower())
    threshold_value = clamp01(threshold)
    output: dict[str, dict[str, float]] = {}
    auxiliary_groups = 0
    for group, values in groups.items():
        numeric: list[tuple[str, float]] = []
        for name, value in values.items():
            try:
                score = clamp01(value)
            except Exception:
                continue
            numeric.append((str(name), score))
        numeric.sort(key=lambda item: (-item[1], item[0].lower()))
        limit = max(0, int(general_limit if group.lower() == "general" else category_limit))
        group_requested = any(name.strip().lower() in requested for name, _score in numeric)
        if group.lower() != "general" and auxiliary_groups >= 8 and not group_requested:
            continue
        selected: dict[str, float] = {
            name: score for name, score in numeric if score >= threshold_value
        }
        selected = dict(list(selected.items())[:limit])
        # Explicitly requested tags win a slot and are retained at low score.
        for name, score in numeric:
            if name.strip().lower() in requested:
                selected[name] = score
        if selected:
            output[group] = selected
            if group.lower() != "general":
                auxiliary_groups += 1
    return output


compact_tags = compact_tag_evidence


def _candidate_hashes(candidate: Mapping[str, Any]) -> tuple[str | None, str | None, str | None, str | None]:
    """Read full/crop pHash and dHash fields from a candidate mapping."""

    hashes = candidate.get("hashes")
    if not isinstance(hashes, Mapping):
        hashes = {}
    full_phash = candidate.get("phash", hashes.get("phash"))
    full_dhash = candidate.get("dhash", hashes.get("dhash"))
    crop_phash = candidate.get("crop_phash", hashes.get("crop_phash"))
    crop_dhash = candidate.get("crop_dhash", hashes.get("crop_dhash"))
    full_image = candidate.get("image", candidate.get("_frame"))
    crop_image = candidate.get("crop_image", candidate.get("_crop"))
    if full_image is not None:
        try:
            full_phash = full_phash or phash(full_image)
            full_dhash = full_dhash or dhash(full_image)
        except Exception:
            pass
    if crop_image is not None:
        try:
            crop_phash = crop_phash or phash(crop_image)
            crop_dhash = crop_dhash or dhash(crop_image)
        except Exception:
            pass
    return (
        str(full_phash) if full_phash is not None else None,
        str(full_dhash) if full_dhash is not None else None,
        str(crop_phash) if crop_phash is not None else None,
        str(crop_dhash) if crop_dhash is not None else None,
    )


def _aligned_target_similarity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[int, int, float] | None:
    """Compare the common normalized target region of two full-frame images.

    Detector boxes can jitter between nearby frames, making hashes of their
    individual crops disagree even when the underlying target is unchanged.
    Comparing the shared normalized intersection removes that padding jitter.
    The returned pHash/dHash pair remains pixel-only; the grayscale MAD guard
    keeps a small expression change from being treated as a duplicate.
    """

    left_image = left.get("image", left.get("_frame"))
    right_image = right.get("image", right.get("_frame"))
    left_box = left.get("bbox")
    right_box = right.get("bbox")
    if left_image is None or right_image is None or not hasattr(left_image, "crop") or not hasattr(right_image, "crop"):
        return None
    if not isinstance(left_box, Sequence) or not isinstance(right_box, Sequence) or len(left_box) != 4 or len(right_box) != 4:
        return None
    try:
        common = [
            max(float(left_box[0]), float(right_box[0])),
            max(float(left_box[1]), float(right_box[1])),
            min(float(left_box[2]), float(right_box[2])),
            min(float(left_box[3]), float(right_box[3])),
        ]
    except (TypeError, ValueError):
        return None
    if common[2] <= common[0] or common[3] <= common[1]:
        return None
    # A tiny intersection is dominated by interpolation/padding noise and is
    # not enough evidence that two target crops depict the same subject region.
    if (common[2] - common[0]) * (common[3] - common[1]) < 0.08:
        return None
    try:
        # Ignore a narrow border where detector jitter, subtitles, credits, or
        # moving background are most likely to dominate. The inner target region
        # still retains eyes/mouth and other expression evidence.
        margin = 0.15
        common_width = common[2] - common[0]
        common_height = common[3] - common[1]
        common = [
            common[0] + common_width * margin,
            common[1] + common_height * margin,
            common[2] - common_width * margin,
            common[3] - common_height * margin,
        ]
        left_crop = _crop_from_box(left_image, common)
        right_crop = _crop_from_box(right_image, common)
        left_gray = _resize_gray(left_crop, (64, 64))
        right_gray = _resize_gray(right_crop, (64, 64))
        mad = float(np.mean(np.abs(left_gray - right_gray))) / 255.0
        return hamming_distance(phash(left_crop), phash(right_crop)), hamming_distance(dhash(left_crop), dhash(right_crop)), mad
    except (TypeError, ValueError, OSError):
        return None


def are_pixel_duplicates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    hash_distance: int = 6,
    time_window: float = 2.0,
    scene_distance: float | None = None,
) -> bool:
    """Decide whether two nearby candidates are pixel duplicates.

    Full-frame hashes establish scene similarity, while crop hashes protect
    different expressions in the same scene.  A model/CCIP distance is never
    consulted here.
    """

    try:
        if abs(float(left.get("time", 0.0)) - float(right.get("time", 0.0))) > float(time_window):
            return False
    except (TypeError, ValueError):
        return False
    # Head and body/context crops can share a PTS and scene while carrying very
    # different scope evidence. Keep both through subject-tag gating. A large
    # coverage ratio is likewise a signal that the crop windows represent
    # different context, even when their pixels are otherwise similar.
    left_kind = str(left.get("_detection_kind", "")).lower()
    right_kind = str(right.get("_detection_kind", "")).lower()
    if {left_kind, right_kind} == {"head", "person"}:
        return False
    left_box = left.get("bbox")
    right_box = right.get("bbox")
    try:
        if isinstance(left_box, Sequence) and isinstance(right_box, Sequence) and len(left_box) == 4 and len(right_box) == 4:
            left_area = max(0.0, float(left_box[2]) - float(left_box[0])) * max(0.0, float(left_box[3]) - float(left_box[1]))
            right_area = max(0.0, float(right_box[2]) - float(right_box[0])) * max(0.0, float(right_box[3]) - float(right_box[1]))
            if min(left_area, right_area) > 0 and max(left_area, right_area) / min(left_area, right_area) > 1.6:
                return False
    except (TypeError, ValueError):
        pass
    left_full_p, left_full_d, left_crop_p, left_crop_d = _candidate_hashes(left)
    right_full_p, right_full_d, right_crop_p, right_crop_d = _candidate_hashes(right)
    if scene_distance is None:
        # A caller may have precomputed a pixel scene similarity; this remains a
        # pixel-only signal and is useful when full-frame hashes were downscaled.
        left_scene = left.get("scene_similarity")
        right_scene = right.get("scene_similarity")
        if left_scene is not None and right_scene is not None:
            try:
                full_similar = abs(float(left_scene) - float(right_scene)) <= 0.12
            except (TypeError, ValueError):
                full_similar = False
        else:
            full_similar = bool(
                left_full_p
                and right_full_p
                and left_full_d
                and right_full_d
                and hamming_distance(left_full_p, right_full_p) <= hash_distance
                and hamming_distance(left_full_d, right_full_d) <= hash_distance
            )
    else:
        try:
            full_similar = float(scene_distance) <= 0.12
        except (TypeError, ValueError):
            full_similar = False
    aligned = _aligned_target_similarity(left, right)
    aligned_strong = False
    if aligned is not None:
        aligned_phash, aligned_dhash, aligned_mad = aligned
        left_box = left.get("bbox")
        right_box = right.get("bbox")
        try:
            bbox_overlap = box_iou(left_box, right_box) if left_box is not None and right_box is not None else 0.0
        except (TypeError, ValueError):
            bbox_overlap = 0.0
        aligned_strong = aligned_phash <= hash_distance and aligned_dhash <= hash_distance and aligned_mad <= 0.045 and bbox_overlap >= 0.65
    if not full_similar and not aligned_strong:
        return False
    # If crop hashes exist, they must also agree.  This intentionally keeps a
    # smile/open-eye change as a separate candidate even in the same shot.
    crop_hashes_agree = True
    if left_crop_p and right_crop_p and hamming_distance(left_crop_p, right_crop_p) > hash_distance:
        crop_hashes_agree = False
    if left_crop_d and right_crop_d and hamming_distance(left_crop_d, right_crop_d) > hash_distance:
        crop_hashes_agree = False
    if crop_hashes_agree:
        return True
    # Fall back to the common target region only when full-frame scene hashes
    # already agree. This handles crop-box jitter without collapsing a changed
    # shot or a clearly different expression.
    if aligned is None:
        return False
    aligned_phash, aligned_dhash, aligned_mad = aligned
    return aligned_phash <= hash_distance and aligned_dhash <= hash_distance and aligned_mad <= 0.045


def deduplicate_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    hash_distance: int = 6,
    time_window: float = 2.0,
    scene_distance: float | None = None,
) -> list[dict[str, Any]]:
    """Keep the strongest nearby pixel-distinct candidates."""

    ordered = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda item: (-float(item.get("score", 0.0)), float(item.get("time", 0.0)), str(item.get("id", ""))),
    )
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        if any(
            are_pixel_duplicates(
                candidate,
                existing,
                hash_distance=hash_distance,
                time_window=time_window,
                scene_distance=scene_distance,
            )
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


# Alternate spelling used by a few call sites.
dedupe_candidates = deduplicate_candidates


__all__ = [
    "DEFAULT_WEIGHTS",
    "aggregate_reference_distances",
    "are_pixel_duplicates",
    "box_iou",
    "clamp01",
    "combine_score",
    "compact_tag_evidence",
    "compact_tags",
    "compute_quality",
    "dedupe_candidates",
    "deduplicate_candidates",
    "dhash",
    "distance_to_identity_score",
    "evaluate_tags",
    "hamming_distance",
    "identity_display_score",
    "identity_match",
    "identity_score",
    "multi_reference_match",
    "normalize_weights",
    "overlap_penalty",
    "phash",
    "quality_metrics",
    "score_candidate",
    "subject_ambiguity",
    "subject_tag_ambiguity",
    "SUBJECT_AMBIGUOUS_GENERAL_TAGS",
    "tag_gate",
]
