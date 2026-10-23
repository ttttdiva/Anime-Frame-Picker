"""Group detector crops into "characters likely in this video" for the UI.

This module adds no model.  It reuses the configured ModelService detector
crops and its official CCIP ``embed``/``distance`` contract with the service
threshold, the same evidence ``--mode auto-topn`` uses for recurring
identity groups.  Grouping is anchor based and incremental:

* a new crop joins the existing candidate group with the closest CCIP member
  when that distance is within the official threshold;
* otherwise it starts a new group whose stable key is derived from the source
  fingerprint and the first crop's exact PTS/box.

The key therefore never depends on display order, page order, or which crop
is currently shown as representative.  Outfit, expression, and angle are not
used to split groups; only the CCIP decision does.  A missing CCIP contract
(or cached crops without features) degrades to one group per crop with
``identity_grouping`` reported as false rather than guessing.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

DEFAULT_IDENTITY_THRESHOLD = 0.2132311898
KEY_PATTERN = r"[A-Za-z0-9_-]{1,64}"


def _round_box(box: Sequence[float]) -> str:
    return ",".join(f"{float(value):.4f}" for value in box)


def detection_key(fingerprint: str, pts: int, box: Sequence[float]) -> str:
    """Stable id for one exact-PTS detector crop."""

    payload = f"{fingerprint}|{int(pts)}|{_round_box(box)}"
    return "det_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def character_key(fingerprint: str, anchor_detection_key: str) -> str:
    """Stable id for a candidate character group, fixed by its first crop."""

    payload = f"{fingerprint}|anchor|{anchor_detection_key}"
    return "chr_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def representative_score(member: Mapping[str, Any]) -> float:
    """Prefer a clear, large person/upper-body crop with a confident box."""

    box = member.get("box") or [0.0, 0.0, 0.0, 0.0]
    try:
        width = max(0.0, float(box[2]) - float(box[0]))
        height = max(0.0, float(box[3]) - float(box[1]))
    except (TypeError, ValueError, IndexError):
        return 0.0
    area = width * height
    # Very small boxes are background people; whole-frame boxes are rarely a
    # clear face.  sqrt keeps mid-size upper-body crops ahead of tiny heads.
    size = min(1.0, math.sqrt(area) / 0.45) if area > 0 else 0.0
    if area > 0.85:
        size *= 0.8
    score = member.get("score")
    try:
        confidence = float(score) if score is not None else 0.6
    except (TypeError, ValueError):
        confidence = 0.6
    if not math.isfinite(confidence):
        confidence = 0.6
    kind = str(member.get("kind", "person")).casefold()
    kind_weight = 1.0 if kind == "person" else 0.9 if kind == "head" else 0.8
    return float(size * (0.5 + 0.5 * confidence) * kind_weight)


@dataclass
class CandidateGroup:
    key: str
    anchor_detection_key: str
    members: list[dict[str, Any]] = field(default_factory=list)
    features: list[np.ndarray] = field(default_factory=list)

    def representative(self) -> dict[str, Any]:
        return max(
            self.members,
            key=lambda item: (representative_score(item), -float(item.get("time", 0.0)), item["detection_key"]),
        )


class CharacterCandidateIndex:
    """Incremental per-source candidate-character index."""

    def __init__(self, fingerprint: str, *, threshold: float | None = None) -> None:
        self.fingerprint = str(fingerprint)
        self.threshold = float(threshold) if threshold is not None else DEFAULT_IDENTITY_THRESHOLD
        self.groups: list[CandidateGroup] = []
        self._seen: set[str] = set()
        self.identity_grouping = True

    def __len__(self) -> int:
        return len(self.groups)

    def add(
        self,
        records: Sequence[Mapping[str, Any]],
        features: Sequence[Any] | None,
        distance: Callable[[np.ndarray, np.ndarray], Any] | None,
    ) -> list[str]:
        """Add crops; return the character key assigned to each record."""

        assigned: list[str] = []
        usable = features is not None and distance is not None and len(features) == len(records)
        if not usable and records:
            self.identity_grouping = False
        for index, raw in enumerate(records):
            record = dict(raw)
            key = record.get("detection_key") or detection_key(self.fingerprint, int(record["pts"]), record["box"])
            record["detection_key"] = key
            if key in self._seen:
                existing = next(group for group in self.groups if any(m["detection_key"] == key for m in group.members))
                assigned.append(existing.key)
                continue
            vector = np.asarray(features[index], dtype=np.float32).reshape(-1) if usable else None
            target: CandidateGroup | None = None
            if vector is not None and self.groups:
                best_distance = math.inf
                for group in self.groups:
                    if not group.features:
                        continue
                    matrix = np.stack(group.features).astype(np.float32, copy=False)
                    values = np.asarray(distance(matrix, vector[None, :]), dtype=np.float64).reshape(-1)
                    finite = values[np.isfinite(values)]
                    if finite.size and float(finite.min()) < best_distance:
                        best_distance = float(finite.min())
                        target = group
                if not (best_distance <= self.threshold):
                    target = None
            if target is None:
                target = CandidateGroup(character_key(self.fingerprint, key), key)
                self.groups.append(target)
            record["character_key"] = target.key
            target.members.append(record)
            if vector is not None:
                target.features.append(vector)
            self._seen.add(key)
            assigned.append(target.key)
        return assigned

    def public(self, preview_url: Callable[[Mapping[str, Any]], str], *, member_limit: int = 8) -> list[dict[str, Any]]:
        """Return groups ordered for display; keys, not positions, identify them."""

        ordered = sorted(
            self.groups,
            key=lambda group: (-len({int(m["pts"]) for m in group.members}), min(float(m.get("time", 0.0)) for m in group.members), group.key),
        )
        result: list[dict[str, Any]] = []
        for group in ordered:
            representative = group.representative()
            members = sorted(group.members, key=lambda item: (-representative_score(item), float(item.get("time", 0.0)), item["detection_key"]))
            result.append({
                "character_key": group.key,
                "frame_count": len({int(m["pts"]) for m in group.members}),
                "member_count": len(group.members),
                "representative": _public_member(representative, preview_url),
                "members": [_public_member(item, preview_url) for item in members[:max(1, member_limit)]],
                "has_more_members": len(members) > member_limit,
            })
        return result


def _public_member(member: Mapping[str, Any], preview_url: Callable[[Mapping[str, Any]], str]) -> dict[str, Any]:
    return {
        "detection_key": member["detection_key"],
        "character_key": member.get("character_key"),
        "pts": int(member["pts"]),
        "time": float(member.get("time", 0.0)),
        "box": [float(value) for value in member["box"]],
        "kind": str(member.get("kind", "person")),
        "score": None if member.get("score") is None else float(member["score"]),
        "preview_url": preview_url(member),
    }


def model_threshold(models: Any) -> float:
    value = getattr(models, "threshold", None)
    if callable(value):
        try:
            value = value()
        except Exception:
            value = None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return DEFAULT_IDENTITY_THRESHOLD
    return number if math.isfinite(number) and number > 0 else DEFAULT_IDENTITY_THRESHOLD


def embed_crops(models: Any, crops: Iterable[Any]) -> tuple[np.ndarray | None, Callable[..., Any] | None]:
    """Embed crops through the existing CCIP contract when it is available."""

    embed = getattr(models, "embed", None)
    distance = getattr(models, "distance", None)
    crops = list(crops)
    if not callable(embed) or not callable(distance) or not crops:
        return None, None
    features = np.asarray(embed(crops), dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(crops):
        raise RuntimeError(f"CCIP embed returned shape {features.shape}; expected ({len(crops)}, D)")
    return features, distance
