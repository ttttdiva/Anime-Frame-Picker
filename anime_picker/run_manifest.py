"""Private run manifests for the interactive character-selection flow.

A run manifest records exactly what the user selected (stable character keys
and the digests of their reference crops) and every analysis setting that can
change which cuts are found.  The config hash is computed from those records,
so a resume or rerun with the same selection and thresholds maps to the same
run, while any change to the selection, ``cut_quality_min``, identity/tag/
quality thresholds, sampling, or conditions yields a different run that never
reuses older results as complete.

Manifests live under the private cache (``<cache>/runs/<run_id>/``), never in
the public ``outputs/<video>/images/`` directory.  Export-only options such as
logo removal and the output root are recorded but are not part of the hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RUN_MANIFEST_REVISION = "ui-character-run-v1"
# Settings that influence analysis results.  ``cut_quality_min`` is included
# deliberately: changing the quality floor must invalidate completed results.
ANALYSIS_SETTING_KEYS = (
    "selection_mode",
    "start",
    "end",
    "coarse_step",
    "auto_sample_step",
    "dense_step",
    "candidate_count",
    "identity_threshold",
    "tag_threshold",
    "quality_min",
    "cut_quality_min",
    "rescue",
    "identity_weight",
    "condition_weight",
    "quality_weight",
)
EXPORT_SETTING_KEYS = ("logo_removal", "logo_profiles", "output_root")
TERMINAL_COMPLETE = "completed"
RESUMABLE_STATES = ("running", "queued", "cancelled", "interrupted", "error")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (list, tuple)):
        return [_round(item) for item in value]
    return value


def selection_record(selection_mode: str, references: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Order-independent record of the selected characters and their crops."""

    characters: dict[str, list[dict[str, Any]]] = {}
    for item in references:
        key = str(item.get("character_key") or "_unkeyed")
        characters.setdefault(key, []).append({
            "sha256": str(item.get("sha256") or ""),
            "box": _round(item.get("box")),
            "source_pts": item.get("source_pts"),
        })
    ordered = []
    for key in sorted(characters):
        refs = sorted(characters[key], key=lambda ref: (ref["sha256"], _canonical(ref["box"]), str(ref["source_pts"])))
        ordered.append({"character_key": key, "references": refs})
    return {"mode": str(selection_mode), "characters": ordered}


def analysis_record(settings: Mapping[str, Any], wanted: Iterable[Mapping[str, Any]], avoided: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    def tags(items: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
        values = [{"tag": str(item.get("tag")), "scope": str(item.get("scope", "subject"))} for item in items]
        return sorted(values, key=lambda item: (item["tag"], item["scope"]))

    return {
        "settings": {key: _round(settings.get(key)) for key in ANALYSIS_SETTING_KEYS},
        "wanted": tags(wanted),
        "avoided": tags(avoided),
    }


def config_hash(fingerprint: str, selection: Mapping[str, Any], analysis: Mapping[str, Any]) -> str:
    payload = {
        "revision": RUN_MANIFEST_REVISION,
        "fingerprint": str(fingerprint),
        "selection": selection,
        "analysis": analysis,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:16]


def run_id_for(fingerprint: str, digest: str) -> str:
    source = hashlib.sha256(str(fingerprint).encode("utf-8")).hexdigest()[:12]
    return f"{source}_{digest}"


def source_prefix(fingerprint: str) -> str:
    return hashlib.sha256(str(fingerprint).encode("utf-8")).hexdigest()[:12] + "_"


def manifest_path(root: Path, run_id: str) -> Path:
    return Path(root) / run_id / "run-manifest.json"


def read_manifest(root: Path, run_id: str) -> dict[str, Any] | None:
    path = manifest_path(root, run_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        return None
    return value


def write_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
    """Atomically replace one run manifest."""

    path = manifest_path(root, str(manifest["run_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(manifest)
    data["updated_at"] = utc_now()
    handle, temp_name = tempfile.mkstemp(prefix=".run-manifest-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def verified_complete(manifest: Mapping[str, Any] | None, config: str, allowed_root: Path) -> bool:
    """Return whether a manifest is a complete result for exactly ``config``.

    A completed manifest is reused only when its revision and config hash
    match and every stored preview/crop file still exists under the private
    result root.  Anything else is re-executed instead of trusted.
    """

    if not isinstance(manifest, Mapping):
        return False
    if manifest.get("revision") != RUN_MANIFEST_REVISION or manifest.get("config_hash") != config:
        return False
    if manifest.get("status") != TERMINAL_COMPLETE:
        return False
    candidates = manifest.get("candidates")
    result_root = manifest.get("result_root")
    if not isinstance(candidates, list) or not isinstance(result_root, str) or not result_root:
        return False
    root = Path(result_root)
    if not _inside(root, allowed_root):
        return False
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            return False
        for key in ("_preview_file", "_crop_file"):
            value = candidate.get(key)
            if value is None:
                continue
            path = Path(str(value))
            if not _inside(path, root) or not path.is_file():
                return False
    return True


def list_manifests(root: Path, fingerprint: str) -> list[dict[str, Any]]:
    prefix = source_prefix(fingerprint)
    result: list[dict[str, Any]] = []
    base = Path(root)
    if not base.is_dir():
        return result
    for child in base.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        manifest = read_manifest(base, child.name)
        if manifest is not None and manifest.get("fingerprint") == fingerprint:
            result.append(manifest)
    result.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return result


def public_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    selection = manifest.get("selection") if isinstance(manifest.get("selection"), Mapping) else {}
    characters = selection.get("characters") if isinstance(selection.get("characters"), list) else []
    return {
        "run_id": manifest.get("run_id"),
        "config_hash": manifest.get("config_hash"),
        "status": manifest.get("status"),
        "attempts": manifest.get("attempts", 0),
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "selection_mode": selection.get("mode"),
        "characters": [
            {"character_key": item.get("character_key"), "references": len(item.get("references") or [])}
            for item in characters
            if isinstance(item, Mapping)
        ],
        "analysis": manifest.get("analysis"),
        "export_settings": manifest.get("export_settings"),
        "candidate_count": len(manifest.get("candidates") or []) if manifest.get("status") == TERMINAL_COMPLETE else None,
        "error": manifest.get("error"),
    }
