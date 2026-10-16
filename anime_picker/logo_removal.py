"""Fixed-logo detection and optional Anime-LaMa inpainting.

The logo remover is deliberately a small, offline-first boundary around the
TorchScript Anime-LaMa model.  Profiles describe a logo in reference-frame
coordinates and keep their template/mask assets in a user cache.  A model is
loaded only after a registered profile has matched the input frame; an
unmatched frame is returned byte-for-byte outside the reported edit mask.

The implementation ports the inpainting part of the original fixed-watermark
script without importing that script or depending on its workspace paths:
RGB ``uint8`` input, a binary mask, ``BORDER_REFLECT_101`` padding to a
multiple of eight, ``torch.inference_mode()``, and an 11x11 Gaussian blend
with sigma 3.

Profile JSON schema (version 1)
-------------------------------

``reference_size`` and ``roi`` use full-frame reference pixels.  The
``mask.rectangles`` and ``mask.polygons`` entries use full-frame reference
pixels too (``mask_coordinates`` is explicit in the schema).  A
``custom_mask`` is an ROI-local or reference-frame PNG whose non-zero pixels
are the inpaint mask.  ``template_image`` is normally an ROI-local PNG; a
reference-frame template is accepted as a convenience and cropped to ``roi``.

    {
      "version": 1,
      "id": "museasia",
      "name": "MUSE ASIA",
      "reference_size": [3840, 2160],
      "roi": [3000, 0, 3840, 260],
      "mask_coordinates": "full_frame",
      "mask": {"rectangles": [[3270, 30, 3500, 120]], "polygons": []},
      "template_image": "template.png",
      "custom_mask": "mask.png",
      "detector": {"threshold": 0.62, "aspect_tolerance": 0.025}
    }

The schema is intentionally kept in this module so profile creation workers
can use the same validation rules without adding another package dependency.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _datetime
import hashlib
import io
import json
import math
import os
import re
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


PROFILE_SCHEMA_VERSION = 1
MODEL_FILENAME = "anime-manga-big-lama.pt"
MODEL_URL = "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt"
# This is the pinned release currently used by the original workflow.  Keep
# both digests: MD5 is the historical operator check and SHA-256 is the
# integrity check used by prepare()/status().
MODEL_SIZE = 205_717_870
MODEL_MD5 = "29f284f36a0a510bcacf39ecf4c4d54f"
MODEL_SHA256 = "479d3afdcb7ed2fd944ed4ebcc39ca45b33491f0f2e43eb1000bd623cfb41823"
DEFAULT_IMAGE_MODEL_ROOT = Path(r"D:\AI\models\Cold_F\image")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Model status is queried by the UI and can be called repeatedly while the
# server is running.  Keep the expensive digest pair in-process, keyed by the
# resolved path and the file's current size/mtime.  A changed file naturally
# misses the entry and gets a complete fresh verification.
_MODEL_DIGEST_CACHE: dict[Path, tuple[int, int, str, str]] = {}
_MODEL_DIGEST_LOCK = threading.RLock()


class LogoError(RuntimeError):
    """Base error for logo profile and model failures."""


class ProfileError(LogoError, ValueError):
    """A profile is malformed or references an unsafe/missing asset."""


class LogoModelUnavailable(LogoError):
    """A matched logo needs an unavailable or invalid model."""


class GPURequiredError(LogoModelUnavailable):
    """GPU execution was requested but CUDA is not available."""


class PrepareError(LogoError):
    """The explicit model preparation operation failed."""


class LogoResult:
    """Result returned by :meth:`LogoProcessor.process`.

    ``LogoResult`` is intentionally a tiny value object.  It also supports
    tuple unpacking for callers that prefer ``image, metadata, mask = result``
    while keeping named attributes for the service layer.
    """

    __slots__ = ("image", "metadata", "edit_mask")

    def __init__(self, image: Image.Image, metadata: Mapping[str, Any], edit_mask: Image.Image):
        self.image = image
        self.metadata = dict(metadata)
        self.edit_mask = edit_mask

    def __iter__(self):
        yield self.image
        yield self.metadata
        yield self.edit_mask

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"LogoResult(image={self.image!r}, metadata={self.metadata!r}, edit_mask={self.edit_mask!r})"


def _default_profile_root() -> Path:
    value = os.environ.get("ANIME_PICKER_LOGO_PROFILE_ROOT", "").strip()
    # Profiles are calibration state, not disposable frame/cache data.  Keep
    # them under the private project-local ``.local`` directory by default;
    # callers can point this at an OS/user cache with the environment variable.
    project_root = Path(__file__).resolve().parents[1]
    return Path(value) if value else project_root / ".local" / "logo-profiles"


def _model_directory(model_root: Path | str | None) -> Path:
    if model_root is None:
        value = os.environ.get("ANIME_PICKER_LOGO_MODEL_ROOT", "").strip()
        if value:
            return Path(value)
        value = os.environ.get("ANIME_PICKER_MODEL_ROOT", "").strip()
        return Path(value) if value else DEFAULT_IMAGE_MODEL_ROOT
    return Path(model_root)


def _model_path(model_root: Path | str | None) -> Path:
    root = _model_directory(model_root)
    if root.suffix.lower() == ".pt":
        return root
    if root.name.lower() == "anime-lama" or (root / MODEL_FILENAME).is_file():
        return root / MODEL_FILENAME
    # The public model_root contract is the main image-model root (for
    # example D:\\AI\\models\\Cold_F\\image), where Anime-LaMa is kept in
    # its own subdirectory.  The existing-file check above keeps a direct
    # model directory convenient for explicit callers and tests.
    return root / "anime-lama" / MODEL_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_digests(path: Path) -> tuple[int, str, str]:
    """Return size/MD5/SHA-256, reusing a digest for an unchanged file.

    The first call always computes both digests.  Later calls reuse the pair
    only when the resolved path, size, and nanosecond mtime are unchanged.  A
    second stat after hashing protects against a replacement while the file is
    being read; in that case the changed file is hashed again on the next
    pass.
    """

    resolved = path.resolve()
    with _MODEL_DIGEST_LOCK:
        stat_before = resolved.stat()
        key = resolved
        cached = _MODEL_DIGEST_CACHE.get(key)
        if cached is not None:
            cached_size, cached_mtime_ns, cached_md5, cached_sha256 = cached
            if cached_size == stat_before.st_size and cached_mtime_ns == stat_before.st_mtime_ns:
                return cached_size, cached_md5, cached_sha256

        md5 = _md5(resolved)
        sha256 = _sha256(resolved)
        stat_after = resolved.stat()
        if stat_after.st_size != stat_before.st_size or stat_after.st_mtime_ns != stat_before.st_mtime_ns:
            # A concurrent replacement means the bytes just read are not a
            # stable identity.  Drop any prior entry and let the next status
            # call perform a complete verification of the new file.
            _MODEL_DIGEST_CACHE.pop(key, None)
            return stat_after.st_size, md5, sha256
        _MODEL_DIGEST_CACHE[key] = (stat_after.st_size, stat_after.st_mtime_ns, md5, sha256)
        return stat_after.st_size, md5, sha256


def _invalidate_model_digest(path: Path) -> None:
    with _MODEL_DIGEST_LOCK:
        _MODEL_DIGEST_CACHE.pop(path.resolve(), None)


def _json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _write_model_manifest(path: Path, *, url: str, size: int, md5: str, sha256: str) -> dict[str, Any]:
    """Atomically write the local model manifest beside a verified model."""

    manifest_path = path.parent / "manifest.json"
    current = _json_value(manifest_path)
    try:
        current_size = int(current.get("size", -1)) if isinstance(current, Mapping) else -1
    except (TypeError, ValueError):
        current_size = -1
    if (
        isinstance(current, Mapping)
        and current.get("filename") == path.name
        and current.get("url") == url
        and current_size == size
        and str(current.get("md5", "")).lower() == md5
        and str(current.get("sha256", "")).lower() == sha256
    ):
        return dict(current)
    manifest = {
        "version": 1,
        "filename": path.name,
        "url": url,
        "size": size,
        "md5": md5,
        "sha256": sha256,
        "prepared_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
    }
    fd_manifest, manifest_name = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd_manifest, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(manifest_name, manifest_path)
    finally:
        try:
            os.unlink(manifest_name)
        except FileNotFoundError:
            pass
    return manifest


def _number(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ProfileError(f"{name} must contain numbers") from error
    if not math.isfinite(result):
        raise ProfileError(f"{name} must contain finite numbers")
    return result


def _int_number(value: Any, *, name: str) -> int:
    result = _number(value, name=name)
    rounded = int(round(result))
    if abs(result - rounded) > 1e-6:
        raise ProfileError(f"{name} must contain integer pixel coordinates")
    return rounded


def _box(value: Any, *, name: str, width: int, height: int, allow_edge: bool = True) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ProfileError(f"{name} must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (_int_number(item, name=name) for item in value)
    if x1 <= x0 or y1 <= y0:
        raise ProfileError(f"{name} must have positive area")
    if allow_edge:
        valid = 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
    else:
        valid = 0 <= x0 < x1 < width and 0 <= y0 < y1 < height
    if not valid:
        raise ProfileError(f"{name} is outside reference_size")
    return [x0, y0, x1, y1]


def _polygon(value: Any, *, name: str, width: int, height: int) -> list[list[int]]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ProfileError(f"{name} must contain at least three [x, y] points")
    result: list[list[int]] = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ProfileError(f"{name}[{index}] must be [x, y]")
        x = _int_number(point[0], name=f"{name}[{index}]")
        y = _int_number(point[1], name=f"{name}[{index}]")
        if not (0 <= x <= width and 0 <= y <= height):
            raise ProfileError(f"{name}[{index}] is outside reference_size")
        result.append([x, y])
    return result


def _safe_relative_asset(profile_dir: Path, value: Any, *, field: str, required: bool) -> Path | None:
    if value in (None, ""):
        if required:
            raise ProfileError(f"{field} is required")
        return None
    if not isinstance(value, str):
        raise ProfileError(f"{field} must be a relative asset path")
    candidate = Path(value)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ProfileError(f"{field} must stay inside the profile directory")
    resolved = (profile_dir / candidate).resolve()
    try:
        resolved.relative_to(profile_dir.resolve())
    except ValueError as error:
        raise ProfileError(f"{field} must stay inside the profile directory") from error
    if required and not resolved.is_file():
        raise ProfileError(f"{field} asset does not exist: {candidate}")
    return resolved if resolved.is_file() else None


def _validate_profile(raw: Mapping[str, Any], profile_dir: Path | None = None) -> dict[str, Any]:
    """Validate and normalize a profile mapping.

    This function is shared by ``ProfileStore`` and direct ``process`` calls.
    Direct profiles may carry an in-memory ``template``/``custom_mask`` value;
    assets loaded from disk are resolved and recorded under private keys.
    """

    if not isinstance(raw, Mapping):
        raise ProfileError("logo profile must be an object")
    version = raw.get("version", raw.get("schema_version"))
    if version != PROFILE_SCHEMA_VERSION:
        raise ProfileError(f"unsupported logo profile version: {version!r}")
    profile_id = raw.get("id")
    if not isinstance(profile_id, str) or not _ID_RE.fullmatch(profile_id):
        raise ProfileError("profile id must contain 1-64 letters, numbers, '.', '_' or '-'")
    if profile_dir is not None and profile_dir.name != profile_id:
        raise ProfileError("profile directory name must match profile id")
    name = raw.get("name", profile_id)
    if not isinstance(name, str) or not name.strip():
        raise ProfileError("profile name must be a non-empty string")

    reference_size = raw.get("reference_size")
    if not isinstance(reference_size, (list, tuple)) or len(reference_size) != 2:
        raise ProfileError("reference_size must be [width, height]")
    ref_w = _int_number(reference_size[0], name="reference_size")
    ref_h = _int_number(reference_size[1], name="reference_size")
    if ref_w <= 0 or ref_h <= 0:
        raise ProfileError("reference_size must be positive")
    roi = _box(raw.get("roi"), name="roi", width=ref_w, height=ref_h)

    coordinates = raw.get("mask_coordinates", "full_frame")
    if coordinates != "full_frame":
        raise ProfileError("mask_coordinates must be 'full_frame'")
    mask_raw = raw.get("mask", {})
    if mask_raw is None:
        mask_raw = {}
    if not isinstance(mask_raw, Mapping):
        raise ProfileError("mask must be an object")
    rectangles_raw = mask_raw.get("rectangles", raw.get("mask_rectangles", []))
    polygons_raw = mask_raw.get("polygons", raw.get("mask_polygons", []))
    if rectangles_raw is None:
        rectangles_raw = []
    if polygons_raw is None:
        polygons_raw = []
    if not isinstance(rectangles_raw, (list, tuple)) or not isinstance(polygons_raw, (list, tuple)):
        raise ProfileError("mask rectangles/polygons must be arrays")
    rectangles = [_box(item, name="mask.rectangles", width=ref_w, height=ref_h) for item in rectangles_raw]
    polygons = [_polygon(item, name="mask.polygons", width=ref_w, height=ref_h) for item in polygons_raw]
    custom_mask_declared = raw.get("custom_mask", raw.get("mask_image"))
    if not rectangles and not polygons and custom_mask_declared is None:
        raise ProfileError("profile must define mask rectangles, polygons, or custom_mask")

    detector_raw = raw.get("detector", {})
    if detector_raw is None:
        detector_raw = {}
    if not isinstance(detector_raw, Mapping):
        raise ProfileError("detector must be an object")
    threshold = _number(detector_raw.get("threshold", raw.get("threshold", 0.62)), name="detector.threshold")
    aspect_tolerance = _number(detector_raw.get("aspect_tolerance", 0.025), name="detector.aspect_tolerance")
    if not 0.0 <= threshold <= 1.0:
        raise ProfileError("detector.threshold must be between 0 and 1")
    if not 0.0 <= aspect_tolerance <= 0.5:
        raise ProfileError("detector.aspect_tolerance must be between 0 and 0.5")

    normalized: dict[str, Any] = {
        "version": PROFILE_SCHEMA_VERSION,
        "id": profile_id,
        "name": name.strip(),
        "reference_size": [ref_w, ref_h],
        "roi": roi,
        "mask_coordinates": "full_frame",
        "mask": {"rectangles": rectangles, "polygons": polygons},
        "detector": {
            "threshold": threshold,
            "aspect_tolerance": aspect_tolerance,
            "edge_weight": _number(detector_raw.get("edge_weight", 0.65), name="detector.edge_weight"),
            "detail_weight": _number(detector_raw.get("detail_weight", 0.35), name="detector.detail_weight"),
        },
    }
    if normalized["detector"]["edge_weight"] < 0 or normalized["detector"]["detail_weight"] < 0:
        raise ProfileError("detector weights must be non-negative")

    if profile_dir is not None:
        profile_dir = profile_dir.resolve()
        template_value = raw.get("template_image", raw.get("template"))
        custom_value = raw.get("custom_mask", raw.get("mask_image"))
        template_path = _safe_relative_asset(profile_dir, template_value, field="template_image", required=True)
        custom_path = _safe_relative_asset(
            profile_dir,
            custom_value,
            field="custom_mask",
            required=custom_value not in (None, ""),
        )
        assert template_path is not None
        normalized["template_image"] = Path(template_value).as_posix()
        if custom_path is not None:
            normalized["custom_mask"] = Path(custom_value).as_posix()
        normalized["_profile_dir"] = str(profile_dir)
        normalized["_template_path"] = str(template_path)
        if custom_path is not None:
            normalized["_custom_mask_path"] = str(custom_path)
    else:
        # Preserve in-memory template values for worker/tests.  A direct path
        # is retained and resolved later relative to its parent only when it is
        # already absolute; this keeps direct, trusted profiles useful while
        # ProfileStore remains strict about relative assets.
        for field in ("template_image", "custom_mask"):
            if field in raw:
                normalized[field] = raw[field]
        if "template" in raw and "template_image" not in normalized:
            normalized["template_image"] = raw["template"]
        if "mask_image" in raw and "custom_mask" not in normalized:
            normalized["custom_mask"] = raw["mask_image"]
        for private_key in ("_profile_dir", "_template_path", "_custom_mask_path"):
            if private_key in raw:
                normalized[private_key] = raw[private_key]
        if "template" in raw:
            normalized["_template_value"] = raw["template"]
        elif "template_image" in raw:
            normalized["_template_value"] = raw["template_image"]
        if "custom_mask" in raw:
            normalized["_custom_mask_value"] = raw["custom_mask"]
        elif "mask_image" in raw:
            normalized["_custom_mask_value"] = raw["mask_image"]

    return normalized


class ProfileStore:
    """Load validated logo profiles from a user-local cache.

    The repository contains no raw logo crops.  Each profile lives under
    ``<root>/<id>/profile.json`` with its template and optional mask assets.
    """

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else _default_profile_root()

    def _profile_dirs(self, ids: Iterable[str] | None = None) -> list[Path]:
        if ids is None:
            if not self.root.is_dir():
                return []
            return sorted((item for item in self.root.iterdir() if item.is_dir()), key=lambda path: path.name.lower())
        result = []
        for profile_id in ids:
            if not isinstance(profile_id, str) or not _ID_RE.fullmatch(profile_id):
                raise ProfileError(f"invalid profile id: {profile_id!r}")
            result.append(self.root / profile_id)
        return result

    def load(self, ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        for directory in self._profile_dirs(ids):
            path = directory / "profile.json"
            if not path.is_file():
                if ids is not None:
                    raise ProfileError(f"profile.json is missing: {path}")
                continue
            raw = _json_value(path)
            if not isinstance(raw, Mapping):
                if ids is not None:
                    raise ProfileError(f"invalid profile JSON: {path}")
                continue
            try:
                profiles.append(_validate_profile(raw, directory))
            except ProfileError:
                if ids is not None:
                    raise
        return profiles

    def list(self) -> list[dict[str, Any]]:
        """Return validated, JSON-serializable profile metadata."""

        result = []
        for profile in self.load():
            public = {key: copy.deepcopy(value) for key, value in profile.items() if not key.startswith("_")}
            # A public listing should reveal where a template comes from only
            # through its profile-relative filename, never an absolute cache
            # path.  Consumers needing assets call load().
            result.append(public)
        return result


def _as_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_asset(value: Any, *, profile: Mapping[str, Any], template: bool = False) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim == 2:
            mode = "L"
        elif array.ndim == 3 and array.shape[2] in (3, 4):
            mode = "RGBA" if array.shape[2] == 4 else "RGB"
        else:
            raise ProfileError("in-memory logo asset must be HxW or HxWx3/4")
        return Image.fromarray(array.astype(np.uint8), mode=mode)
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return Image.open(io.BytesIO(bytes(value))).copy()
        except Exception as error:
            raise ProfileError("invalid in-memory logo asset") from error
    if isinstance(value, (str, os.PathLike)):
        path_value = profile.get("_template_path" if template else "_custom_mask_path") or value
        path = Path(path_value)
        if not path.is_absolute() and profile.get("_profile_dir"):
            path = Path(str(profile["_profile_dir"])) / path
        if not path.is_file():
            return None
        try:
            return Image.open(path).copy()
        except Exception as error:
            raise ProfileError(f"unable to read logo asset: {path}") from error
    raise ProfileError("unsupported logo asset value")


def _resize_mask(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST), dtype=np.uint8)


def build_mask(size: Sequence[int], roi: Sequence[int], area: Sequence[int]) -> np.ndarray:
    """Build the legacy binary ROI mask from full-frame coordinates."""

    if len(size) != 2 or len(roi) != 4 or len(area) != 4:
        raise ValueError("size, roi, and area must contain two/four coordinates")
    width, height = (int(value) for value in size)
    x0, y0, x1, y1 = (int(value) for value in roi)
    left, top, right, bottom = (int(value) for value in area)
    if right <= left or bottom <= top or not (x0 <= left < right <= x1 and y0 <= top < bottom <= y1):
        raise ValueError("mask coordinates are outside ROI or empty")
    if width != x1 - x0 or height != y1 - y0:
        raise ValueError("ROI size is invalid")
    result = np.zeros((height, width), dtype=np.uint8)
    result[top - y0 : bottom - y0, left - x0 : right - x0] = 255
    return result


def _scaled_box(box: Sequence[int], sx: float, sy: float, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(width, int(round(float(box[0]) * sx))))
    y0 = max(0, min(height, int(round(float(box[1]) * sy))))
    x1 = max(0, min(width, int(round(float(box[2]) * sx))))
    y1 = max(0, min(height, int(round(float(box[3]) * sy))))
    return x0, y0, x1, y1


def _geometry(profile: Mapping[str, Any], image_size: tuple[int, int]) -> tuple[tuple[int, int, int, int], np.ndarray, str | None]:
    width, height = image_size
    ref_w, ref_h = (int(value) for value in profile["reference_size"])
    current_aspect = width / max(1, height)
    reference_aspect = ref_w / max(1, ref_h)
    tolerance = float(profile.get("detector", {}).get("aspect_tolerance", 0.025))
    if abs(current_aspect / reference_aspect - 1.0) > tolerance:
        return (0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8), "unsupported aspect ratio"
    sx, sy = width / ref_w, height / ref_h
    x0, y0, x1, y1 = _scaled_box(profile["roi"], sx, sy, width, height)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0), np.zeros((0, 0), dtype=np.uint8), "ROI is outside image"
    roi_w, roi_h = x1 - x0, y1 - y0
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    shape = profile.get("mask", {})
    for rectangle in shape.get("rectangles", []):
        rx0, ry0, rx1, ry1 = _scaled_box(rectangle, sx, sy, width, height)
        rx0, rx1 = max(x0, rx0), min(x1, rx1)
        ry0, ry1 = max(y0, ry0), min(y1, ry1)
        if rx1 > rx0 and ry1 > ry0:
            mask[ry0 - y0 : ry1 - y0, rx0 - x0 : rx1 - x0] = 255
    if shape.get("polygons"):
        # Keep a reference to the PIL image while drawing; an ephemeral image
        # would otherwise discard the polygon result.
        mask_image = Image.fromarray(mask, mode="L")
        draw = ImageDraw.Draw(mask_image)
        for polygon in shape["polygons"]:
            points = [
                (int(round(float(x) * sx)) - x0, int(round(float(y) * sy)) - y0)
                for x, y in polygon
            ]
            draw.polygon(points, fill=255)
        mask = np.asarray(mask_image, dtype=np.uint8)

    custom_value = profile.get("_custom_mask_value", profile.get("custom_mask"))
    if profile.get("_custom_mask_path"):
        custom_value = profile["_custom_mask_path"]
    custom = _load_asset(custom_value, profile=profile)
    if custom is not None:
        ref_size = (ref_w, ref_h)
        roi_ref_size = (int(profile["roi"][2] - profile["roi"][0]), int(profile["roi"][3] - profile["roi"][1]))
        if custom.size == ref_size:
            full = _resize_mask(custom, (width, height))
            mask = np.maximum(mask, np.where(full[y0:y1, x0:x1] > 0, 255, 0).astype(np.uint8))
        elif custom.size == roi_ref_size:
            local = _resize_mask(custom, (roi_w, roi_h))
            mask = np.maximum(mask, np.where(local > 0, 255, 0).astype(np.uint8))
        else:
            local = _resize_mask(custom, (roi_w, roi_h))
            mask = np.maximum(mask, np.where(local > 0, 255, 0).astype(np.uint8))
    if not np.any(mask):
        return (x0, y0, x1, y1), mask, "empty logo mask"
    return (x0, y0, x1, y1), mask, None


def _template_for_roi(profile: Mapping[str, Any], roi: tuple[int, int, int, int]) -> Image.Image | None:
    template_value = profile.get("_template_value", profile.get("template_image"))
    if profile.get("_template_path"):
        template_value = profile["_template_path"]
    template = _load_asset(template_value, profile=profile, template=True)
    if template is None:
        return None
    ref_w, ref_h = (int(value) for value in profile["reference_size"])
    if template.size == (ref_w, ref_h):
        template = template.crop(tuple(int(value) for value in profile["roi"]))
    return template.convert("RGB").resize((roi[2] - roi[0], roi[3] - roi[1]), Image.Resampling.BILINEAR)


def _ncc(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float32)[mask]
    b = np.asarray(second, dtype=np.float32)[mask]
    if a.size < 3 or b.size != a.size:
        return 0.0
    a = a - float(a.mean())
    b = b - float(b.mean())
    da = float(np.sqrt(np.dot(a, a)))
    db = float(np.sqrt(np.dot(b, b)))
    if da <= 1e-6 or db <= 1e-6:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / (da * db))))


def _gray(image: np.ndarray) -> np.ndarray:
    # ITU-R BT.601 weights match the usual OpenCV/Pillow grayscale conversion
    # closely enough for a detector while keeping matching dependency-light.
    return image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114


def _blur(gray: np.ndarray) -> np.ndarray:
    try:
        import cv2

        return cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    except Exception:
        return np.asarray(Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)


def _edge(gray: np.ndarray) -> np.ndarray:
    # Gradient magnitude is less brittle than a fixed Canny threshold when
    # source videos are recompressed or their overall brightness changes.
    gy, gx = np.gradient(gray.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)


def _match_score(current: np.ndarray, template: np.ndarray, mask: np.ndarray, detector: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    active = mask > 0
    if current.shape[:2] != template.shape[:2] or not np.any(active):
        return 0.0, {"edge": 0.0, "detail": 0.0, "score": 0.0}
    current_gray = _gray(current)
    template_gray = _gray(template)
    current_detail = current_gray - _blur(current_gray)
    template_detail = template_gray - _blur(template_gray)
    current_edge = _edge(current_gray)
    template_edge = _edge(template_gray)
    edge_score = _ncc(current_edge, template_edge, active)
    detail_score = _ncc(current_detail, template_detail, active)
    edge_weight = max(0.0, float(detector.get("edge_weight", 0.65)))
    detail_weight = max(0.0, float(detector.get("detail_weight", 0.35)))
    total = edge_weight + detail_weight
    score = (edge_weight * edge_score + detail_weight * detail_score) / total if total else 0.0
    return float(max(0.0, min(1.0, score))), {
        "edge": float(edge_score),
        "detail": float(detail_score),
        "score": float(score),
    }


def _inpaint_patch(patch: np.ndarray, mask: np.ndarray, model: Any, device: Any, torch_mod: Any) -> np.ndarray:
    """Run Anime-LaMa on one RGB uint8 patch, preserving legacy semantics."""

    import cv2

    height, width = patch.shape[:2]
    pad_y = (-height) % 8
    pad_x = (-width) % 8
    padded = cv2.copyMakeBorder(patch, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT_101)
    padded_mask = cv2.copyMakeBorder(mask, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)
    image = torch_mod.from_numpy(np.ascontiguousarray(padded)).to(device, dtype=torch_mod.float32)
    image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
    binary = torch_mod.from_numpy(np.ascontiguousarray(padded_mask)).to(device, dtype=torch_mod.float32)
    binary = binary.unsqueeze(0).unsqueeze(0) / 255.0
    with torch_mod.inference_mode():
        output = model(image, binary)[0].permute(1, 2, 0).cpu().numpy()
    output = (np.clip(output[:height, :width], 0, 1) * 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(mask, (11, 11), 3).astype(np.float32)[:, :, None] / 255.0
    return np.clip(patch.astype(np.float32) * (1 - alpha) + output.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


# Public alias useful for the server's lightweight test seam and for callers
# that want to verify the exact legacy kernel independently.
inpaint_patch = _inpaint_patch


class LogoProcessor:
    """Detect registered fixed logos and remove them with Anime-LaMa."""

    model_filename = MODEL_FILENAME
    model_url = MODEL_URL
    model_md5 = MODEL_MD5
    model_sha256 = MODEL_SHA256
    model_size = MODEL_SIZE

    def __init__(self, model_root: Path | str | None = None, require_gpu: bool = True):
        self.model_root = _model_directory(model_root)
        self.require_gpu = bool(require_gpu)
        self._model: Any = None
        self._torch: Any = None
        self._device: Any = None
        self._model_metadata: dict[str, Any] | None = None

    @staticmethod
    def status(model_root: Path | str | None = None) -> dict[str, Any]:
        path = _model_path(model_root)
        base: dict[str, Any] = {
            "ready": False,
            "path": str(path),
            "filename": path.name,
            "expected_size": MODEL_SIZE,
            "expected_md5": MODEL_MD5,
            "expected_sha256": MODEL_SHA256,
        }
        if not path.is_file():
            base["details"] = f"Anime-LaMa model is missing: {path}; run `python -m anime_picker.logo_removal prepare`"
            return base
        try:
            size, md5, sha256 = _verified_digests(path)
            base["size"] = size
            base["md5"] = md5
            base["sha256"] = sha256
        except OSError as error:
            base["details"] = f"unable to inspect Anime-LaMa model: {error}"
            return base
        if size != MODEL_SIZE or base["md5"] != MODEL_MD5 or base["sha256"] != MODEL_SHA256:
            base["details"] = "Anime-LaMa model exists but its size or checksum does not match the pinned release"
            return base
        manifest = _json_value(path.parent / "manifest.json")
        base["manifest"] = manifest if isinstance(manifest, Mapping) else None
        base["ready"] = True
        base["details"] = "Anime-LaMa model is prepared and verified"
        return base

    @staticmethod
    def prepare(model_root: Path | str | None = None, *, url: str = MODEL_URL) -> dict[str, Any]:
        """Explicitly download and verify the pinned model.

        Inference never calls this method.  The download is written to a
        temporary file and atomically moved into place only after all pinned
        checks pass.
        """

        path = _model_path(model_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        # An explicit prepare operation is safe to repeat.  If the pinned
        # bytes are already present, write/update the local manifest and
        # return without touching the network.  A bad existing file falls
        # through to the temporary download below and is never overwritten
        # until the replacement passes every integrity check.
        existing = LogoProcessor.status(model_root)
        if existing.get("ready"):
            size = int(existing.get("size", MODEL_SIZE))
            md5 = str(existing.get("md5", MODEL_MD5)).lower()
            sha256 = str(existing.get("sha256", MODEL_SHA256)).lower()
            _write_model_manifest(path, url=url, size=size, md5=md5, sha256=sha256)
            return LogoProcessor.status(model_root)

        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".download", dir=str(path.parent))
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            try:
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
            except Exception as error:
                raise PrepareError(f"unable to download Anime-LaMa from {url}: {error}") from error
            size = temporary.stat().st_size
            md5 = _md5(temporary)
            sha256 = _sha256(temporary)
            if size != MODEL_SIZE or md5 != MODEL_MD5 or sha256 != MODEL_SHA256:
                raise PrepareError(
                    "downloaded Anime-LaMa bytes do not match the pinned release "
                    f"(size={size}, md5={md5}, sha256={sha256})"
                )
            os.replace(temporary, path)
            _invalidate_model_digest(path)
            _write_model_manifest(path, url=url, size=size, md5=md5, sha256=sha256)
            return LogoProcessor.status(path.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _load_model(self) -> tuple[Any, Any, Any]:
        if self._model is not None:
            return self._model, self._torch, self._device
        path = _model_path(self.model_root)
        status = self.status(self.model_root)
        if not status.get("ready"):
            raise LogoModelUnavailable(str(status.get("details", "Anime-LaMa model is unavailable")))
        try:
            import torch
        except Exception as error:
            raise LogoModelUnavailable(f"PyTorch is required after a logo match: {error}") from error
        if self.require_gpu:
            try:
                available = bool(torch.cuda.is_available())
            except Exception:
                available = False
            if not available:
                raise GPURequiredError("logo removal requires CUDA; set require_gpu=False only for explicit CPU operation")
            device = "cuda"
        else:
            device = "cpu"
        try:
            model = torch.jit.load(str(path), map_location=device).eval()
        except Exception as error:
            raise LogoModelUnavailable(f"unable to load verified Anime-LaMa TorchScript model: {error}") from error
        self._model, self._torch, self._device = model, torch, device
        self._model_metadata = {
            "model_sha256": status.get("sha256", MODEL_SHA256),
            "model_md5": status.get("md5", MODEL_MD5),
            "dtype": "float32",
            "device": device,
            "path": str(path),
        }
        return model, torch, device

    def _inpaint(self, patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
        model, torch_mod, device = self._load_model()
        return _inpaint_patch(patch, mask, model, device, torch_mod)

    def _profiles(self, profiles: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
        if profiles is None:
            return ProfileStore().load()
        result = []
        for profile in profiles:
            try:
                profile_dir = Path(str(profile["_profile_dir"])) if isinstance(profile, Mapping) and profile.get("_profile_dir") else None
                # ``ProfileStore.list()`` intentionally strips private asset
                # paths.  Accepting that public representation here is useful
                # to service callers while preserving strict validation.
                if profile_dir is None and isinstance(profile, Mapping) and profile.get("id"):
                    candidate = ProfileStore().root / str(profile["id"])
                    if candidate.is_dir():
                        profile_dir = candidate
                result.append(_validate_profile(profile, profile_dir))
            except ProfileError:
                # Direct service payloads should be observable as skipped
                # profiles, while ProfileStore.load(ids=...) remains strict.
                continue
        return result

    def process(
        self,
        image: Image.Image,
        profiles: Iterable[Mapping[str, Any]] | None = None,
        force: bool = False,
    ) -> LogoResult:
        """Return a processed image and auditable metadata.

        ``profiles=None`` reads the default local profile store.  ``force`` is
        an explicit manual preview escape hatch: when no profile clears its
        threshold, the highest-scoring valid profile is used once.  It never
        invents a rectangle or bypasses profile geometry.
        """

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        original = image.copy()
        rgb = image.convert("RGB")
        rgb_array = _as_rgb_array(rgb)
        height, width = rgb_array.shape[:2]
        output_array = rgb_array.copy()
        output_mask = np.zeros((height, width), dtype=np.uint8)
        metadata: dict[str, Any] = {
            "status": "off",
            "reason": None,
            "warning": None,
            "profile_hits": [],
            "profiles": [],
            "force": bool(force),
            "bounds": [],
            "edit_bounds": [],
            "mask_bounds": [],
            "model_sha256": None,
            "model_md5": None,
            "dtype": None,
            "device": None,
        }
        loaded_profiles = self._profiles(profiles)
        metadata["profiles"] = [profile["id"] for profile in loaded_profiles]
        if not loaded_profiles:
            metadata["reason"] = "logo removal is off: no registered profiles"
            metadata["warning"] = metadata["reason"]
            return LogoResult(self._restore_mode(original, output_array), metadata, Image.fromarray(output_mask, mode="L"))

        candidates: list[dict[str, Any]] = []
        skipped_reasons: list[str] = []
        for profile in loaded_profiles:
            try:
                roi, mask, geometry_reason = _geometry(profile, (width, height))
                if geometry_reason:
                    skipped_reasons.append(f"{profile['id']}: {geometry_reason}")
                    continue
                template = _template_for_roi(profile, roi)
                if template is None:
                    skipped_reasons.append(f"{profile['id']}: template is unavailable")
                    continue
                patch = rgb_array[roi[1] : roi[3], roi[0] : roi[2]]
                template_array = _as_rgb_array(template)
                score, detail = _match_score(patch, template_array, mask, profile.get("detector", {}))
                threshold = float(profile.get("detector", {}).get("threshold", 0.62))
                mask_y, mask_x = np.where(mask > 0)
                mask_bounds = [
                    roi[0] + int(mask_x.min()),
                    roi[1] + int(mask_y.min()),
                    roi[0] + int(mask_x.max()) + 1,
                    roi[1] + int(mask_y.max()) + 1,
                ]
                candidates.append({
                    "profile": profile,
                    "roi": roi,
                    "mask": mask,
                    "template": template,
                    "score": score,
                    "threshold": threshold,
                    "detail": detail,
                    "mask_bounds": mask_bounds,
                })
            except (ProfileError, OSError, ValueError) as error:
                skipped_reasons.append(f"{profile.get('id', '?')}: {error}")

        hits = [candidate for candidate in candidates if candidate["score"] >= candidate["threshold"]]
        metadata["profile_scores"] = [
            {"id": candidate["profile"]["id"], "confidence": candidate["score"], "threshold": candidate["threshold"]}
            for candidate in candidates
        ]
        if not hits and force and candidates:
            hits = [max(candidates, key=lambda candidate: candidate["score"])]
            metadata["reason"] = "forced highest-scoring registered profile"
        if not hits:
            metadata["status"] = "skipped"
            metadata["reason"] = "no registered logo matched"
            metadata["warning"] = metadata["reason"]
            if skipped_reasons:
                metadata["skipped_profiles"] = skipped_reasons
            return LogoResult(self._restore_mode(original, output_array), metadata, Image.fromarray(output_mask, mode="L"))

        # Model loading occurs here, after all matching work, so an absent logo
        # never imports Torch or demands a model file/GPU.
        for hit in hits:
            roi = hit["roi"]
            patch = output_array[roi[1] : roi[3], roi[0] : roi[2]].copy()
            fixed = self._inpaint(patch, hit["mask"])
            if fixed.shape != patch.shape or fixed.dtype != np.uint8:
                raise LogoError("Anime-LaMa returned an invalid patch")
            try:
                import cv2

                support = cv2.GaussianBlur(hit["mask"], (11, 11), 3) > 0
            except Exception:
                support = hit["mask"] > 0
            # Keep the processor's edit contract even when a test double or
            # a future backend returns non-identical pixels outside its mask.
            fixed = np.where(support[..., None], fixed, patch).astype(np.uint8)
            output_array[roi[1] : roi[3], roi[0] : roi[2]] = fixed
            # GaussianBlur in the legacy blend is the declared edit support;
            # mark every non-zero support pixel so callers can audit all
            # pixels that may differ from the source.
            output_mask[roi[1] : roi[3], roi[0] : roi[2]] = np.maximum(
                output_mask[roi[1] : roi[3], roi[0] : roi[2]], support.astype(np.uint8) * 255
            )
            support_y, support_x = np.where(support)
            edit_bounds = [
                roi[0] + int(support_x.min()),
                roi[1] + int(support_y.min()),
                roi[0] + int(support_x.max()) + 1,
                roi[1] + int(support_y.max()) + 1,
            ]
            metadata["profile_hits"].append({
                "id": hit["profile"]["id"],
                "name": hit["profile"]["name"],
                "confidence": hit["score"],
                "threshold": hit["threshold"],
                # ``bounds`` is the actual support of the Gaussian blend;
                # callers can still inspect the original binary mask bounds
                # separately when reviewing the calibrated shape.
                "bounds": edit_bounds,
                "edit_bounds": edit_bounds,
                "mask_bounds": hit["mask_bounds"],
                "roi": list(hit["roi"]),
                "forced": bool(force and hit["score"] < hit["threshold"]),
                "match": hit["detail"],
            })
            metadata["bounds"].append(edit_bounds)
            metadata["edit_bounds"].append(edit_bounds)
            metadata["mask_bounds"].append(hit["mask_bounds"])

        model_metadata = self._model_metadata or {}
        metadata.update({
            "status": "processed",
            "reason": metadata.get("reason") or "registered logo matched",
            "model_sha256": model_metadata.get("model_sha256"),
            "model_md5": model_metadata.get("model_md5"),
            "dtype": model_metadata.get("dtype", "float32"),
            "device": model_metadata.get("device"),
        })
        return LogoResult(self._restore_mode(original, output_array), metadata, Image.fromarray(output_mask, mode="L"))

    @staticmethod
    def _restore_mode(original: Image.Image, rgb_array: np.ndarray) -> Image.Image:
        processed = Image.fromarray(rgb_array, mode="RGB")
        if original.mode == "RGBA":
            processed.putalpha(original.getchannel("A"))
            return processed
        if original.mode == "RGB":
            return processed
        # Keep grayscale callers grayscale when no alpha is present.  Paletted
        # and unusual modes are safely represented as RGB because an inpainted
        # image cannot retain their palette semantics without changing colors.
        if original.mode in {"1", "L", "I", "F", "I;16"}:
            return processed.convert("L")
        return processed


def status(model_root: Path | str | None = None) -> dict[str, Any]:
    """Module-level status convenience matching the CLI/service boundary."""

    return LogoProcessor.status(model_root)


def prepare(model_root: Path | str | None = None, *, url: str = MODEL_URL) -> dict[str, Any]:
    """Module-level explicit model preparation convenience."""

    return LogoProcessor.prepare(model_root, url=url)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or inspect the offline Anime-LaMa logo-removal model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="download and verify the pinned model")
    prepare.add_argument("--model-root", type=Path, default=None)
    status = subparsers.add_parser("status", help="inspect local model status")
    status.add_argument("--model-root", type=Path, default=None)
    args = parser.parse_args(argv)
    result = LogoProcessor.prepare(args.model_root) if args.command == "prepare" else LogoProcessor.status(args.model_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("ready") else 1


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke checks
    raise SystemExit(_main())


__all__ = [
    "GPURequiredError",
    "LogoError",
    "LogoModelUnavailable",
    "LogoProcessor",
    "LogoResult",
    "MODEL_FILENAME",
    "MODEL_MD5",
    "MODEL_SHA256",
    "MODEL_SIZE",
    "MODEL_URL",
    "DEFAULT_IMAGE_MODEL_ROOT",
    "prepare",
    "status",
    "PrepareError",
    "ProfileError",
    "ProfileStore",
    "PROFILE_SCHEMA_VERSION",
    "build_mask",
    "inpaint_patch",
]
