"""Create, calibrate, preview, and verify local fixed-logo profiles.

This command line tool is deliberately a thin orchestration layer around
``anime_picker.logo_removal``.  It owns profile assets and exact frame I/O;
template matching, mask geometry, and inpainting remain in the core module.
All paths supplied for input media are local paths.  Profile assets are copied
to the private, ignored profile store and are never uploaded. No code or
configuration in another workspace is required.

Examples (run from the repository root)::

    python scripts/logo_profiles.py list
    python scripts/logo_profiles.py create --id hi-anime --name "hi!anime.at" \
        --image frame-20s.png --roi 1440 0 1920 256 \
        --mask-rect 1718 18 1910 72
    python scripts/logo_profiles.py calibrate --profile muse \
        --video "source/The World's Strongest Witch Episode 1.mp4" \
        --positive-time 110 --negative-time 20 --negative-time 70
    python scripts/logo_profiles.py preview --profile hi-anime \
        --video episode.mp4 --time 20 --out .cache/logo-preview/hi-anime-20
    python scripts/logo_profiles.py verify --profile muse \
        --video "source/The World's Strongest Witch Episode 1.mp4" \
        --present-time 110 --absent-time 20 --absent-time 70

``preview`` performs real Anime-LaMa processing only when a verified model is
available.  ``calibrate`` and ``verify`` are CPU detector checks and replace
the inpainting boundary with a deterministic test seam, so they never reserve
the GPU.  Use ``--cpu`` with preview only for an explicitly requested local
CPU operation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


# A script launched as ``python scripts/logo_profiles.py`` has ``scripts`` as
# sys.path[0].  Insert the repository root before importing the package, while
# preserving an installed package when the script is called from elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anime_picker.logo_removal import (  # noqa: E402
    PROFILE_SCHEMA_VERSION,
    LogoProcessor,
    ProfileError,
    ProfileStore,
)
from anime_picker.video import FrameNotFoundError, VideoReader  # noqa: E402


ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
PNG_SUFFIXES = {".png"}


class CliError(ValueError):
    """An expected command-line or profile-input error."""


@dataclass(frozen=True)
class Sample:
    """An unprocessed local image together with optional source timing."""

    label: str
    image: Image.Image
    source: str
    pts: int | None = None
    time: float | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_profile_root() -> Path:
    configured = os.environ.get("ANIME_PICKER_LOGO_PROFILE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    # ``.local`` is a stable private configuration location.  The setup
    # workflow adds it to .gitignore; no profile asset belongs in Git.
    return (REPO_ROOT / ".local" / "logo-profiles").resolve()


def _root_from_args(args: argparse.Namespace) -> Path:
    value = getattr(args, "profile_root", None)
    return Path(value).expanduser().resolve() if value else _default_profile_root()


def _safe_id(value: str) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise CliError(
            "profile id must start with a letter/number and contain only "
            "letters, numbers, '.', '_' or '-' (max 64 characters)"
        )
    return value


def _local_file(value: str | os.PathLike[str], *, field: str, resolve: bool = True) -> Path:
    if isinstance(value, str) and "\x00" in value:
        raise CliError(f"{field} contains a NUL byte")
    path = Path(value).expanduser()
    if not path.exists():
        raise CliError(f"{field} does not exist: {path}")
    if not path.is_file():
        raise CliError(f"{field} is not a file: {path}")
    # Create-time safety checks need the lexical path as well as its canonical
    # target so a symlink placed inside the profile directory is rejected.
    return path.resolve() if resolve else Path(os.path.abspath(os.fspath(path)))


def _safe_output_dir(value: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
    if isinstance(value, str) and "\x00" in value:
        raise CliError("output path contains a NUL byte")
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise CliError(f"output path must be a directory: {path}")
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise CliError(f"output directory is not empty; pass --overwrite: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalized_path(path: Path, *, resolve: bool) -> str:
    """Return a case-normalized Windows-safe path for comparisons."""

    try:
        candidate = path.resolve(strict=False) if resolve else Path(os.path.abspath(os.fspath(path)))
    except OSError:
        candidate = Path(os.path.abspath(os.fspath(path)))
    return os.path.normcase(os.path.normpath(os.fspath(candidate)))


def _path_is_within(path: Path, directory: Path, *, resolve: bool) -> bool:
    """Whether ``path`` is lexically/canonically inside ``directory``."""

    candidate = _normalized_path(path, resolve=resolve)
    parent = _normalized_path(directory, resolve=resolve)
    try:
        return os.path.commonpath([candidate, parent]) == parent
    except ValueError:
        # Different Windows drives cannot be nested.
        return False


def _same_path(first: Path, second: Path) -> bool:
    """Match ordinary paths and symlink aliases without requiring both files."""

    lexical_match = _normalized_path(first, resolve=False) == _normalized_path(second, resolve=False)
    canonical_match = _normalized_path(first, resolve=True) == _normalized_path(second, resolve=True)
    if lexical_match or canonical_match:
        return True
    try:
        return first.exists() and second.exists() and os.path.samefile(first, second)
    except OSError:
        return False


def _assert_create_inputs_safe(profile_dir: Path, inputs: Iterable[tuple[str, Path]]) -> None:
    """Reject source assets that could be overwritten during profile creation.

    ``--overwrite`` authorizes replacing the profile's own files only.  It
    must never authorize reading a source from that same directory: writing
    ``template.png`` first would otherwise mutate the source before
    ``reference.png`` is copied.  Both lexical and resolved comparisons catch
    direct paths, case variants, and symlink aliases.
    """

    destinations = [
        profile_dir / "template.png",
        profile_dir / "mask.png",
        profile_dir / "reference.png",
        profile_dir / "profile.json",
    ]
    for field, source in inputs:
        if _path_is_within(source, profile_dir, resolve=False) or _path_is_within(source, profile_dir, resolve=True):
            raise CliError(f"{field} must be outside the target profile directory: {source}")
        for destination in destinations:
            if _same_path(source, destination):
                raise CliError(f"{field} aliases a generated profile asset: {source}")


def _positive_int(value: str, *, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise CliError(f"{field} must be an integer") from error
    return result


def _finite_float(value: str, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CliError(f"{field} must be a number") from error
    if not math.isfinite(result):
        raise CliError(f"{field} must be finite")
    return result


def _box(values: Sequence[Any], *, field: str, width: int | None = None, height: int | None = None) -> list[int]:
    if len(values) != 4:
        raise CliError(f"{field} must contain x0 y0 x1 y1")
    try:
        box = [int(value) for value in values]
    except (TypeError, ValueError) as error:
        raise CliError(f"{field} must contain integer coordinates") from error
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise CliError(f"{field} must be a non-empty box inside the image")
    if width is not None and (x1 > width or y1 > height):
        raise CliError(f"{field} is outside the source image {width}x{height}")
    return box


def _parse_polygon(value: str, *, field: str) -> list[list[int]]:
    """Accept JSON or compact ``x,y;x,y;...`` polygon syntax."""

    text = str(value).strip()
    if not text:
        raise CliError(f"{field} is empty")
    parsed: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        points: list[list[int]] = []
        for item in text.split(";"):
            pair = [part.strip() for part in item.split(",")]
            if len(pair) != 2:
                raise CliError(f"{field} must use JSON or x,y;x,y syntax")
            try:
                points.append([int(pair[0]), int(pair[1])])
            except ValueError as error:
                raise CliError(f"{field} must contain integer coordinates") from error
        parsed = points
    if not isinstance(parsed, list) or len(parsed) < 3:
        raise CliError(f"{field} must contain at least three [x, y] points")
    result: list[list[int]] = []
    for point in parsed:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise CliError(f"{field} must contain [x, y] points")
        try:
            result.append([int(point[0]), int(point[1])])
        except (TypeError, ValueError) as error:
            raise CliError(f"{field} must contain integer coordinates") from error
    return result


def _validate_points(points: Iterable[Sequence[int]], *, field: str, width: int, height: int) -> list[list[int]]:
    result: list[list[int]] = []
    for point in points:
        if len(point) != 2:
            raise CliError(f"{field} must contain [x, y] points")
        x, y = int(point[0]), int(point[1])
        if not (0 <= x <= width and 0 <= y <= height):
            raise CliError(f"{field} point {(x, y)} is outside the source image {width}x{height}")
        result.append([x, y])
    if len(result) < 3:
        raise CliError(f"{field} must contain at least three points")
    return result


def _open_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.copy()
    except Exception as error:
        raise CliError(f"unable to read image {path}: {error}") from error


def _save_png(image: Image.Image, path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise CliError(f"refusing to overwrite existing file; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save an actual PNG regardless of the source extension.  PNG is required
    # for lossless template/reference evidence and native export checks.
    image.save(path, format="PNG")


def _copy_png_as_rgb(source: Path, destination: Path, *, overwrite: bool) -> None:
    image = _open_image(source)
    # Preserve a supplied PNG byte-for-byte when possible so the stored
    # reference remains the native source artifact.  Non-PNG inputs are
    # normalized to a lossless RGB PNG for the local profile store.
    if source.suffix.lower() == ".png":
        if destination.exists() and not overwrite:
            raise CliError(f"refusing to overwrite existing file; pass --overwrite: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    else:
        _save_png(image.convert("RGB"), destination, overwrite=overwrite)


def _load_profile(root: Path, profile_id: str) -> dict[str, Any]:
    profile_id = _safe_id(profile_id)
    profiles = ProfileStore(root).load([profile_id])
    if not profiles:
        raise CliError(f"profile is unavailable: {profile_id}")
    return profiles[0]


def _profile_dir(root: Path, profile_id: str) -> Path:
    return root / _safe_id(profile_id)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise CliError(f"generated asset escaped profile directory: {path}") from error


def _resolve_frame_from_video(path: Path, *, time_value: float | None, pts: int | None) -> Sample:
    if (time_value is None) == (pts is None):
        raise CliError("choose exactly one of --time or --pts for a video")
    try:
        reader = VideoReader(path)
        if pts is not None:
            frame = reader.get_frame(int(pts), max_side=None)
        else:
            target = float(time_value)
            if target < 0:
                raise CliError("--time must be >= 0")
            # sample() selects the first presentation frame at or after the
            # requested time.  Keep the bound wider than one frame period so
            # a target between VFR frames still resolves deterministically.
            span = max(1.0 / max(reader.info.fps, 1.0), 0.001)
            frame = next(iter(reader.sample(target, target + span, span, max_side=None)))
        return Sample(path.name, frame.image.copy(), str(path), int(frame.pts), float(frame.time))
    except StopIteration as error:
        raise CliError(f"no video frame found at requested time: {path}") from error
    except FrameNotFoundError as error:
        raise CliError(str(error)) from error


def _load_sample(path_value: str, *, time_value: float | None = None, pts: int | None = None, label: str | None = None) -> Sample:
    path = _local_file(path_value, field="input")
    if path.suffix.lower() in VIDEO_SUFFIXES:
        sample = _resolve_frame_from_video(path, time_value=time_value, pts=pts)
        return Sample(label or sample.label, sample.image, sample.source, sample.pts, sample.time)
    if time_value is not None or pts is not None:
        raise CliError(f"--time/--pts can only be used with a video: {path}")
    return Sample(label or path.name, _open_image(path), str(path))


def _selected_image(args: argparse.Namespace) -> Sample:
    image_path = getattr(args, "image", None)
    video_path = getattr(args, "video", None)
    if bool(image_path) == bool(video_path):
        raise CliError("choose exactly one of --image or --video")
    return _load_sample(
        image_path or video_path,
        time_value=getattr(args, "time", None),
        pts=getattr(args, "pts", None),
    )


def _profile_assets_from_create(args: argparse.Namespace, source: Image.Image, profile_dir: Path) -> tuple[str, str | None]:
    width, height = source.size
    roi = _box(args.roi, field="--roi", width=width, height=height)
    template_path = profile_dir / "template.png"
    if args.template:
        template_source = _local_file(args.template, field="--template")
        template = _open_image(template_source).convert("RGB")
    else:
        template = source.convert("RGB").crop(tuple(roi))
    _save_png(template, template_path, overwrite=True)

    custom_name: str | None = None
    if args.mask_file:
        mask_source = _local_file(args.mask_file, field="--mask-file")
        mask = _open_image(mask_source).convert("L")
        custom_path = profile_dir / "mask.png"
        _save_png(mask, custom_path, overwrite=True)
        custom_name = "mask.png"
    return "template.png", custom_name


def _profile_mapping(args: argparse.Namespace, source: Image.Image, *, template_name: str, custom_name: str | None) -> dict[str, Any]:
    width, height = source.size
    roi = _box(args.roi, field="--roi", width=width, height=height)
    rectangles = [_box(values, field="--mask-rect", width=width, height=height) for values in (args.mask_rect or [])]
    polygons = [_validate_points(_parse_polygon(value, field="--polygon"), field="--polygon", width=width, height=height) for value in (args.polygon or [])]
    if not rectangles and not polygons and custom_name is None:
        raise CliError("provide --mask-rect, --polygon, or --mask-file")

    context = None
    if args.context_roi:
        context = _box(args.context_roi, field="--context-roi", width=width, height=height)
    detector = {
        "threshold": float(args.threshold),
        "aspect_tolerance": float(args.aspect_tolerance),
        "edge_weight": float(args.edge_weight),
        "detail_weight": float(args.detail_weight),
    }
    payload: dict[str, Any] = {
        "version": PROFILE_SCHEMA_VERSION,
        "id": _safe_id(args.id),
        "name": str(args.name).strip() or _safe_id(args.id),
        "reference_size": [width, height],
        "roi": roi,
        "mask_coordinates": "full_frame",
        "mask": {"rectangles": rectangles, "polygons": polygons},
        "template_image": template_name,
        "detector": detector,
        # Keep profile JSON portable.  The full reference PNG is kept next to
        # profile.json, while the original source path is intentionally not
        # embedded because it may contain a user-specific absolute path.
        "source": {"basename": Path(args.image).name, "created_at": _utc_now()},
    }
    if custom_name is not None:
        payload["custom_mask"] = custom_name
    if context is not None:
        payload["context"] = {"roi": context, "description": "optional context region; mask remains separately declared"}
    if args.notes:
        payload["notes"] = str(args.notes)
    return payload


def command_list(args: argparse.Namespace) -> int:
    root = _root_from_args(args)
    result = {"root": str(root), "profiles": ProfileStore(root).list()}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_create(args: argparse.Namespace) -> int:
    profile_id = _safe_id(args.id)
    source_path = _local_file(args.image, field="--image", resolve=False)
    root = _root_from_args(args)
    directory = _profile_dir(root, profile_id)
    profile_json = directory / "profile.json"
    if directory.exists() and any(directory.iterdir()) and not args.overwrite:
        raise CliError(f"profile already exists; pass --overwrite to replace it: {directory}")

    # Resolve every source asset before creating or replacing any profile
    # asset.  In particular, ``--image <profile>/template.png`` must fail
    # before template.png is rewritten by --overwrite.
    create_inputs: list[tuple[str, Path]] = [("--image", source_path)]
    if args.template:
        create_inputs.append(("--template", _local_file(args.template, field="--template", resolve=False)))
    if args.mask_file:
        create_inputs.append(("--mask-file", _local_file(args.mask_file, field="--mask-file", resolve=False)))
    _assert_create_inputs_safe(directory, create_inputs)

    source = _open_image(source_path)
    root.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)

    # Assets are written before profile.json.  A failed copy therefore leaves
    # no newly loadable profile, and an explicit overwrite never touches the
    # source image.
    template_name, custom_name = _profile_assets_from_create(args, source, directory)
    payload = _profile_mapping(args, source, template_name=template_name, custom_name=custom_name)
    _copy_png_as_rgb(source_path, directory / "reference.png", overwrite=True)
    if profile_json.exists() and not args.overwrite:
        raise CliError(f"refusing to overwrite profile.json: {profile_json}")
    profile_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    try:
        loaded = _load_profile(root, profile_id)
    except Exception:
        # Keep malformed assets inspectable for an explicit overwrite retry,
        # but never claim success when the core schema rejects them.
        raise
    result = {
        "status": "created",
        "profile": {key: value for key, value in loaded.items() if not key.startswith("_")},
        "directory": str(directory),
        "assets": [str(path) for path in sorted(directory.iterdir()) if path.is_file()],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _score_sample(processor: LogoProcessor, profile: Mapping[str, Any], sample: Sample) -> tuple[float | None, dict[str, Any]]:
    result = processor.process(sample.image, [profile])
    hit = result.metadata.get("profile_hits") or []
    if hit:
        score = float(hit[0].get("confidence", 0.0))
    else:
        scores = result.metadata.get("profile_scores") or []
        score = float(scores[0].get("confidence")) if scores else None
    return score, {
        "label": sample.label,
        "source": sample.source,
        "pts": sample.pts,
        "time": sample.time,
        "score": score,
        "status": result.metadata.get("status"),
        "reason": result.metadata.get("reason"),
    }


def _dry_processor() -> LogoProcessor:
    processor = LogoProcessor(model_root=None, require_gpu=False)

    def deterministic_dry_inpaint(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Calibration and verification need a hit/mask signal but must never
        # load Anime-LaMa.  Keep the actual image unchanged for calibration;
        # verify installs its own replacement to exercise edit boundaries.
        return patch.copy()

    processor._inpaint = deterministic_dry_inpaint  # type: ignore[method-assign]
    return processor


def _samples_from_calibration(args: argparse.Namespace) -> tuple[list[Sample], list[Sample]]:
    positives = [_load_sample(value, label=f"positive:{Path(value).name}") for value in (args.positive or [])]
    negatives = [_load_sample(value, label=f"negative:{Path(value).name}") for value in (args.negative or [])]
    if args.video:
        if not args.positive_time and not args.positive_pts and not args.negative_time and not args.negative_pts:
            raise CliError("calibrate --video requires positive/negative time or PTS samples")
        for value in args.positive_time or []:
            positives.append(_load_sample(args.video, time_value=float(value), label=f"positive@{value}s"))
        for value in args.positive_pts or []:
            positives.append(_load_sample(args.video, pts=int(value), label=f"positive@pts={value}"))
        for value in args.negative_time or []:
            negatives.append(_load_sample(args.video, time_value=float(value), label=f"negative@{value}s"))
        for value in args.negative_pts or []:
            negatives.append(_load_sample(args.video, pts=int(value), label=f"negative@pts={value}"))
    if not positives and not negatives:
        raise CliError("provide --positive/--negative images or --video samples")
    return positives, negatives


def _recommended_threshold(positive: Sequence[float], negative: Sequence[float], current: float) -> tuple[float, str]:
    if positive and negative:
        lower = max(negative) + 0.01
        upper = min(positive) - 0.01
        if lower <= upper:
            return max(0.0, min(1.0, (lower + upper) / 2.0)), "separated midpoint"
        midpoint = (max(negative) + min(positive)) / 2.0
        return max(0.0, min(1.0, midpoint)), "overlapping midpoint; inspect samples"
    if positive:
        return max(0.0, min(1.0, min(positive) * 0.9)), "positive-only conservative threshold"
    if negative:
        return max(0.0, min(1.0, max(negative) + 0.05)), "negative-only threshold"
    return float(current), "no usable samples; unchanged"


def command_calibrate(args: argparse.Namespace) -> int:
    root = _root_from_args(args)
    profile_id = _safe_id(args.profile)
    profile = _load_profile(root, profile_id)
    positives, negatives = _samples_from_calibration(args)
    processor = _dry_processor()
    positive_rows = [_score_sample(processor, profile, sample)[1] for sample in positives]
    negative_rows = [_score_sample(processor, profile, sample)[1] for sample in negatives]
    positive_scores = [float(row["score"]) for row in positive_rows if row["score"] is not None]
    negative_scores = [float(row["score"]) for row in negative_rows if row["score"] is not None]
    current = float(profile.get("detector", {}).get("threshold", 0.62))
    threshold, rationale = _recommended_threshold(positive_scores, negative_scores, current)
    report: dict[str, Any] = {
        "status": "calibrated",
        "profile": profile_id,
        "cpu_only": True,
        "current_threshold": current,
        "recommended_threshold": threshold,
        "rationale": rationale,
        "positive": positive_rows,
        "negative": negative_rows,
        "source_inputs_unprocessed": True,
    }
    if args.write:
        if not args.overwrite:
            raise CliError("writing calibration requires explicit --overwrite")
        directory = _profile_dir(root, profile_id)
        path = directory / "profile.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        detector = raw.setdefault("detector", {})
        detector["threshold"] = threshold
        raw["calibration"] = {
            "updated_at": _utc_now(),
            "recommended_threshold": threshold,
            "rationale": rationale,
            "positive_scores": positive_scores,
            "negative_scores": negative_scores,
        }
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        _load_profile(root, profile_id)
        report["written"] = str(path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _process_preview_image(args: argparse.Namespace, sample: Sample, profile: Mapping[str, Any]) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, Any]]:
    processor = LogoProcessor(model_root=args.model_root, require_gpu=not args.cpu)
    if args.dry:
        processor = _dry_processor()
    result = processor.process(sample.image, [profile])
    metadata = dict(result.metadata)
    metadata.update({
        "source": sample.source,
        "source_pts": sample.pts,
        "source_time": sample.time,
        "native_size": list(sample.image.size),
        "profile_id": profile["id"],
        "dry": bool(args.dry),
        "created_at": _utc_now(),
    })
    return sample.image, result.image, result.edit_mask, metadata | {"edit_mask_size": list(result.edit_mask.size), "edit_mask_nonzero": int(np.count_nonzero(np.asarray(result.edit_mask)))}


def command_preview(args: argparse.Namespace) -> int:
    sample = _selected_image(args)
    root = _root_from_args(args)
    profile = _load_profile(root, args.profile)
    output = _safe_output_dir(args.out, overwrite=args.overwrite)
    before, after, edit_mask, metadata = _process_preview_image(args, sample, profile)
    _save_png(before, output / "before.png", overwrite=True)
    _save_png(after, output / "after.png", overwrite=True)
    _save_png(edit_mask.convert("L"), output / "mask.png", overwrite=True)
    metadata["files"] = {"before": "before.png", "after": "after.png", "mask": "mask.png", "metadata": "metadata.json"}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "previewed", "directory": str(output), "metadata": metadata}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _verification_inpaint(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    replacement = patch.copy()
    replacement[mask > 0] = np.array([17, 29, 43], dtype=np.uint8)
    return replacement


def _verification_rows(args: argparse.Namespace) -> list[tuple[str, Sample, str | None]]:
    rows: list[tuple[str, Sample, str | None]] = []
    for value in args.present or []:
        rows.append((f"present:{Path(value).name}", _load_sample(value), "present"))
    for value in args.absent or []:
        rows.append((f"absent:{Path(value).name}", _load_sample(value), "absent"))
    for value in args.case or []:
        if "=" not in value:
            raise CliError("--case uses PATH=present or PATH=absent")
        path, expected = value.rsplit("=", 1)
        expected = expected.strip().lower()
        if expected not in {"present", "absent"}:
            raise CliError("--case expectation must be present or absent")
        rows.append((f"case:{Path(path).name}", _load_sample(path), expected))
    if args.video:
        for value in args.present_time or []:
            rows.append((f"present@{value}s", _load_sample(args.video, time_value=float(value)), "present"))
        for value in args.absent_time or []:
            rows.append((f"absent@{value}s", _load_sample(args.video, time_value=float(value)), "absent"))
        for value in args.present_pts or []:
            rows.append((f"present@pts={value}", _load_sample(args.video, pts=int(value)), "present"))
        for value in args.absent_pts or []:
            rows.append((f"absent@pts={value}", _load_sample(args.video, pts=int(value)), "absent"))
    if not rows:
        raise CliError("provide --present/--absent/--case or --video samples")
    return rows


def command_verify(args: argparse.Namespace) -> int:
    root = _root_from_args(args)
    profile = _load_profile(root, args.profile)
    output = _safe_output_dir(args.out, overwrite=args.overwrite) if args.out else None
    processor = _dry_processor()
    processor._inpaint = _verification_inpaint  # type: ignore[method-assign]
    rows = []
    failures: list[str] = []
    for label, sample, expected in _verification_rows(args):
        before = np.asarray(sample.image.convert("RGB"))
        result = processor.process(sample.image, [profile])
        after = np.asarray(result.image.convert("RGB"))
        edit_mask = np.asarray(result.edit_mask) > 0
        status = str(result.metadata.get("status"))
        detected = status == "processed"
        outside_equal = bool(np.array_equal(before[~edit_mask], after[~edit_mask]))
        native_size = list(sample.image.size) == list(result.image.size)
        all_equal_if_absent = bool(np.array_equal(before, after)) if not detected else None
        passed = outside_equal and native_size
        if expected == "present":
            passed = passed and detected
        elif expected == "absent":
            passed = passed and not detected and bool(all_equal_if_absent)
        if not passed:
            failures.append(label)
        rows.append({
            "label": label,
            "source": sample.source,
            "pts": sample.pts,
            "time": sample.time,
            "expected": expected,
            "detected": detected,
            "status": status,
            "profile_hits": result.metadata.get("profile_hits", []),
            "native_size": list(result.image.size),
            "outside_mask_pixel_exact": outside_equal,
            "absent_frame_pixel_exact": all_equal_if_absent,
            "edit_mask_nonzero": int(np.count_nonzero(edit_mask)),
            "passed": passed,
        })
    report = {
        "status": "pass" if not failures else "fail",
        "profile": profile["id"],
        "cpu_only": True,
        "source_inputs_unprocessed": True,
        "rows": rows,
        "failures": failures,
        "created_at": _utc_now(),
    }
    if output:
        (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
        report["report"] = str(output / "verification.json")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


def _add_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile-root",
        "--root",
        dest="profile_root",
        type=Path,
        default=argparse.SUPPRESS,
        help="profile store (default: .local/logo-profiles or ANIME_PICKER_LOGO_PROFILE_ROOT)",
    )


def _add_overwrite(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--overwrite", action="store_true", help="explicitly allow replacing profile/output files")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_root_arg(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="list validated local profiles")
    _add_root_arg(listing)
    listing.set_defaults(handler=command_list)

    create = subparsers.add_parser("create", help="create a profile from an unprocessed native PNG")
    _add_root_arg(create)
    create.add_argument("--id", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--image", required=True, help="unprocessed local native PNG/reference image")
    create.add_argument("--roi", nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    create.add_argument("--template", help="optional local template image; default is the ROI crop")
    create.add_argument("--mask-file", "--maskfile", dest="mask_file", help="optional ROI-local/full-frame mask PNG")
    create.add_argument("--mask-rect", "--maskrect", dest="mask_rect", nargs=4, action="append", metavar=("X0", "Y0", "X1", "Y1"))
    create.add_argument("--polygon", action="append", help="mask polygon as JSON or x,y;x,y;...")
    create.add_argument("--context-roi", nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="optional context region kept separate from mask")
    create.add_argument("--threshold", type=float, default=0.62)
    create.add_argument("--aspect-tolerance", type=float, default=0.025)
    create.add_argument("--edge-weight", type=float, default=0.65)
    create.add_argument("--detail-weight", type=float, default=0.35)
    create.add_argument("--notes")
    _add_overwrite(create)
    create.set_defaults(handler=command_create)

    calibrate = subparsers.add_parser("calibrate", help="calibrate threshold using CPU-only detector checks")
    _add_root_arg(calibrate)
    calibrate.add_argument("--profile", required=True)
    calibrate.add_argument("--positive", action="append", help="unprocessed positive image")
    calibrate.add_argument("--negative", action="append", help="unprocessed negative image")
    calibrate.add_argument("--video", help="local video for native PTS/time samples")
    calibrate.add_argument("--positive-time", action="append", type=float)
    calibrate.add_argument("--negative-time", action="append", type=float)
    calibrate.add_argument("--positive-pts", action="append", type=int)
    calibrate.add_argument("--negative-pts", action="append", type=int)
    calibrate.add_argument("--write", action="store_true", help="write recommended threshold into profile.json")
    _add_overwrite(calibrate)
    calibrate.set_defaults(handler=command_calibrate)

    preview = subparsers.add_parser("preview", help="write before/after/mask PNGs and metadata")
    _add_root_arg(preview)
    preview.add_argument("--profile", required=True)
    preview.add_argument("--image")
    preview.add_argument("--video")
    preview.add_argument("--time", type=float)
    preview.add_argument("--pts", type=int)
    preview.add_argument("--out", required=True, help="output directory")
    preview.add_argument("--model-root", type=Path)
    preview.add_argument("--cpu", action="store_true", help="explicitly run real inpainting on CPU")
    preview.add_argument("--dry", action="store_true", help="CPU detector-only preview; after.png is unchanged")
    _add_overwrite(preview)
    preview.set_defaults(handler=command_preview)

    verify = subparsers.add_parser("verify", help="CPU-only present/absent and exact-mask verification")
    _add_root_arg(verify)
    verify.add_argument("--profile", required=True)
    verify.add_argument("--present", action="append")
    verify.add_argument("--absent", action="append")
    verify.add_argument("--case", action="append", help="PATH=present or PATH=absent")
    verify.add_argument("--video")
    verify.add_argument("--present-time", action="append", type=float)
    verify.add_argument("--absent-time", action="append", type=float)
    verify.add_argument("--present-pts", action="append", type=int)
    verify.add_argument("--absent-pts", action="append", type=int)
    verify.add_argument("--out", help="optional report directory")
    _add_overwrite(verify)
    verify.set_defaults(handler=command_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (CliError, ProfileError, OSError, ValueError, RuntimeError) as error:
        print(f"logo profile error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
