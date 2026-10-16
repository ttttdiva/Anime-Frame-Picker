"""Local anime detection, identity, and tag models.

The application deliberately keeps model acquisition separate from inference.
``ModelService.prepare`` downloads a pinned, audited bundle, while all of the
runtime methods below use files under ``model_root`` and set the Hugging Face
and imgutils caches to that same directory.  The default runtime requires a
real CUDA PyTorch installation and a CUDA ONNX Runtime provider.  Tests and
diagnostic tools may opt into CPU execution explicitly with
``require_gpu=False``.

There are no model objects imported at module import time.  This is important
for the web application's setup screen (and for machines on which the large
optional dependencies have not been installed yet).
"""

from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:  # Pillow is a required application dependency, but keep import friendly.
    from PIL import Image
except Exception:  # pragma: no cover - exercised only in a minimal setup env
    Image = Any  # type: ignore[assignment,misc]


DEFAULT_MODEL_ROOT = Path(r"D:\AI\models\Cold_F\image")
IDENTITY_MODEL = "ccip-caformer_b36-24"
IDENTITY_THRESHOLD = 0.2132311898
# This repository commit carries the current OpenRAIL model card while the
# immutable model object IDs remain the same as the earlier audited pin.
CCIP_REVISION = "eb2acdd29af1703388d3d0c04221add322bc9110"
PIXAI_REPOSITORY = "pixai-labs/pixai-tagger-v1.0"
PIXAI_REVISION = "9fe10addf9326e292da8a85a98ea74cd91b41771"
PERSON_REPOSITORY = "deepghs/anime_person_detection"
PERSON_REVISION = "e39c744c22432ad01f91dd254fe2b02c8d878b8c"
HEAD_REPOSITORY = "deepghs/anime_head_detection"
HEAD_REVISION = "06604feee81983792a57c21081e539c0ae229833"


# These names are intentionally module globals.  They make the model boundary
# easy to replace in unit tests, and allow the actual optional packages to be
# imported only when an inference method is first used.
ccip_batch_extract_features: Callable[..., Any] | None = None
ccip_batch_differences: Callable[..., Any] | None = None
ccip_difference: Callable[..., Any] | None = None
detect_person: Callable[..., Any] | None = None
detect_heads: Callable[..., Any] | None = None
detect_head: Callable[..., Any] | None = None
snapshot_download: Callable[..., Any] | None = None


class ModelError(RuntimeError):
    """Base class for local model errors."""


class ModelUnavailable(ModelError):
    """The prepared model bundle or a required runtime dependency is absent."""


class GPURequiredError(ModelUnavailable):
    """GPU inference was requested but the actual runtime cannot provide it."""


class PrepareError(ModelError):
    """A pinned model bundle could not be downloaded or verified."""


def _root_from_argument(model_root: Path | str | None) -> Path:
    if model_root is None:
        model_root = os.environ.get("ANIME_PICKER_MODEL_ROOT")
    return Path(model_root) if model_root else DEFAULT_MODEL_ROOT


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return None


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _image_size(image: Any) -> tuple[int, int]:
    size = getattr(image, "size", None)
    if isinstance(size, Sequence) and len(size) >= 2:
        try:
            return max(1, int(size[0])), max(1, int(size[1]))
        except (TypeError, ValueError):
            pass
    shape = getattr(image, "shape", None)
    if isinstance(shape, Sequence) and len(shape) >= 2:
        return max(1, int(shape[1])), max(1, int(shape[0]))
    raise TypeError("model input must expose a PIL size or an array shape")


def _relative_file(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate = candidate.resolve()
        root_resolved = root.resolve()
        candidate.relative_to(root_resolved)
    except (OSError, ValueError):
        return None
    return candidate


def _first_file(directory: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    return None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _extract_box(raw: Any) -> tuple[Sequence[Any] | None, str, float]:
    """Normalize the tuple and mapping variants returned by imgutils."""

    if isinstance(raw, Mapping):
        box = raw.get("box", raw.get("bbox", raw.get("xyxy")))
        label = raw.get("kind", raw.get("label", raw.get("class", "person")))
        score = raw.get("score", raw.get("confidence", raw.get("conf", 1.0)))
        return box, str(label or "person"), _safe_float(score, 0.0)
    if isinstance(raw, (tuple, list)):
        if len(raw) >= 3 and isinstance(raw[0], (tuple, list, np.ndarray)):
            return raw[0], str(raw[1] or "person"), _safe_float(raw[2], 0.0)
        if len(raw) == 4:
            return raw, "person", 1.0
    box = getattr(raw, "box", getattr(raw, "bbox", None))
    label = getattr(raw, "kind", getattr(raw, "label", "person"))
    score = getattr(raw, "score", getattr(raw, "confidence", 1.0))
    return box, str(label or "person"), _safe_float(score, 0.0)


def _normalise_detection(raw: Any, width: int, height: int) -> dict[str, Any] | None:
    box, label, score = _extract_box(raw)
    if box is None:
        return None
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    # imgutils returns pixel coordinates.  Accept normalized values as a small
    # adapter convenience for test doubles and other detector implementations.
    if max(abs(value) for value in values) <= 1.5 and max(width, height) > 4:
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    x0, x1 = sorted((max(0.0, min(float(width), values[0])), max(0.0, min(float(width), values[2]))))
    y0, y1 = sorted((max(0.0, min(float(height), values[1])), max(0.0, min(float(height), values[3]))))
    if x1 <= x0 or y1 <= y0:
        return None
    kind = "head" if str(label).lower() in {"head", "face"} else "person"
    return {"box": [x0, y0, x1, y1], "score": max(0.0, min(1.0, score)), "kind": kind}


def _patch_hf_resolver(module: Any, resolver: Callable[..., str]) -> None:
    """Patch an official imgutils module's hub helper to the prepared files."""

    if module is None:
        return
    if hasattr(module, "hf_hub_download"):
        try:
            setattr(module, "hf_hub_download", resolver)
        except Exception:
            pass


def _audit_python_file(path: Path) -> dict[str, Any]:
    """Audit the pinned PixAI custom module for outbound/process primitives."""

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        return {"passed": False, "error": str(error), "imports": [], "calls": []}
    banned_imports = {"requests", "httpx", "urllib", "urllib3", "socket", "subprocess", "ftplib"}
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names if alias.name.split(".", 1)[0] in banned_imports)
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0] in banned_imports:
            imports.append(node.module)
        elif isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Attribute) and target.attr in {"system", "popen", "run", "Popen", "call"}:
                calls.append(target.attr)
            elif isinstance(target, ast.Name) and target.id in {"eval", "exec", "compile"}:
                calls.append(target.id)
    return {"passed": not imports and not calls, "imports": sorted(set(imports)), "calls": sorted(set(calls))}


class ModelService:
    """Offline anime detector, CCIP identity model, and PixAI tagger."""

    identity_model = IDENTITY_MODEL
    threshold = IDENTITY_THRESHOLD

    def __init__(self, model_root: Path | str | None = None, require_gpu: bool = True) -> None:
        self.model_root = _root_from_argument(model_root)
        self.root = self.model_root
        self.require_gpu = bool(require_gpu)
        self._manifest = _read_json(self.model_root / "manifest.json")
        self.manifest_revision = self._manifest_revision(self._manifest)
        self._tag_precision: str | None = None
        self._tag_precision_reason: str | None = None
        self._tag_precision_requested = os.environ.get("ANIME_PICKER_TAG_DTYPE", "auto").strip().lower() or "auto"
        self._tagger: Any = None
        self._ccip_module: Any = None
        self._runtime_checked = False
        self._runtime_error: str | None = None
        self._tag_vocabulary: dict[str, str] | None = None
        self._configure_cache_env()

    @property
    def tag_precision(self) -> str:
        """Effective PixAI inference precision for this service instance."""

        if self._tag_precision is not None:
            return self._tag_precision
        self._resolve_tag_precision()
        assert self._tag_precision is not None  # for type checkers
        return self._tag_precision

    @property
    def revision(self) -> str:
        """Cache revision including the effective PixAI inference precision.

        ``manifest_revision`` remains the stable SHA of the prepared manifest.
        The public service revision adds the precision policy because the
        pipeline uses it in feature/tag cache keys; switching between float32
        and bfloat16 must never reuse stale tag scores.
        """

        try:
            precision = self.tag_precision
        except GPURequiredError:
            # Reading metadata must remain possible on an unprepared or
            # CPU-only host; the inference entry points still raise the same
            # error before they can silently execute on CPU.
            precision = "unresolved"
        return f"{self.manifest_revision}:tag-{precision}"

    @property
    def identity_revision(self) -> str:
        """Stable CCIP cache revision, independent of tagger precision."""

        return self.manifest_revision

    @property
    def tag_revision(self) -> str:
        """Precision-aware cache revision for PixAI tag scores."""

        return self.revision

    def _resolve_tag_precision(self) -> None:
        requested = self._tag_precision_requested
        if requested not in {"auto", "float32", "bfloat16"}:
            raise ModelError(
                "ANIME_PICKER_TAG_DTYPE must be one of auto, float32, or bfloat16"
            )
        self._prepare_runtime()
        try:
            torch_module = importlib.import_module("torch")
        except Exception as error:
            if self.require_gpu:
                raise GPURequiredError(f"PyTorch is required for PixAI precision selection: {error}") from error
            self._tag_precision = "float32"
            self._tag_precision_reason = "torch unavailable; CPU float32"
            return
        if not self.require_gpu:
            # CPU inference stays in the model's native float32 path even when
            # a caller requests bfloat16; this keeps the explicit test mode
            # portable across CPU architectures and avoids an accidental CPU
            # autocast fallback.
            self._tag_precision = "float32"
            self._tag_precision_reason = "require_gpu=False; CPU float32"
            return
        if requested == "float32":
            self._tag_precision = "float32"
            self._tag_precision_reason = "explicit float32"
            return
        try:
            bf16_supported = bool(torch_module.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False
        if requested == "bfloat16" and not bf16_supported:
            raise GPURequiredError("bfloat16 PixAI inference was requested but this CUDA device does not support it")
        if bf16_supported:
            self._tag_precision = "bfloat16"
            self._tag_precision_reason = "CUDA bfloat16 autocast"
        else:
            # ``auto`` has an explicit, observable GPU float32 fallback.  It
            # never changes the device or retries on CPU.
            self._tag_precision = "float32"
            self._tag_precision_reason = "CUDA bfloat16 unsupported; GPU float32 fallback"

    @staticmethod
    def _manifest_revision(manifest: Any) -> str:
        if isinstance(manifest, Mapping):
            revision = manifest.get("revision") or manifest.get("manifest_sha256")
            if isinstance(revision, str) and revision:
                return revision
            payload = {key: value for key, value in manifest.items() if key not in {"revision", "manifest_sha256"}}
            return hashlib.sha256(_canonical_json(payload)).hexdigest()
        return "unprepared"

    def _configure_cache_env(self) -> None:
        # Override inherited cache settings so an inference process cannot
        # accidentally fetch or read model material outside the prescribed root.
        cache_root = self.model_root / "cache"
        paths = {
            "HF_HOME": cache_root / "huggingface",
            "HF_HUB_CACHE": cache_root / "huggingface" / "hub",
            "HF_MODULES_CACHE": cache_root / "huggingface" / "modules",
            "HF_ASSETS_CACHE": cache_root / "huggingface" / "assets",
            "HF_DATASETS_CACHE": cache_root / "huggingface" / "datasets",
            "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers",
            "IMGUTILS_CACHE": cache_root / "imgutils",
            "IMGUTILS_CACHE_DIR": cache_root / "imgutils",
            "TORCH_HOME": cache_root / "torch",
            "XDG_CACHE_HOME": cache_root,
        }
        for key, value in paths.items():
            os.environ[key] = str(value)
        # Runtime must stay offline.  prepare() temporarily removes this flag
        # around snapshot_download and restores it before returning.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    @staticmethod
    def status(model_root: Path | str | None = None) -> dict[str, Any]:
        """Return setup status without importing a model runtime or using a network."""

        root = _root_from_argument(model_root)
        marker = root / ".status"
        manifest_path = root / "manifest.json"
        if not root.is_dir():
            return {"ready": False, "details": f"model root does not exist: {root}"}
        if not marker.is_file():
            return {"ready": False, "details": "model bundle is not prepared (.status is missing)"}
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, Mapping):
            return {"ready": False, "details": "model manifest is missing or invalid"}
        status_data = _read_json(marker)
        if isinstance(status_data, Mapping):
            marker_ready = bool(status_data.get("ready"))
        else:
            marker_ready = marker.read_text(encoding="utf-8", errors="ignore").strip().lower() in {"ready", "true", "1"}
        if not marker_ready:
            return {"ready": False, "details": "model preparation did not complete"}
        entries = manifest.get("files", [])
        missing: list[str] = []
        if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, bytearray)):
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                path_value = entry.get("path")
                path = _relative_file(root, path_value)
                if path is None or not path.is_file():
                    missing.append(str(path_value or "<unknown>"))
                    continue
                size = entry.get("size")
                if size is not None:
                    try:
                        if path.stat().st_size != int(size):
                            missing.append(str(path_value))
                    except (OSError, TypeError, ValueError):
                        missing.append(str(path_value))
        if missing:
            return {"ready": False, "details": f"model files are missing or incomplete: {', '.join(missing[:3])}"}
        return {
            "ready": True,
            "details": f"local model bundle ready ({manifest.get('revision', 'manifest verified')})",
        }

    def _resolver(self, repo_id: str, filename: str, **_: Any) -> str:
        """Resolve an imgutils Hub request to an audited local file."""

        repo = str(repo_id)
        filename = str(filename).replace("\\", "/").lstrip("/")
        if repo == "deepghs/ccip_onnx":
            base = self.model_root / "ccip-caformer_b36-24"
        elif repo == PERSON_REPOSITORY:
            base = self.model_root / "person"
        elif repo == HEAD_REPOSITORY:
            base = self.model_root / "head"
        else:
            # The resolver is deliberately conservative: only known model
            # repositories can be used by an inference process.
            raise ModelUnavailable(f"offline model repository is not prepared: {repo}")
        candidate = (base / filename).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError as error:
            raise ModelUnavailable(f"invalid local model path: {filename}") from error
        if not candidate.is_file():
            # A few snapshots flatten one-file model directories.  Matching by
            # basename keeps the pinned repository layout flexible while still
            # refusing any file outside model_root.
            fallback = next((path for path in base.rglob(Path(filename).name) if path.is_file()), None)
            if fallback is not None:
                candidate = fallback
        if not candidate.is_file():
            # imgutils probes optional metadata files (currently
            # ``model_type.json`` and ``threshold.json``) and handles the Hub
            # EntryNotFoundError by selecting its built-in defaults.  Raising
            # that same sentinel keeps the resolver offline while preserving
            # the official detector behavior.
            if Path(filename).name in {"model_type.json", "threshold.json"}:
                try:
                    hub_errors = importlib.import_module("huggingface_hub.errors")
                    raise getattr(hub_errors, "EntryNotFoundError")(f"offline optional file is absent: {filename}")
                except ImportError:
                    raise FileNotFoundError(filename)
            raise ModelUnavailable(f"prepared model file is missing: {base / filename}")
        return str(candidate)

    def _prepare_runtime(self) -> None:
        if self._runtime_checked:
            if self._runtime_error:
                raise ModelUnavailable(self._runtime_error)
            return
        self._runtime_checked = True
        # Import and touch PyTorch first.  On Windows this loads the CUDA DLLs
        # before ONNX Runtime gets imported, which avoids a common provider
        # mismatch caused by DLL search order.
        torch_module: Any = None
        try:
            torch_module = importlib.import_module("torch")
            cuda_available = bool(torch_module.cuda.is_available())
            if cuda_available:
                try:
                    torch_module.cuda.current_device()
                except Exception:
                    pass
        except Exception as error:
            cuda_available = False
            if self.require_gpu:
                self._runtime_error = f"PyTorch CUDA runtime is unavailable: {error}"
                raise GPURequiredError(self._runtime_error) from error
        if self.require_gpu:
            if not cuda_available:
                self._runtime_error = "CUDA is required but torch.cuda.is_available() is false"
                raise GPURequiredError(self._runtime_error)
            try:
                ort = importlib.import_module("onnxruntime")
                providers = list(ort.get_available_providers())
            except Exception as error:
                self._runtime_error = f"CUDA ONNX Runtime is unavailable: {error}"
                raise GPURequiredError(self._runtime_error) from error
            if "CUDAExecutionProvider" not in providers:
                self._runtime_error = f"CUDAExecutionProvider is unavailable (providers: {providers})"
                raise GPURequiredError(self._runtime_error)
            os.environ["ONNX_MODE"] = "gpu"
        else:
            os.environ["ONNX_MODE"] = "cpu"

    def _patch_imgutils_modules(self) -> None:
        modules = (
            "imgutils.metrics.ccip",
            "imgutils.generic.yolo",
            "imgutils.detect.person",
            "imgutils.detect.head",
        )
        for name in modules:
            try:
                module = importlib.import_module(name)
            except Exception:
                continue
            _patch_hf_resolver(module, self._resolver)
            # generic.yolo imports open_onnx_model into its own module.  It
            # reads ONNX_MODE, configured above, and therefore enforces the
            # chosen provider for detector sessions too.

    def _session_providers(self, session: Any) -> list[str]:
        try:
            providers = session.get_providers()
            return [str(item) for item in providers]
        except Exception:
            return []

    def _validate_ccip_session_provider(self) -> None:
        if not self.require_gpu or self._ccip_module is None:
            return
        providers: list[str] = []
        for name in ("_open_feat_model", "_open_metric_model"):
            opener = getattr(self._ccip_module, name, None)
            if not callable(opener):
                continue
            try:
                providers.extend(self._session_providers(opener(self.identity_model)))
            except Exception as error:
                raise ModelUnavailable(f"unable to open local CCIP session: {error}") from error
        if providers and "CUDAExecutionProvider" not in providers:
            raise GPURequiredError(f"CCIP ONNX session is not using CUDAExecutionProvider: {providers}")

    def _ensure_ccip(self, *, need_extract: bool = False, need_distance: bool = False) -> None:
        global ccip_batch_extract_features, ccip_batch_differences, ccip_difference
        self._prepare_runtime()
        self._patch_imgutils_modules()
        missing_requested = (need_extract and ccip_batch_extract_features is None) or (
            need_distance and ccip_batch_differences is None and ccip_difference is None
        )
        if missing_requested:
            try:
                module = importlib.import_module("imgutils.metrics")
                ccip_module = importlib.import_module("imgutils.metrics.ccip")
            except Exception as error:
                raise ModelUnavailable(f"imgutils CCIP is unavailable: {error}") from error
            if ccip_batch_extract_features is None:
                ccip_batch_extract_features = getattr(module, "ccip_batch_extract_features", None)
            if ccip_batch_differences is None:
                ccip_batch_differences = getattr(module, "ccip_batch_differences", None)
            if ccip_difference is None:
                ccip_difference = getattr(module, "ccip_difference", None)
            self._ccip_module = ccip_module
        elif self._ccip_module is None:
            # A test double may provide the public functions without the
            # package; provider inspection is then intentionally unavailable.
            try:
                self._ccip_module = importlib.import_module("imgutils.metrics.ccip")
            except Exception:
                self._ccip_module = None
        if need_extract and not callable(ccip_batch_extract_features):
            raise ModelUnavailable("imgutils ccip_batch_extract_features is unavailable")
        if need_distance and not callable(ccip_batch_differences or ccip_difference):
            raise ModelUnavailable("imgutils CCIP distance functions are unavailable")
        if ccip_batch_extract_features is None and ccip_batch_differences is None and ccip_difference is None:
            raise ModelUnavailable("imgutils CCIP functions are unavailable")
        self._validate_ccip_session_provider()

    def _ensure_detector(self) -> None:
        global detect_person, detect_heads, detect_head
        self._prepare_runtime()
        self._patch_imgutils_modules()
        if detect_person is None:
            try:
                module = importlib.import_module("imgutils.detect")
            except Exception as error:
                raise ModelUnavailable(f"imgutils person detector is unavailable: {error}") from error
            detect_person = getattr(module, "detect_person", None)
            detect_heads = getattr(module, "detect_heads", None)
            detect_head = getattr(module, "detect_head", None)
        if not callable(detect_person):
            raise ModelUnavailable("imgutils detect_person is unavailable")

    def _tagger_path(self) -> Path:
        path = _first_file(self.model_root, ("pixai-tagger-v1.0/config.json", "pixai/config.json", "tagger/config.json"))
        if path is None:
            for candidate in (self.model_root / "pixai-tagger-v1.0", self.model_root / "pixai", self.model_root / "tagger"):
                if (candidate / "config.json").is_file():
                    return candidate
            raise ModelUnavailable("PixAI tagger is not prepared locally")
        return path.parent

    def _load_tagger(self) -> Any:
        if self._tagger is not None:
            return self._tagger
        self._prepare_runtime()
        path = self._tagger_path()
        try:
            transformers = importlib.import_module("transformers")
            factory = getattr(transformers, "pipeline")
        except Exception as error:
            raise ModelUnavailable(f"Transformers is unavailable: {error}") from error
        kwargs = {
            "model": str(path),
            "image_processor": str(path),
            "trust_remote_code": True,
            "local_files_only": True,
            "cache_dir": str(self.model_root / "cache" / "huggingface"),
            "device": 0 if self.require_gpu else -1,
        }
        try:
            self._tagger = factory(**kwargs)
        except TypeError:
            # Older Transformers versions infer the task only when explicitly
            # supplied, while lightweight test doubles often accept one arg.
            try:
                self._tagger = factory("image-classification", **kwargs)
            except TypeError:
                kwargs.pop("image_processor", None)
                kwargs.pop("cache_dir", None)
                self._tagger = factory("image-classification", **kwargs)
        except Exception as error:
            raise ModelUnavailable(f"unable to load local PixAI tagger: {error}") from error
        return self._tagger

    def vocabulary(self) -> dict[str, str]:
        """Return every local PixAI tag mapped to its category."""

        if self._tag_vocabulary is not None:
            return dict(self._tag_vocabulary)
        path = self._tagger_path()
        config = _read_json(path / "config.json")
        if not isinstance(config, Mapping):
            raise ModelUnavailable("PixAI config.json is missing or invalid")
        result: dict[str, str] = {}

        def add(name: Any, category: Any = "general") -> None:
            if isinstance(name, Mapping):
                tag_name = name.get("tag", name.get("name", name.get("label")))
                tag_category = name.get("category", name.get("group", category))
            else:
                tag_name, tag_category = name, category
            if tag_name is None:
                return
            text = str(tag_name).strip()
            if text:
                result[text] = str(tag_category or "general")

        tags = config.get("tags", config.get("labels", config.get("vocabulary")))
        splits = config.get("tags_split", config.get("tag_splits", []))
        split_ranges: list[tuple[str, int]] = []
        if isinstance(splits, Sequence) and not isinstance(splits, (str, bytes, bytearray)):
            for split in splits:
                if isinstance(split, Sequence) and not isinstance(split, (str, bytes, bytearray)) and len(split) >= 2:
                    try:
                        split_ranges.append((str(split[0]), int(split[1])))
                    except (TypeError, ValueError):
                        continue
        if isinstance(tags, Mapping):
            for category, values in tags.items():
                if isinstance(values, Mapping):
                    for name, _value in values.items():
                        add(name, category)
                elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                    for value in values:
                        add(value, category)
        elif isinstance(tags, Sequence) and not isinstance(tags, (str, bytes, bytearray)):
            index = 0
            for tag in tags:
                category = "general"
                offset = 0
                for split_category, count in split_ranges:
                    if index < offset + count:
                        category = split_category
                        break
                    offset += count
                add(tag, category)
                index += 1
        id2label = config.get("id2label")
        if isinstance(id2label, Mapping):
            categories = config.get("tag_categories", {})
            for index, name in id2label.items():
                if isinstance(categories, Mapping):
                    category = categories.get(str(index), "general")
                    if category == "general":
                        try:
                            category = categories.get(int(index), category)
                        except (TypeError, ValueError):
                            pass
                else:
                    category = "general"
                add(name, category)
        categories = config.get("categories")
        if isinstance(categories, Mapping):
            for category, values in categories.items():
                if isinstance(values, Mapping):
                    for name in values:
                        add(name, category)
                elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                    for value in values:
                        add(value, category)
        self._tag_vocabulary = result
        return dict(result)

    def detect(self, image: Any) -> list[dict[str, Any]]:
        """Detect anime people and, when available, anime heads in pixel space."""

        self._ensure_detector()
        width, height = _image_size(image)
        try:
            raw_person = detect_person(image, level="m", version="v1.1", conf_threshold=0.3, iou_threshold=0.5)  # type: ignore[misc]
        except TypeError:
            raw_person = detect_person(image)  # type: ignore[misc]
        result: list[dict[str, Any]] = []
        for item in raw_person or []:
            parsed = _normalise_detection(item, width, height)
            if parsed is not None:
                parsed["kind"] = "person"
                result.append(parsed)
        head_detector = detect_heads or detect_head
        if callable(head_detector):
            try:
                raw_head = head_detector(image, model_name="head_detect_v2.0_s", conf_threshold=0.3, iou_threshold=0.5)
            except TypeError:
                try:
                    raw_head = head_detector(image)
                except Exception:
                    raw_head = []
            for item in raw_head or []:
                parsed = _normalise_detection(item, width, height)
                if parsed is not None:
                    parsed["kind"] = "head"
                    result.append(parsed)
        return result

    def embed(self, images: list[Any]) -> np.ndarray:
        """Extract official imgutils CCIP feature vectors for ``images``."""

        if not images:
            return np.empty((0, 0), dtype=np.float32)
        self._ensure_ccip(need_extract=True)
        if not callable(ccip_batch_extract_features):
            raise ModelUnavailable("imgutils ccip_batch_extract_features is unavailable")
        try:
            output = ccip_batch_extract_features(images, size=384, model=self.identity_model)
        except TypeError:
            output = ccip_batch_extract_features(images)  # type: ignore[misc]
        array = np.asarray(output, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[0] != len(images):
            raise ModelError(f"CCIP feature extractor returned {array.shape}; expected ({len(images)}, features)")
        return array

    def distance(self, ref_features: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Return the official CCIP distance matrix (references x images)."""

        refs = np.asarray(ref_features, dtype=np.float32)
        values = np.asarray(features, dtype=np.float32)
        if refs.ndim == 1:
            refs = refs.reshape(1, -1)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if refs.ndim != 2 or values.ndim != 2:
            raise ValueError("CCIP features must be two-dimensional arrays")
        if refs.shape[1:] != values.shape[1:] and refs.size and values.size:
            raise ValueError(f"CCIP feature widths differ: {refs.shape[1:]} and {values.shape[1:]}")
        if refs.shape[0] == 0 or values.shape[0] == 0:
            return np.empty((refs.shape[0], values.shape[0]), dtype=np.float32)
        self._ensure_ccip(need_distance=True)
        combined = np.concatenate((refs, values), axis=0).astype(np.float32, copy=False)
        if callable(ccip_batch_differences):
            try:
                matrix = ccip_batch_differences(combined, size=384, model=self.identity_model)
            except TypeError:
                matrix = ccip_batch_differences(combined)  # type: ignore[misc]
            matrix = np.asarray(matrix, dtype=np.float32)
            expected = combined.shape[0]
            if matrix.shape != (expected, expected):
                raise ModelError(f"CCIP distance function returned {matrix.shape}; expected {(expected, expected)}")
            return matrix[: refs.shape[0], refs.shape[0] :].astype(np.float32, copy=False)
        if not callable(ccip_difference):
            raise ModelUnavailable("imgutils CCIP distance functions are unavailable")
        output = np.empty((refs.shape[0], values.shape[0]), dtype=np.float32)
        for row, reference in enumerate(refs):
            for column, value in enumerate(values):
                try:
                    distance = ccip_difference(reference, value, size=384, model=self.identity_model)
                except TypeError:
                    distance = ccip_difference(reference, value)  # type: ignore[misc]
                output[row, column] = _safe_float(distance, math.inf)
        return output

    def _normalise_tags(self, raw: Any) -> dict[str, dict[str, float]]:
        if isinstance(raw, Mapping) and "results" in raw:
            raw = raw["results"]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            if len(raw) == 1 and isinstance(raw[0], Mapping):
                raw = raw[0]
            else:
                flat: dict[str, float] = {}
                vocab = self.vocabulary()
                for item in raw:
                    if not isinstance(item, Mapping):
                        continue
                    name = item.get("label", item.get("tag", item.get("name")))
                    score = item.get("score", item.get("confidence", item.get("probability")))
                    category = vocab.get(str(name), "general")
                    if name is not None:
                        flat[str(name)] = _safe_float(score, 0.0)
                return {"general": flat} if flat else {}
        if not isinstance(raw, Mapping):
            return {}
        result: dict[str, dict[str, float]] = {}
        for category, values in raw.items():
            if isinstance(values, Mapping):
                result[str(category)] = {
                    str(tag): max(0.0, min(1.0, _safe_float(score, 0.0)))
                    for tag, score in values.items()
                    if str(tag).strip()
                }
            elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                converted: dict[str, float] = {}
                for item in values:
                    if isinstance(item, Mapping):
                        name = item.get("tag", item.get("label", item.get("name")))
                        score = item.get("score", item.get("confidence", 0.0))
                        if name is not None:
                            converted[str(name)] = max(0.0, min(1.0, _safe_float(score, 0.0)))
                result[str(category)] = converted
        return result

    def tag(self, image: Any) -> dict[str, dict[str, float]]:
        """Return low-cutoff PixAI scores grouped by the model's categories."""

        tagger = self._load_tagger()
        precision = self.tag_precision
        try:
            torch_module = importlib.import_module("torch") if precision == "bfloat16" else None
            autocast = (
                torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)
                if torch_module is not None
                else nullcontext()
            )
            inference = torch_module.inference_mode() if torch_module is not None else nullcontext()
            with inference, autocast:
                try:
                    raw = tagger(image, threshold=0.0, min_threshold=0.0)
                except TypeError:
                    try:
                        raw = tagger(image, threshold=0.0)
                    except TypeError:
                        raw = tagger(image)
        except Exception as error:
            if precision == "bfloat16" and self.require_gpu:
                raise ModelError(f"CUDA bfloat16 PixAI inference failed; refusing CPU fallback: {error}") from error
            raise
        return self._normalise_tags(raw)

    def diagnostics(self) -> dict[str, Any]:
        """Report runtime, provider, and local bundle evidence for ``doctor``."""

        status = self.status(self.model_root)
        try:
            service_revision = self.revision
        except Exception as error:
            service_revision = f"{self.manifest_revision}:tag-unresolved"
            self._tag_precision_reason = str(error)
        diagnostics: dict[str, Any] = {
            "ready": bool(status.get("ready")),
            "details": status.get("details", ""),
            "model_root": str(self.model_root),
            "require_gpu": self.require_gpu,
            "identity_model": self.identity_model,
            "threshold": self.threshold,
            "revision": service_revision,
            "manifest_revision": self.manifest_revision,
            "identity_revision": self.identity_revision,
            "tag_revision": service_revision,
            "tag_precision": self._tag_precision,
            "tag_precision_requested": self._tag_precision_requested,
            "tag_precision_reason": self._tag_precision_reason,
            "torch_cuda_available": None,
            "onnx_available_providers": [],
            "onnx_active_providers": [],
            "cuda_execution_provider_active": False,
            "cache_root": str(self.model_root / "cache"),
        }
        try:
            torch_module = importlib.import_module("torch")
            diagnostics["torch_cuda_available"] = bool(torch_module.cuda.is_available())
        except Exception as error:
            diagnostics["torch_error"] = str(error)
        try:
            ort = importlib.import_module("onnxruntime")
            diagnostics["onnx_available_providers"] = list(ort.get_available_providers())
        except Exception as error:
            diagnostics["onnx_error"] = str(error)
        sessions: list[Any] = []
        if self._ccip_module is not None:
            for name in ("_open_feat_model", "_open_metric_model"):
                opener = getattr(self._ccip_module, name, None)
                if callable(opener):
                    try:
                        sessions.append(opener(self.identity_model))
                    except Exception as error:
                        diagnostics.setdefault("session_errors", []).append(str(error))
        active = []
        for session in sessions:
            active.extend(self._session_providers(session))
        diagnostics["onnx_active_providers"] = sorted(set(active))
        diagnostics["cuda_execution_provider_active"] = "CUDAExecutionProvider" in active
        if self._runtime_error:
            diagnostics["runtime_error"] = self._runtime_error
        return diagnostics

    @classmethod
    def prepare(
        cls,
        model_root: Path | str | None = None,
        progress: Callable[[float, str], Any] | None = None,
    ) -> dict[str, Any]:
        """Download and verify the pinned offline bundle."""

        root = _root_from_argument(model_root)
        root.mkdir(parents=True, exist_ok=True)
        progress = progress or (lambda _value, _message: None)
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        previous_datasets_offline = os.environ.get("HF_DATASETS_OFFLINE")
        try:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("HF_DATASETS_OFFLINE", None)
            repos = [
                {
                    "id": PIXAI_REPOSITORY,
                    "revision": PIXAI_REVISION,
                    "path": "pixai-tagger-v1.0",
                    "license": "Apache-2.0",
                    "patterns": ["config.json", "preprocessor_config.json", "tagger_pipeline.py", "model.safetensors", "README.md", "LICENSE*"],
                    "required": ["config.json", "preprocessor_config.json", "tagger_pipeline.py", "model.safetensors"],
                    "audit": True,
                },
                {
                    "id": "deepghs/ccip_onnx",
                    "revision": CCIP_REVISION,
                    "path": "ccip-caformer_b36-24",
                    "license": "OpenRAIL",
                    "model_license": "OpenRAIL",
                    "patterns": ["ccip-caformer_b36-24/model_feat.onnx", "ccip-caformer_b36-24/model_metrics.onnx", "ccip-caformer_b36-24/metrics.json", "README.md", "LICENSE*", "ccip-caformer_b36-24/README.md", "ccip-caformer_b36-24/LICENSE*"],
                    "required": ["model_feat.onnx", "model_metrics.onnx", "metrics.json"],
                    "audit": False,
                },
                {
                    "id": PERSON_REPOSITORY,
                    "revision": PERSON_REVISION,
                    "path": "person",
                    "license": "MIT",
                    "model_license": "GPL-3.0",
                    "model_source": "Ultralytics 8.0.20",
                    "patterns": ["person_detect_v1.1_m/model.onnx", "README.md", "LICENSE*"],
                    "required": ["model.onnx"],
                    "audit": False,
                },
                {
                    "id": HEAD_REPOSITORY,
                    "revision": HEAD_REVISION,
                    "path": "head",
                    "license": "MIT",
                    "model_license": "AGPL-3.0",
                    "model_source": "Ultralytics 8.0.196",
                    "patterns": ["head_detect_v2.0_s/model.onnx", "README.md", "LICENSE*"],
                    "required": ["model.onnx"],
                    "audit": False,
                },
            ]
            manifest_repos: list[dict[str, Any]] = []
            for index, spec in enumerate(repos):
                progress(index / len(repos), f"downloading {spec['id']} at {spec['revision']}")
                destination = root / str(spec["path"])
                cls._download_repository(root, destination, spec)
                required = list(spec["required"])
                missing = []
                for name in required:
                    matches = list(destination.rglob(name))
                    if not matches:
                        missing.append(name)
                if missing:
                    raise PrepareError(f"{spec['id']} did not provide required files: {', '.join(missing)}")
                audits: dict[str, Any] = {}
                if spec.get("audit"):
                    source = destination / "tagger_pipeline.py"
                    audits["tagger_pipeline.py"] = _audit_python_file(source)
                    if not audits["tagger_pipeline.py"].get("passed"):
                        raise PrepareError("PixAI custom code failed the outbound-call audit")
                files: list[dict[str, Any]] = []
                for path in sorted(destination.rglob("*")):
                    # huggingface_hub creates mutable download metadata below
                    # ``.cache`` inside local_dir.  Those files are transport
                    # state rather than model artifacts and must not enter the
                    # immutable manifest (their contents legitimately change
                    # between resumable prepare runs).
                    if path.is_file() and ".cache" not in path.parts and not path.name.endswith((".lock", ".incomplete")):
                        files.append({"path": str(path.relative_to(root)).replace("\\", "/"), "size": path.stat().st_size, "sha256": _sha256_path(path)})
                manifest_repos.append({
                    "id": spec["id"],
                    "revision": spec["revision"],
                    "path": spec["path"],
                    "license": spec["license"],
                    "model_license": spec.get("model_license", spec["license"]),
                    "model_source": spec.get("model_source"),
                    "audits": audits,
                    "files": files,
                })
            all_files = [file for repo in manifest_repos for file in repo["files"]]
            manifest: dict[str, Any] = {
                "schema": 1,
                "identity_model": IDENTITY_MODEL,
                "threshold": IDENTITY_THRESHOLD,
                "repos": manifest_repos,
                "files": all_files,
            }
            manifest["revision"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
            _write_json_atomic(root / "manifest.json", manifest)
            _write_json_atomic(root / ".status", {"ready": True, "revision": manifest["revision"]})
            progress(1.0, "model bundle ready")
            return cls.status(root)
        except Exception:
            # A stale ready marker must never survive an incomplete prepare.
            try:
                (root / ".status").unlink(missing_ok=True)
            except OSError:
                pass
            raise
        finally:
            if previous_offline is None:
                os.environ["HF_HUB_OFFLINE"] = "1"
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline
            if previous_datasets_offline is None:
                os.environ["HF_DATASETS_OFFLINE"] = "1"
            else:
                os.environ["HF_DATASETS_OFFLINE"] = previous_datasets_offline

    @staticmethod
    def _download_repository(root: Path, destination: Path, spec: Mapping[str, Any]) -> None:
        global snapshot_download
        if snapshot_download is None:
            try:
                hub = importlib.import_module("huggingface_hub")
                snapshot_download = getattr(hub, "snapshot_download")
            except Exception as error:
                raise PrepareError(f"huggingface_hub is unavailable: {error}") from error
        destination.mkdir(parents=True, exist_ok=True)
        kwargs = {
            "repo_id": spec["id"],
            "revision": spec["revision"],
            "local_dir": str(destination),
            "cache_dir": str(root / "cache" / "huggingface" / "hub"),
            "allow_patterns": list(spec.get("patterns", [])),
            "local_dir_use_symlinks": False,
        }
        try:
            snapshot_download(**kwargs)
        except TypeError:
            kwargs.pop("local_dir_use_symlinks", None)
            snapshot_download(**kwargs)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the pinned local anime model bundle")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="download and verify model files")
    prepare_parser.add_argument("--model-root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        def report(value: float, message: str) -> None:
            print(f"[{value * 100:6.2f}%] {message}", flush=True)

        result = ModelService.prepare(args.model_root, progress=report)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("ready") else 1
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI
    raise SystemExit(_main())
