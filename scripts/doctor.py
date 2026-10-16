"""Check the local runtime, CUDA providers, and optional model assets.

The checks deliberately exercise the runtime instead of only checking that a
package can be imported.  In particular, ``onnxruntime.get_available_providers``
can list CUDA while a real session silently falls back to CPU, so this script
loads and runs a tiny ONNX graph and checks the session provider order.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
DEFAULT_MODEL_ROOT = Path(r"D:\AI\models\Cold_F\image")
DEFAULT_OUTPUT_ROOT = (ROOT / "outputs").resolve()
DEFAULT_LOGO_MODEL_SUBDIR = "anime-lama"
DEFAULT_LOGO_MODEL_ROOT = DEFAULT_MODEL_ROOT / DEFAULT_LOGO_MODEL_SUBDIR
DEFAULT_LOGO_PROFILE_ROOT = ROOT / ".local" / "logo-profiles"
_DLL_HANDLES: list[Any] = []

# Executing ``python scripts\\doctor.py`` places ``scripts`` (rather than the
# repository root) first on sys.path.  Add the project root explicitly so the
# model status check sees the same local ``anime_picker`` package as `python -m`.
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))
try:
    # Keep direct ``cmd.exe`` runs from failing on a diagnostic character that
    # the active Windows code page cannot represent.
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass


class Report:
    """Collect check results while keeping output readable in setup logs."""

    def __init__(self, *, as_json: bool = False) -> None:
        self.as_json = as_json
        self.results: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        self.results.append({"name": name, "ok": ok, "detail": detail, "required": required})
        if self.as_json:
            return
        marker = "OK" if ok else ("WARN" if not required else "FAIL")
        print(f"[{marker:4}] {name}: {detail}")

    @property
    def failed(self) -> bool:
        return any(not result["ok"] and result["required"] for result in self.results)

    def emit_json(self) -> None:
        if self.as_json:
            print(json.dumps(self.results, ensure_ascii=False, indent=2, default=str))


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _path_from_environment(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    path = Path(value).expanduser()
    if not path.is_absolute() and (path.drive or path.root):
        raise ValueError("path must be repository-relative or absolute")
    return path if path.is_absolute() else (ROOT / path).resolve()


def _configure_cuda_dll_search_path() -> None:
    """Expose Torch's bundled CUDA 12.8 DLLs to ONNX Runtime on Windows.

    The CUDA wheel keeps its DLLs inside ``torch/lib``. Windows does not
    search that directory when ONNX Runtime loads its CUDA provider, so add it
    to this process before importing either runtime. This does not modify the
    system PATH on disk.
    """

    if os.name != "nt":
        return
    torch_lib = VENV / "Lib" / "site-packages" / "torch" / "lib"
    if not torch_lib.is_dir():
        return
    current_path = os.environ.get("PATH", "")
    if os.fspath(torch_lib) not in current_path.split(os.pathsep):
        os.environ["PATH"] = os.fspath(torch_lib) + os.pathsep + current_path
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        try:
            _DLL_HANDLES.append(add_dll_directory(os.fspath(torch_lib)))
        except OSError:
            # PATH is still useful on older Windows/Python combinations.
            pass


def _check_python(report: Report) -> None:
    expected_prefix = VENV.resolve()
    actual_prefix = Path(sys.prefix).resolve()
    report.add(
        "python environment",
        actual_prefix == expected_prefix,
        f"{sys.executable} (prefix={actual_prefix})"
        if actual_prefix == expected_prefix
        else f"using {sys.executable}; run .venv\\Scripts\\python.exe or setup.bat",
    )
    version_ok = sys.version_info[:2] == (3, 12)
    report.add(
        "python version",
        version_ok,
        platform.python_version() if version_ok else f"{platform.python_version()} (requires 3.12.x)",
    )


def _check_imports(report: Report) -> None:
    packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "python-multipart": "multipart",
        "PyAV": "av",
        "Pillow": "PIL",
        "NumPy": "numpy",
        "OpenCV": "cv2",
        "dghs-imgutils": "imgutils",
        "Transformers": "transformers",
        "timm": "timm",
        "safetensors": "safetensors",
        "accelerate": "accelerate",
        "huggingface-hub": "huggingface_hub",
        "httpx": "httpx",
    }
    for distribution, module_name in packages.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # import errors can include native DLL errors
            report.add(
                f"import {distribution}",
                False,
                f"{type(exc).__name__}: {exc}; reinstall from requirements.txt",
            )
        else:
            report.add(
                f"import {distribution}",
                True,
                _version(distribution) or "imported (version metadata unavailable)",
            )


def _check_onnx(report: Report) -> None:
    try:
        ort = importlib.import_module("onnxruntime")
    except Exception as exc:
        report.add("ONNX Runtime import", False, f"{type(exc).__name__}: {exc}")
        return

    cpu_dist = _version("onnxruntime")
    gpu_dist = _version("onnxruntime-gpu")
    if cpu_dist and gpu_dist:
        report.add(
            "ONNX Runtime distribution",
            False,
            f"CPU {cpu_dist} and GPU {gpu_dist} are both installed; uninstall onnxruntime, then reinstall onnxruntime-gpu",
        )
    elif gpu_dist:
        report.add("ONNX Runtime distribution", True, f"onnxruntime-gpu {gpu_dist}")
    elif cpu_dist:
        report.add(
            "ONNX Runtime distribution",
            False,
            f"CPU-only onnxruntime {cpu_dist} is installed; install onnxruntime-gpu and remove onnxruntime",
        )
    else:
        report.add("ONNX Runtime distribution", False, "neither onnxruntime nor onnxruntime-gpu metadata is installed")

    try:
        available = list(ort.get_available_providers())
    except Exception as exc:
        report.add("ONNX Runtime providers", False, f"provider query failed: {exc}")
        return
    cuda_available = "CUDAExecutionProvider" in available
    report.add(
        "ONNX CUDA provider available",
        cuda_available,
        ", ".join(available) if available else "no providers reported",
    )
    if not cuda_available:
        return

    # Loading this graph through CUDAExecutionProvider is runtime evidence
    # that the provider's native DLLs can initialize.  The graph is generated
    # in memory and removed immediately; it is not a model asset.
    try:
        onnx = importlib.import_module("onnx")
        import numpy as np

        input_info = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2])
        output_info = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["x"], ["y"])],
            "anime-picker-doctor",
            [input_info],
            [output_info],
        )
        model = onnx.helper.make_model(
            graph,
            producer_name="anime-frame-picker-doctor",
            opset_imports=[onnx.helper.make_opsetid("", 17)],
        )
        # ONNX 1.17 can emit IR versions newer than some ORT builds.  IR 9 is
        # sufficient for this graph and keeps the probe portable.
        model.ir_version = min(model.ir_version, 9)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as handle:
            model_path = Path(handle.name)
            handle.write(model.SerializeToString())
        try:
            session = ort.InferenceSession(
                os.fspath(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            active = list(session.get_providers())
            session.run(None, {"x": np.zeros((1, 2), dtype=np.float32)})
        finally:
            model_path.unlink(missing_ok=True)
        active_cuda = bool(active) and active[0] == "CUDAExecutionProvider"
        report.add(
            "ONNX CUDA session",
            active_cuda,
            f"active providers: {', '.join(active)}"
            if active
            else "session returned no providers",
        )
    except Exception as exc:
        report.add(
            "ONNX CUDA session",
            False,
            f"provider initialization or probe graph failed: {type(exc).__name__}: {exc}",
        )


def _check_torch(report: Report) -> None:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        report.add("Torch import", False, f"{type(exc).__name__}: {exc}")
        return

    torch_version = str(getattr(torch, "__version__", "unknown"))
    expected_pair = torch_version.startswith("2.9.1+cu128")
    report.add(
        "Torch CUDA build",
        expected_pair,
        f"torch {torch_version}, CUDA build {getattr(torch.version, 'cuda', None)}"
        if expected_pair
        else f"torch {torch_version}; expected 2.9.1+cu128",
    )
    try:
        cuda_ok = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_ok else 0
        device_name = torch.cuda.get_device_name(0) if cuda_ok else "unavailable"
    except Exception as exc:
        report.add("Torch CUDA runtime", False, f"CUDA query failed: {type(exc).__name__}: {exc}")
        return
    report.add(
        "Torch CUDA runtime",
        cuda_ok and device_count > 0,
        f"{device_count} device(s); {device_name}" if cuda_ok else "torch.cuda.is_available() is false",
    )
    try:
        torchvision = importlib.import_module("torchvision")
        vision_version = str(getattr(torchvision, "__version__", "unknown"))
        report.add(
            "TorchVision pair",
            vision_version.startswith("0.24.1+cu128"),
            f"torchvision {vision_version}"
            if vision_version.startswith("0.24.1+cu128")
            else f"torchvision {vision_version}; expected 0.24.1+cu128",
        )
    except Exception as exc:
        report.add("TorchVision pair", False, f"{type(exc).__name__}: {exc}")


def _check_paths(report: Report, *, skip_models: bool) -> None:
    model_root = _path_from_environment("ANIME_PICKER_MODEL_ROOT", DEFAULT_MODEL_ROOT)
    output_root = _path_from_environment("ANIME_PICKER_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    report.add("model root path", model_root.is_dir(), os.fspath(model_root))
    report.add("output root path", output_root.is_dir(), os.fspath(output_root))
    if skip_models:
        return

    try:
        models = importlib.import_module("anime_picker.models")
        service = getattr(models, "ModelService")
        status = service.status(model_root)
        if isinstance(status, dict):
            ready = status.get("ready", status.get("available", True))
            detail = json.dumps(status, ensure_ascii=False, default=str)
        else:
            ready = getattr(status, "ready", getattr(status, "available", status))
            detail = str(status)
        report.add("model assets", bool(ready), detail)
    except Exception as exc:
        report.add(
            "model assets",
            False,
            f"status check failed: {type(exc).__name__}: {exc}; run setup.bat to prepare models",
        )

    # Logo cleanup is an export-time enhancement.  A missing or invalid logo
    # model must be visible to operators, but it does not prevent the main
    # analysis pipeline from running with the logo toggle disabled.
    logo_model_root = _path_from_environment(
        "ANIME_PICKER_LOGO_MODEL_ROOT",
        model_root / DEFAULT_LOGO_MODEL_SUBDIR,
    )
    logo_profile_root = _path_from_environment(
        "ANIME_PICKER_LOGO_PROFILE_ROOT",
        DEFAULT_LOGO_PROFILE_ROOT,
    )
    try:
        logo_module = importlib.import_module("anime_picker.logo_removal")
        processor = getattr(logo_module, "LogoProcessor")
        status = processor.status(logo_model_root)
        if isinstance(status, dict):
            logo_ready = bool(status.get("ready", status.get("available", False)))
            detail = json.dumps(status, ensure_ascii=False, default=str)
        else:
            logo_ready = bool(getattr(status, "ready", getattr(status, "available", False)))
            detail = str(status)
        if not logo_ready:
            detail = (
                f"{detail}; logo cleanup is unavailable. Main analysis remains available "
                "with logo removal OFF; run setup.bat or `python -m anime_picker.logo_removal prepare`."
            )
        report.add("logo model", logo_ready, detail, required=False)

        store = getattr(logo_module, "ProfileStore")
        profiles = store(logo_profile_root).list()
        if profiles:
            report.add(
                "logo profiles",
                True,
                f"{len(profiles)} registered profile(s) under {logo_profile_root}",
                required=False,
            )
        else:
            report.add(
                "logo profiles",
                False,
                f"no registered profiles under {logo_profile_root}; logo matches are skipped with a warning",
                required=False,
            )
    except Exception as exc:
        report.add(
            "logo model",
            False,
            f"logo status check failed: {type(exc).__name__}: {exc}; logo cleanup is unavailable",
            required=False,
        )
        report.add(
            "logo profiles",
            False,
            f"logo profile status unavailable: {type(exc).__name__}: {exc}",
            required=False,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="check the base environment without requiring downloaded model assets",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable results instead of the human-readable table",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = Report(as_json=args.json)
    _configure_cuda_dll_search_path()
    _check_python(report)
    _check_imports(report)
    _check_torch(report)
    _check_onnx(report)
    _check_paths(report, skip_models=args.skip_models)
    report.emit_json()
    if report.failed:
        if not args.json:
            print("[doctor:fail] Fix the failed checks above, then rerun scripts\\doctor.py.")
        return 1
    if not args.json:
        print("[doctor:ok] runtime checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
