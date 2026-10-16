"""Create the project virtual environment and install its pinned runtime.

This script is intentionally executable with the system Python.  It creates
``.venv`` beside the repository, installs into that environment only, and
never imports or mutates another Python installation.  ``setup.bat`` is a
small wrapper around this module so the same workflow also works from CI.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / "Scripts" / "python.exe" if os.name == "nt" else VENV / "bin" / "python"
REQUIREMENTS = ROOT / "requirements.txt"
LOCK_REQUIREMENTS = ROOT / "requirements.lock.txt"
CACHE_DIR = ROOT / ".cache"
LOG_PATH = CACHE_DIR / "bootstrap.log"

DEFAULT_MODEL_ROOT = Path(r"D:\AI\models\Cold_F\image")
DEFAULT_OUTPUT_ROOT = (ROOT / "outputs").resolve()
DEFAULT_LOGO_MODEL_SUBDIR = "anime-lama"
DEFAULT_LOGO_MODEL_ROOT = DEFAULT_MODEL_ROOT / DEFAULT_LOGO_MODEL_SUBDIR
DEFAULT_LOGO_PROFILE_ROOT = ROOT / ".local" / "logo-profiles"


def _resolve_output_root(value: str | os.PathLike[str] | None = None) -> Path:
    """Resolve output paths against the repository and reject drive-relative paths."""

    raw = os.fspath(value) if value is not None else ""
    if not isinstance(raw, str) or "\x00" in raw:
        raise ValueError("invalid output root")
    if not raw.strip():
        return DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    if raw.startswith(("\\\\", "//")):
        raise ValueError("network paths are not allowed")
    path = Path(raw).expanduser()
    if not path.is_absolute() and (path.drive or path.root):
        raise ValueError("output root must be repository-relative or absolute")
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve(strict=False)


def _windows_subprocess_kwargs() -> dict[str, object]:
    """Prevent helper commands from opening a visible console on Windows."""

    if os.name != "nt":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


def _runtime_environment() -> dict[str, str]:
    """Return the environment shared by setup, model preparation, and doctor."""

    environment = os.environ.copy()
    model_root = Path(
        environment.get("ANIME_PICKER_MODEL_ROOT", os.fspath(DEFAULT_MODEL_ROOT))
    ).expanduser()
    output_root = _resolve_output_root(environment.get("ANIME_PICKER_OUTPUT_ROOT"))
    # Keep the values absolute and stable when a caller supplied a relative
    # path.  The application can therefore resolve them independently of cwd.
    if not model_root.is_absolute():
        model_root = (ROOT / model_root).resolve()
    logo_model_root = Path(
        environment.get(
            "ANIME_PICKER_LOGO_MODEL_ROOT",
            os.fspath(model_root / DEFAULT_LOGO_MODEL_SUBDIR),
        )
    ).expanduser()
    logo_profile_root = Path(
        environment.get("ANIME_PICKER_LOGO_PROFILE_ROOT", os.fspath(DEFAULT_LOGO_PROFILE_ROOT))
    ).expanduser()
    if not logo_model_root.is_absolute():
        logo_model_root = (ROOT / logo_model_root).resolve()
    if not logo_profile_root.is_absolute():
        logo_profile_root = (ROOT / logo_profile_root).resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    environment["ANIME_PICKER_MODEL_ROOT"] = os.fspath(model_root)
    environment["ANIME_PICKER_OUTPUT_ROOT"] = os.fspath(output_root)
    environment["ANIME_PICKER_LOGO_MODEL_ROOT"] = os.fspath(logo_model_root)
    environment["ANIME_PICKER_LOGO_PROFILE_ROOT"] = os.fspath(logo_profile_root)
    environment.setdefault("ANIME_PICKER_SOURCE_ROOT", os.fspath(ROOT / "source"))
    # Hugging Face honours these variables for transitive model downloads.
    # They stay under the configured model root and never create cache files in
    # the repository.
    environment.setdefault("HF_HOME", os.fspath(model_root / ".hf"))
    environment.setdefault("HF_HUB_CACHE", os.fspath(model_root / ".hf" / "hub"))
    environment.setdefault("PYTHONUTF8", "1")
    return environment


def _append_log(command: Sequence[str], return_code: int | None = None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(_quote_for_log(item) for item in command)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        if return_code is None:
            handle.write(f"\n$ {rendered}\n")
        else:
            handle.write(f"[exit {return_code}]\n")


def _quote_for_log(value: str) -> str:
    # Do not log environment values or shell-evaluated text.  Commands are
    # argv arrays, so this is only for a readable diagnostic transcript.
    if any(character.isspace() for character in value) or '"' in value:
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _run(command: Sequence[str], *, environment: dict[str, str], label: str) -> None:
    """Run a command, stream its output, and preserve a useful log on error."""

    command = [os.fspath(item) for item in command]
    print(f"[{label}] {' '.join(_quote_for_log(item) for item in command)}")
    _append_log(command)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=os.fspath(ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **_windows_subprocess_kwargs(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            _write_console(line)
            log.write(line)
        return_code = process.wait()
    _append_log(command, return_code)
    if return_code != 0:
        raise RuntimeError(
            f"{label} failed with exit code {return_code}. "
            f"See {LOG_PATH} for the complete output."
        )


def _write_console(text: str) -> None:
    """Write subprocess output even when a Windows console uses CP932.

    Model and library diagnostics can contain replacement characters or
    non-Japanese Unicode.  ``print`` raises before it can apply replacement
    handling, so use the stream's byte writer as a safe fallback.
    """

    try:
        print(text, end="")
    except UnicodeEncodeError:
        stream = getattr(sys.stdout, "buffer", None)
        if stream is None:
            return
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        stream.write(text.encode(encoding, errors="replace"))
        stream.flush()


def _python_version(path: Path) -> tuple[int, int, int] | None:
    try:
        completed = subprocess.run(
            [os.fspath(path), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
            cwd=os.fspath(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            **_windows_subprocess_kwargs(),
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    try:
        major, minor, patch = (int(part) for part in completed.stdout.strip().split(".")[:3])
    except (TypeError, ValueError):
        return None
    return major, minor, patch


def _ensure_pip(venv_python: Path, environment: dict[str, str]) -> None:
    """Keep pip available when uv created an unseeded environment."""

    try:
        completed = subprocess.run(
            [os.fspath(venv_python), "-m", "pip", "--version"],
            cwd=os.fspath(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            **_windows_subprocess_kwargs(),
        )
    except OSError:
        completed = None
    if completed is not None and completed.returncode == 0:
        return

    uv = shutil.which("uv")
    if uv:
        _run(
            [uv, "pip", "install", "--python", os.fspath(venv_python), "pip>=25,<27"],
            environment=environment,
            label="pip seed",
        )
    else:
        _run(
            [os.fspath(venv_python), "-m", "ensurepip", "--upgrade"],
            environment=environment,
            label="pip seed",
        )


def _ensure_venv(environment: dict[str, str]) -> Path:
    """Create or validate a Python 3.12 virtual environment."""

    if VENV_PYTHON.exists():
        version = _python_version(VENV_PYTHON)
        if version is None:
            raise RuntimeError(
                f"The existing virtual environment is not executable: {VENV_PYTHON}. "
                "Remove only .venv and rerun setup.bat."
            )
        if version[:2] != (3, 12):
            raise RuntimeError(
                f"{VENV_PYTHON} uses Python {version[0]}.{version[1]}, but this project "
                "requires Python 3.12. Remove only .venv and rerun setup.bat with Python 3.12."
            )
        _ensure_pip(VENV_PYTHON, environment)
        print(f"[venv] using existing isolated environment: {VENV_PYTHON}")
        return VENV_PYTHON

    uv = shutil.which("uv")
    if uv:
        _run(
            [uv, "venv", os.fspath(VENV), "--python", sys.executable, "--seed"],
            environment=environment,
            label="venv",
        )
    else:
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError(
                "uv is unavailable and bootstrap is running outside Python 3.12. "
                "Install Python 3.12 or install uv, then rerun setup.bat."
            )
        _run(
            [sys.executable, "-m", "venv", os.fspath(VENV)],
            environment=environment,
            label="venv",
        )

    if not VENV_PYTHON.exists():
        raise RuntimeError(f"Virtual environment creation did not produce {VENV_PYTHON}.")
    version = _python_version(VENV_PYTHON)
    if version is None or version[:2] != (3, 12):
        raise RuntimeError(f"New environment has an unsupported Python version: {version}.")
    _ensure_pip(VENV_PYTHON, environment)
    return VENV_PYTHON


def _install_dependencies(venv_python: Path, environment: dict[str, str]) -> None:
    if not REQUIREMENTS.is_file():
        raise RuntimeError(f"Dependency manifest is missing: {REQUIREMENTS}")
    manifest = LOCK_REQUIREMENTS if LOCK_REQUIREMENTS.is_file() else REQUIREMENTS
    manifest_label = "locked dependencies" if manifest == LOCK_REQUIREMENTS else "dependencies"

    uv = shutil.which("uv")
    if uv:
        command = [
            uv,
            "pip",
            "install",
            "--python",
            os.fspath(venv_python),
            "--requirement",
            os.fspath(manifest),
            "--index-strategy",
            "unsafe-best-match",
        ]
    else:
        command = [
            os.fspath(venv_python),
            "-m",
            "pip",
            "install",
            "--requirement",
            os.fspath(manifest),
        ]
    _run(command, environment=environment, label=manifest_label)

    if uv:
        check_command = [uv, "pip", "check", "--python", os.fspath(venv_python)]
    else:
        check_command = [os.fspath(venv_python), "-m", "pip", "check"]
    _run(check_command, environment=environment, label="pip check")


def _prepare_models(venv_python: Path, environment: dict[str, str]) -> None:
    """Download/verify model assets through the application's model worker."""

    command = [os.fspath(venv_python), "-m", "anime_picker.models", "prepare"]
    print(
        "[models] preparing assets under "
        f"{environment['ANIME_PICKER_MODEL_ROOT']} (this can take a while)"
    )
    _run(command, environment=environment, label="models")


def _prepare_logo_model(venv_python: Path, environment: dict[str, str]) -> None:
    """Prepare the optional export-time Anime-LaMa model explicitly.

    Logo cleanup is optional at runtime, but the normal setup path prepares its
    pinned local artifact so the default checked export setting is usable.  The
    command is deliberately separate from the main model worker: inference
    never downloads or mutates an existing model environment.
    """

    model_root = environment["ANIME_PICKER_LOGO_MODEL_ROOT"]
    command = [
        os.fspath(venv_python),
        "-m",
        "anime_picker.logo_removal",
        "prepare",
        "--model-root",
        model_root,
    ]
    print(f"[logo-models] preparing Anime-LaMa under {model_root}")
    _run(command, environment=environment, label="logo-models")


def _run_doctor(
    venv_python: Path,
    environment: dict[str, str],
    *,
    skip_models: bool,
) -> None:
    command = [os.fspath(venv_python), os.fspath(ROOT / "scripts" / "doctor.py")]
    if skip_models:
        command.append("--skip-models")
    _run(command, environment=environment, label="doctor")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="install and verify the base environment without downloading model assets",
    )
    parser.add_argument(
        "--check",
        "--doctor",
        dest="check",
        action="store_true",
        help="only run doctor against the existing .venv; do not install or prepare models",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="prepare models and run doctor using the existing .venv",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    environment = _runtime_environment()

    try:
        if args.check or args.models_only:
            if not VENV_PYTHON.exists():
                raise RuntimeError(
                    f"{VENV_PYTHON} is missing. Run setup.bat first to create the isolated environment."
                )
            venv_python = VENV_PYTHON
        else:
            venv_python = _ensure_venv(environment)
            _install_dependencies(venv_python, environment)

        if args.models_only:
            if args.skip_models:
                _run_doctor(venv_python, environment, skip_models=True)
            else:
                _prepare_models(venv_python, environment)
                _prepare_logo_model(venv_python, environment)
                _run_doctor(venv_python, environment, skip_models=False)
        elif not args.check and not args.skip_models:
            _prepare_models(venv_python, environment)
            _prepare_logo_model(venv_python, environment)
            _run_doctor(venv_python, environment, skip_models=False)
        else:
            _run_doctor(venv_python, environment, skip_models=True)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[setup:error] {exc}", file=sys.stderr)
        print(
            "[setup:hint] Review the command above and the log at "
            f"{LOG_PATH}. For a base install without model downloads, rerun: "
            "setup.bat --skip-models. To prepare only the logo model later, "
            "run `python -m anime_picker.logo_removal prepare` in .venv.",
            file=sys.stderr,
        )
        return 1

    print(f"[setup:ok] isolated environment ready: {VENV_PYTHON}")
    print(
        "[setup:next] run.bat starts the local server; model assets are under "
        f"{environment['ANIME_PICKER_MODEL_ROOT']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
