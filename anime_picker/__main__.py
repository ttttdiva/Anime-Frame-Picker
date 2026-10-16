"""Command-line entry point for the local Anime Frame Picker service."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Sequence


def _doctor() -> int:
    checks: dict[str, object] = {}
    for module in ("fastapi", "uvicorn", "PIL", "cv2"):
        try:
            __import__(module)
            checks[module] = True
        except Exception as exc:  # pragma: no cover - environment dependent
            checks[module] = str(exc)
    try:
        from .server import _model_status

        checks["models"] = _model_status()
    except Exception as exc:
        checks["models"] = {"ready": False, "error": str(exc)}
    print(json.dumps(checks, ensure_ascii=False, indent=2, default=str))
    return 0


def _serve(args: argparse.Namespace) -> int:
    allowed_hosts = {"127.0.0.1", "localhost", "::1"}
    if args.host.casefold().strip("[]") not in allowed_hosts:
        print("anime_picker serve only permits a loopback --host (127.0.0.1, localhost, or ::1)", file=sys.stderr)
        return 2
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - handled by doctor in CI
        print(f"uvicorn is required to serve the local app: {exc}", file=sys.stderr)
        return 2

    # Import only after parsing so `--help` and `doctor` remain cheap and safe.
    from .server import app

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        # The browser is explicitly user-facing for an interactive `serve`
        # invocation.  The timer avoids racing uvicorn's socket startup.
        timer = threading.Timer(0.7, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m anime_picker", description="Local Anime Frame Picker")
    subparsers = parser.add_subparsers(dest="command")
    serve = subparsers.add_parser("serve", help="start the local browser UI and HTTP API")
    serve.add_argument("--host", default="127.0.0.1", help="loopback bind address")
    serve.add_argument("--port", default=7866, type=int, help="HTTP port (default: 7866)")
    serve.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    serve.add_argument("--log-level", default="info", choices=("critical", "error", "warning", "info", "debug", "trace"))
    subparsers.add_parser("doctor", help="check local dependencies and model status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    if args.command == "serve":
        return _serve(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
