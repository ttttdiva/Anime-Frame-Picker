"""Anime Frame Picker package.

The HTTP application is available as :data:`anime_picker.app`; it is imported
lazily so model preparation and command-line tooling do not initialize FastAPI
or a worker pool unnecessarily.
"""

from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"app", "create_app"}:
        from .server import app, create_app

        return {"app": app, "create_app": create_app}[name]
    raise AttributeError(name)


__all__ = ["app", "create_app", "__version__"]
