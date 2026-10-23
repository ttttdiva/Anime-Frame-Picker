"""Start the real app on a free loopback port with temp data and CPU fakes.

Used by the headless real-UI acceptance check (tests/ui/char_select_ui.mjs).
Only the GPU model service is replaced (deterministic CPU detector/CCIP
fakes); the FastAPI app, static UI, real PyAV VideoReader, reference capture,
reference pipeline, run manifests, and exporter are the production code.
Every data directory lives under ``--root``; nothing touches the repository's
``outputs/``, ``.cache/``, source videos, or a running batch.

    python tests/ui/fixture_server.py --root <temp dir> --port 0
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_fixture_video(path: Path) -> None:
    import av
    import numpy as np

    from char_select_fakes import BLUE, RED, figure_frame

    red_alt = (200, 60, 50)  # same character, other outfit (within detector tolerance)
    scenes = [
        [(RED, 20, 20, 60, 100)],
        [(BLUE, 150, 20, 60, 100)],
        [(RED, 20, 20, 60, 100), (BLUE, 150, 20, 60, 100)],
        [(red_alt, 90, 30, 55, 95)],
        [],
        [(BLUE, 60, 25, 70, 105)],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 256, 144, "yuv420p"
        stream.options = {"crf": "10", "g": "1"}
        index = 0
        for scene_index, figures in enumerate(scenes):
            image = figure_frame(figures, seed=scene_index)
            for _repeat in range(10):
                frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, 10)
                index += 1
                for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    port = args.port or _free_port()
    for name in ("cache", "outputs", "source", "models", "logo-profiles"):
        (root / name).mkdir(parents=True, exist_ok=True)
    video = root / "source" / "Fixture Episode 01.mp4"
    if not video.is_file():
        _write_fixture_video(video)
    os.environ.update({
        "ANIME_PICKER_CACHE": str(root / "cache"),
        "ANIME_PICKER_OUTPUT_ROOT": str(root / "outputs"),
        "ANIME_PICKER_VIDEO_ROOT": str(root / "source"),
        "ANIME_PICKER_MODEL_ROOT": str(root / "models"),
        "ANIME_PICKER_LOGO_PROFILE_ROOT": str(root / "logo-profiles"),
    })
    os.chdir(root)

    import uvicorn

    from anime_picker import server
    from char_select_fakes import BLUE, RED, FakeModels

    class UiFakeModels(FakeModels):
        colours = (RED, BLUE)
        tolerance = 45

    server.ModelService = UiFakeModels
    server._MODEL_INSTANCE = None
    (root / "server.json").write_text(json.dumps({"port": port, "pid": os.getpid(), "video": str(video)}), encoding="utf-8")
    print(f"[fixture-server] http://127.0.0.1:{port}", flush=True)
    uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
