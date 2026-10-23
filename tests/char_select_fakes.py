"""CPU-only deterministic fakes for the character-selection flow.

Characters are drawn as coloured figures.  The fake detector returns their
boxes, ``embed`` returns the mean colour of a crop, and ``distance`` is the
Euclidean distance, so "same character" means "same colour" no matter the
pose, size, or background.  No real model, GPU, or anime frame is involved.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

RED = (220, 40, 40)
RED_ALT_OUTFIT = (205, 55, 45)   # same character, different outfit/expression
BLUE = (40, 60, 220)
BACKGROUND = (128, 128, 128)
SIZE = (256, 144)


def figure_frame(figures, *, background=BACKGROUND, seed: int = 0) -> Image.Image:
    """``figures`` is a list of (colour, x0, y0, w, h)."""

    image = Image.new("RGB", SIZE, background)
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(seed)
    for x in range(0, SIZE[0], 16):
        shade = int(100 + rng.integers(0, 40))
        draw.rectangle((x, 110, x + 15, 143), fill=(shade, shade, shade))
    for colour, x0, y0, w, h in figures:
        draw.rectangle((x0, y0, x0 + w, y0 + h), fill=colour, outline=(10, 10, 10), width=2)
        draw.ellipse((x0 + w // 4, y0 + 6, x0 + 3 * w // 4, y0 + h // 3), fill=(250, 220, 200), outline=(20, 20, 20))
        draw.ellipse((x0 + w // 3, y0 + 12, x0 + w // 3 + 3, y0 + 15), fill=(0, 0, 0))
        draw.ellipse((x0 + 2 * w // 3 - 3, y0 + 12, x0 + 2 * w // 3, y0 + 15), fill=(0, 0, 0))
    return image


class FakeModels:
    """detect/embed/distance/threshold contract of ModelService, on CPU."""

    require_gpu = True
    identity_revision = "fake-ccip-v1"
    revision = "fake-ccip-v1"
    identity_model = "fake-ccip"
    threshold = 0.2

    colours = (RED, RED_ALT_OUTFIT, BLUE)
    tolerance = 6

    def __init__(self, *, fail_detect: bool = False, **_kwargs):
        self.fail_detect = fail_detect
        self.detect_calls = 0
        self.embed_calls = 0

    @staticmethod
    def status(_root=None):
        return {"ready": True, "details": "cpu fake"}

    def vocabulary(self):
        return {"smile": "general"}

    def detect(self, image):
        self.detect_calls += 1
        if self.fail_detect:
            raise RuntimeError("fixture detector failure")
        array = np.asarray(image.convert("RGB")).astype(np.int16)
        detections = []
        for colour in self.colours:
            mask = np.all(np.abs(array - np.array(colour)) <= self.tolerance, axis=2)
            if mask.sum() < 200:
                continue
            ys, xs = np.nonzero(mask)
            box = [int(xs.min()) - 2, int(ys.min()) - 2, int(xs.max()) + 3, int(ys.max()) + 3]
            box = [max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3])]
            detections.append({"box": box, "score": 0.9, "kind": "person"})
        return detections

    def embed(self, images):
        self.embed_calls += 1
        rows = []
        for image in images:
            array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
            pixels = array.reshape(-1, 3)
            saturated = pixels[(pixels.max(axis=1) - pixels.min(axis=1)) > 0.35]
            colour = saturated.mean(axis=0) if len(saturated) else pixels.mean(axis=0)
            rows.append(colour)
        return np.asarray(rows, dtype=np.float32)

    def distance(self, left, right):
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        return np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)


class FrameListReader:
    """VideoReader double: exact PTS frames at 1 fps."""

    frames: list[Image.Image] = []

    def __init__(self, _path=None, **_kwargs):
        self.info = SimpleNamespace(duration=float(len(self.frames)), fps=1.0, width=SIZE[0], height=SIZE[1])
        self.fingerprint = "fake-fingerprint"

    def sample(self, start, end, step, max_side=960):
        index = int(round(start))
        t = float(start)
        while t <= min(float(end), len(self.frames) - 1) + 1e-9:
            index = int(round(t))
            if 0 <= index < len(self.frames):
                yield SimpleNamespace(pts=1000 + index, time=float(index), stream_index=0, image=self.frames[index])
            t += max(1.0, float(step))


def _get_frame(self, pts, max_side=None):
    index = int(pts) - 1000
    if not 0 <= index < len(self.frames):
        raise KeyError(pts)
    return SimpleNamespace(pts=int(pts), time=float(index), stream_index=0, image=self.frames[index])


FrameListReader.get_frame = _get_frame
