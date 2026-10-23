"""Cross-process exclusion: real child processes, fake models only (no GPU)."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from anime_picker import exclusion
from anime_picker.exclusion import ExclusionBusy

REPO = Path(__file__).resolve().parents[1]

CHILD = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, sys.argv[4])
    from anime_picker import exclusion
    root, kind, key = sys.argv[1:4]
    assert exclusion.__file__.startswith(sys.argv[4]), exclusion.__file__
    if kind == "video":
        ctx = exclusion.video_lease(root, key)
    elif kind == "gpu":
        ctx = exclusion.gpu_gate(root, timeout=0)
    else:
        ctx = exclusion.publish_lock(root, key)  # (output_root, identity)
    if kind != "publish":  # publish_lock() returns an already-acquired lock
        ctx.__enter__()
    print("HELD", flush=True)
    time.sleep(60)
    """
)


def _other_checkout(base: Path) -> Path:
    """A second, independent copy of the module (as another checkout would have)."""

    package = base / "checkout_b" / "anime_picker"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    shutil.copyfile(REPO / "anime_picker" / "exclusion.py", package / "exclusion.py")
    return base / "checkout_b"


def _spawn(root: Path, kind: str, key: str = "k", checkout: Path = REPO) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-I", "-c", CHILD, str(root), kind, key, str(checkout)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "HELD"
    return proc


@pytest.fixture()
def children():
    started: list[subprocess.Popen] = []
    yield started
    for proc in started:
        proc.kill()
        proc.wait()


def test_same_video_rejected_across_processes_but_other_video_allowed(tmp_path, children):
    children.append(_spawn(tmp_path, "video", "fp-A"))
    assert exclusion.video_busy(tmp_path, "fp-A")
    with pytest.raises(ExclusionBusy):
        with exclusion.video_lease(tmp_path, "fp-A"):
            pass
    assert not exclusion.video_busy(tmp_path, "fp-B")
    with exclusion.video_lease(tmp_path, "fp-B"):
        pass


def test_lock_released_after_abnormal_exit(tmp_path, children):
    proc = _spawn(tmp_path, "video", "fp-A")
    children.append(proc)
    assert exclusion.video_busy(tmp_path, "fp-A")
    proc.kill()  # no cleanup code runs
    proc.wait()
    deadline = time.monotonic() + 5
    while exclusion.video_busy(tmp_path, "fp-A") and time.monotonic() < deadline:
        time.sleep(0.05)
    with exclusion.video_lease(tmp_path, "fp-A"):
        pass


def test_threads_of_one_process_share_a_video_lease(tmp_path):
    with exclusion.video_lease(tmp_path, "fp"):
        with exclusion.video_lease(tmp_path, "fp"):
            assert not exclusion.video_busy(tmp_path, "fp")
        # still held after the inner release
        assert exclusion.is_held(exclusion._video_path(tmp_path, "fp"))
    assert not exclusion.is_held(exclusion._video_path(tmp_path, "fp"))


def test_gpu_gate_is_held_only_while_a_model_call_runs(tmp_path, children):
    calls: list[str] = []

    class Fake:
        revision = "r"

        def tag(self, image):
            calls.append("tag")
            assert exclusion.is_held(exclusion.locks_directory() / "gpu.lock")
            return image

        def vocabulary(self):
            return {}

    guarded = exclusion.guard_model(Fake(), tmp_path)
    assert guarded.revision == "r"
    # CPU-side stage: no gate, so another process may take it meanwhile.
    guarded.vocabulary()
    assert not exclusion.gpu_busy(tmp_path)
    assert guarded.tag(1) == 1 and calls == ["tag"]
    assert not exclusion.gpu_busy(tmp_path)


def test_gpu_gate_excludes_other_process_and_times_out(tmp_path, children):
    children.append(_spawn(tmp_path, "gpu"))
    assert exclusion.gpu_busy(tmp_path)
    with pytest.raises(ExclusionBusy):
        with exclusion.gpu_gate(tmp_path, timeout=0.2):
            pass


def test_gpu_gate_waits_for_release(tmp_path):
    other = exclusion.FileLock(exclusion.locks_directory() / "gpu.lock")
    # Same-process lock keyed identically is "busy" for the gate too.
    assert other.acquire()
    threading.Timer(0.3, other.release).start()
    started = time.monotonic()
    with exclusion.gpu_gate(tmp_path, timeout=5):
        pass
    assert time.monotonic() - started >= 0.25


def test_publish_collision_serializes_and_other_videos_proceed(tmp_path, children):
    children.append(_spawn(tmp_path, "publish", "video-A"))
    with pytest.raises(ExclusionBusy):
        exclusion.publish_lock(tmp_path, "video-A", timeout=0.2)
    exclusion.publish_lock(tmp_path, "video-B", timeout=0.2).release()
    exclusion.publish_lock(tmp_path / "other-root", "video-A", timeout=0.2).release()


def test_publish_image_set_takes_the_publication_lock(tmp_path, monkeypatch):
    from PIL import Image
    from anime_picker import output_layout

    out, state = tmp_path / "outputs", tmp_path / "state"
    out.mkdir()
    png = tmp_path / "a.png"
    Image.new("RGB", (4, 4), "red").save(png)
    digest = output_layout._digest_file(png)
    identity = output_layout._source_identity("v.mkv", "fp")
    seen: list[bool] = []
    real = output_layout._publish_image_set_locked

    def spy(*args, **kwargs):
        try:
            exclusion.publish_lock(out, identity, timeout=0).release()
            seen.append(True)
        except ExclusionBusy:
            seen.append(False)  # the publisher holds it
        return real(*args, **kwargs)

    monkeypatch.setattr(output_layout, "_publish_image_set_locked", spy)
    output_layout.publish_image_set(out, "v.mkv", "fp", [{"source": png, "pts": 1, "sha256": digest}], state_root=state, run_id="r")
    assert seen == [False]
    # released afterwards
    exclusion.publish_lock(out, identity, timeout=0.2).release()


def test_concurrent_publishers_of_one_video_do_not_overlap(tmp_path, monkeypatch):
    from PIL import Image
    from anime_picker import output_layout

    out, state = tmp_path / "outputs", tmp_path / "state"
    out.mkdir()
    pngs = []
    for index, colour in enumerate(("red", "blue", "green")):
        png = tmp_path / f"{index}.png"
        Image.new("RGB", (4 + index, 4), colour).save(png)
        pngs.append(png)
    active, overlaps = [0], []
    real = output_layout._publish_image_set_locked

    def spy(*args, **kwargs):
        active[0] += 1
        overlaps.append(active[0])
        time.sleep(0.15)
        try:
            return real(*args, **kwargs)
        finally:
            active[0] -= 1

    monkeypatch.setattr(output_layout, "_publish_image_set_locked", spy)
    errors: list[Exception] = []

    def publish(png: Path, pts: int) -> None:
        try:
            output_layout.publish_image_set(
                out, "v.mkv", "fp", [{"source": png, "pts": pts, "sha256": output_layout._digest_file(png)}],
                state_root=state, run_id=f"r{pts}",
            )
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=publish, args=(png, index + 1)) for index, png in enumerate(pngs)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and max(overlaps) == 1


# --- legacy (lock-less) batch detection ------------------------------------

def _fake_batch(tmp_path: Path, *extra: str) -> subprocess.Popen:
    script = tmp_path / "scripts_fake" / "batch_source.py"
    script.parent.mkdir(exist_ok=True)
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-I", str(script), *extra])
    time.sleep(0.5)
    return proc


@pytest.mark.real_legacy_scan
def test_legacy_batch_is_detected_and_blocks_new_start(tmp_path, children):
    proc = _fake_batch(tmp_path)
    children.append(proc)
    found = exclusion.legacy_batch_processes(tmp_path)
    assert proc.pid in {item["pid"] for item in found}
    with pytest.raises(ExclusionBusy):
        exclusion.ensure_no_legacy_batch(tmp_path)


def test_legacy_detection_is_scoped_to_output_root_and_ignores_dry_run(tmp_path, children):
    other = tmp_path / "elsewhere"
    proc = _fake_batch(tmp_path, "--output-root", str(other))
    children.append(proc)
    dry = _fake_batch(tmp_path, "--dry-run")
    children.append(dry)
    mine = {item["pid"] for item in exclusion.legacy_batch_processes(tmp_path, tmp_path / "outputs")}
    assert proc.pid not in mine and dry.pid not in mine
    assert proc.pid in {item["pid"] for item in exclusion.legacy_batch_processes(tmp_path, other)}


def test_registered_batch_is_not_legacy(tmp_path):
    registration = exclusion.register_batch_process(tmp_path)
    try:
        fake = [
            type("P", (), {"info": {"pid": os.getpid(), "ppid": 1, "cmdline": [sys.executable, "scripts/batch_source.py"], "cwd": str(tmp_path)}})(),
            type("P", (), {"info": {"pid": 999001, "ppid": os.getpid(), "cmdline": [sys.executable, "scripts/batch_source.py"], "cwd": str(tmp_path)}})(),
            type("P", (), {"info": {"pid": 999002, "ppid": 4242, "cmdline": [sys.executable, "scripts/batch_source.py"], "cwd": str(tmp_path)}})(),
        ]
        found = exclusion.legacy_batch_processes(tmp_path, process_iter=lambda attrs: iter(fake))
        assert [item["pid"] for item in found] == [999002]
    finally:
        registration.release()


@pytest.mark.real_legacy_scan
def test_server_refuses_start_when_video_busy_or_legacy_running(tmp_path, children, monkeypatch):
    from fastapi import HTTPException
    from anime_picker import server

    monkeypatch.setattr(server, "PACKAGE_ROOT", tmp_path)
    import psutil

    visible: set[int] = set()
    real_iter = psutil.process_iter
    # Hide this machine's real batches: only the fakes made here are visible.
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: (p for p in real_iter(attrs) if p.pid in visible))
    video = type("V", (), {"fingerprint": "fp-A"})()
    children.append(_spawn(tmp_path, "video", "fp-A"))
    with pytest.raises(HTTPException) as caught:
        server._refuse_conflicting_start(video)
    assert caught.value.status_code == 409
    with pytest.raises(HTTPException):
        with server._video_exclusion(video):
            pass
    other = type("V", (), {"fingerprint": "fp-B"})()
    server._refuse_conflicting_start(other)
    with server._video_exclusion(other):
        pass
    legacy = _fake_batch(tmp_path)
    children.append(legacy)
    visible.add(legacy.pid)
    with pytest.raises(HTTPException):
        server._refuse_conflicting_start(other)


def test_batch_skips_a_video_held_by_another_process_but_runs_the_rest(tmp_path, children, monkeypatch):
    import json
    from types import SimpleNamespace

    from PIL import Image

    import anime_picker.video as video_module
    from tests.test_batch_source import Frame
    from scripts.archive_completed import _fingerprint
    from scripts.batch_source import run

    source = tmp_path / "source"
    source.mkdir()
    for stem in ("First", "Second"):
        (source / f"{stem}.mp4").write_bytes(f"fake video {stem}".encode())

    class Reader:
        def __init__(self, file, *, decode_threads=None):
            self.path = Path(file)
            self.info = SimpleNamespace(duration=10.0, time_base="1/1000", fps=24.0)
            self.frames = [Frame(Image.new("RGB", (96, 54), (120, 50, 30)), 7.001, 7001)]

        @property
        def fingerprint(self):
            return _fingerprint(self.path)

        def iter_frames(self, start=0.0, end=None, max_side=960):
            return iter(self.frames)

        def export_pngs(self, destinations):
            for pts, paths in destinations.items():
                for path in paths if isinstance(paths, list) else [paths]:
                    Image.new("RGB", (40, 30), (pts % 255, 0, 0)).save(path, format="PNG")
            return set()

    monkeypatch.setattr(video_module, "VideoReader", Reader)
    # A different process (e.g. the UI) is working on First.
    children.append(_spawn(REPO, "video", _fingerprint(source / "First.mp4"), _other_checkout(tmp_path)))
    state = tmp_path / ".local" / "batch-state"
    code = run([
        "--source", str(source), "--mode", "all", "--output-root", str(tmp_path / "outputs"),
        "--state-root", str(state), "--cache-root", str(tmp_path / "cache"), "--no-logo-removal",
    ])
    assert code == 1  # incomplete, not an error crash
    report = json.loads(next(state.glob("batch_run_*.json")).read_text(encoding="utf-8"))
    by_name = {Path(job["video"]).stem: job for job in report["jobs"]}
    assert by_name["First"]["status"] == "busy"
    assert by_name["Second"]["status"] == "completed"
    assert not (tmp_path / "outputs" / "First").exists()
    # The busy video stays in source/; only the completed one is archived.
    assert (source / "First.mp4").exists() and not (source / "Second.mp4").exists()
    assert (tmp_path / "completed" / "Second.mp4").exists()
    # The batch released its own lease and registration when it finished.
    assert not exclusion.video_busy(REPO, _fingerprint(tmp_path / "completed" / "Second.mp4"))
    assert not exclusion.legacy_batch_processes(REPO, process_iter=lambda attrs: iter(()))


def test_locks_are_shared_across_checkouts_and_state_roots(tmp_path, children, monkeypatch):
    """A second checkout, resolving the default location, meets the same locks."""

    monkeypatch.delenv("ANIME_PICKER_LOCK_DIR")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "localappdata"))
    checkout = _other_checkout(tmp_path)
    out = tmp_path / "outputs"
    children.append(_spawn(tmp_path, "video", "fp-A", checkout))
    children.append(_spawn(tmp_path, "gpu", "-", checkout))
    children.append(_spawn(out, "publish", "ident", checkout))
    assert exclusion.locks_directory() == (tmp_path / "localappdata" / "anime-picker" / "locks").resolve()
    assert exclusion.video_busy(REPO, "fp-A") and not exclusion.video_busy(REPO, "fp-B")
    assert exclusion.gpu_busy(REPO)
    with pytest.raises(ExclusionBusy):
        exclusion.publish_lock(out, "ident", timeout=0.2)
    # The same output root spelled differently still hits the same lock.
    with pytest.raises(ExclusionBusy):
        exclusion.publish_lock(out / ".." / "outputs", "ident", timeout=0.2)


def test_publish_with_different_state_roots_serializes_on_one_output_root(tmp_path, monkeypatch):
    from PIL import Image
    from anime_picker import output_layout

    out = tmp_path / "outputs"
    out.mkdir()
    pngs = []
    for index, colour in enumerate(("red", "blue")):
        png = tmp_path / f"{index}.png"
        Image.new("RGB", (4 + index, 4), colour).save(png)
        pngs.append(png)
    active, peak, errors = [0], [0], []
    real = output_layout._publish_image_set_locked

    def spy(*args, **kwargs):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        try:
            return real(*args, **kwargs)
        finally:
            active[0] -= 1

    monkeypatch.setattr(output_layout, "_publish_image_set_locked", spy)

    def publish(index: int) -> None:
        try:
            output_layout.publish_image_set(
                out, "v.mkv", "fp", [{"source": pngs[index], "pts": index + 1, "sha256": output_layout._digest_file(pngs[index])}],
                state_root=tmp_path / f"state-{index}", run_id=f"r{index}",
            )
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=publish, args=(index,)) for index in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and peak[0] == 1


def test_callers_resolve_the_shared_lock_directory(tmp_path, monkeypatch):
    """Server and batch take real locks in the shared directory, not fakes."""

    from anime_picker import server

    video = type("V", (), {"fingerprint": "fp-real"})()
    seen = []
    with server._video_exclusion(video):
        seen.append(exclusion.is_held(exclusion._video_path(None, "fp-real")))
    assert seen == [True] and not exclusion.video_busy(None, "fp-real")
    assert list(exclusion.locks_directory().glob("video-*.lock"))


def test_batch_releases_the_video_lease_when_a_stage_raises_unexpectedly(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from PIL import Image

    import anime_picker.video as video_module
    import scripts.batch_source as batch_source
    from scripts.archive_completed import _fingerprint
    from tests.test_batch_source import Frame

    source = tmp_path / "source"
    source.mkdir()
    (source / "One.mp4").write_bytes(b"fake video one")

    class Reader:
        def __init__(self, file, *, decode_threads=None):
            self.path = Path(file)
            self.info = SimpleNamespace(duration=10.0, time_base="1/1000", fps=24.0)

        @property
        def fingerprint(self):
            return _fingerprint(self.path)

        def iter_frames(self, start=0.0, end=None, max_side=960):
            return iter([Frame(Image.new("RGB", (96, 54), (120, 50, 30)), 7.001, 7001)])

        def export_pngs(self, destinations):
            for pts, paths in destinations.items():
                for path in paths if isinstance(paths, list) else [paths]:
                    Image.new("RGB", (40, 30), (pts % 255, 0, 0)).save(path, format="PNG")
            return set()

    def explode(*_args, **_kwargs):
        raise RuntimeError("not one of the exceptions the publish step expects")

    monkeypatch.setattr(video_module, "VideoReader", Reader)
    monkeypatch.setattr(batch_source, "_publish_batch_images", explode)
    with pytest.raises(RuntimeError):
        batch_source.run([
            "--source", str(source), "--mode", "all", "--output-root", str(tmp_path / "outputs"),
            "--state-root", str(tmp_path / ".local" / "s"), "--cache-root", str(tmp_path / "cache"),
            "--no-logo-removal", "--no-archive",
        ])
    fingerprint = _fingerprint(source / "One.mp4")
    assert exclusion._LEASES == {}
    assert not exclusion.is_held(exclusion._video_path(None, fingerprint))
    assert not list(exclusion.locks_directory().glob("batch-*.lock")) or not any(
        exclusion.is_held(path) for path in exclusion.locks_directory().glob("batch-*.lock")
    )


def test_batched_tagging_runs_inside_the_gpu_gate(tmp_path):
    """``tag_many`` (cute mode, automatic UI runs) is a GPU call like ``tag``."""

    class Model:
        def tag_many(self, images, batch_size=8):
            return [exclusion._GPU_STATE["depth"] for _ in images]

        def vocabulary(self):
            return exclusion._GPU_STATE["depth"]

    guarded = exclusion.guard_model(Model(), tmp_path)
    assert guarded.tag_many([1, 2]) == [1, 1]
    assert guarded.vocabulary() == 0  # CPU-only metadata stays outside the gate


def test_a_momentary_probe_does_not_make_a_free_video_busy(tmp_path):
    probe = exclusion.FileLock(exclusion._video_path(tmp_path, "fp-probe"))
    assert probe.acquire()
    timer = threading.Timer(0.2, probe.release)
    timer.start()
    try:
        with exclusion.video_lease(tmp_path, "fp-probe"):
            pass
    finally:
        timer.join()


def test_publication_state_is_shared_by_every_checkout_writing_one_output_root(tmp_path):
    from anime_picker import output_layout

    out = tmp_path / "checkout_a" / "outputs"
    expected = output_layout.publication_state_root(out)
    assert expected == output_layout.publication_state_root(out / ".." / "outputs")
    # The checkout owning the outputs and any other checkout resolve the same place.
    for project in (tmp_path / "checkout_a", tmp_path / "checkout_b"):
        assert output_layout.publication_state_root(out) == expected
        assert output_layout.default_state_root(out, project) in {expected, project / ".local" / "anime-picker"}


def test_batch_publishes_with_the_shared_publication_state(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from PIL import Image

    import anime_picker.video as video_module
    import scripts.batch_source as batch_source
    from anime_picker import output_layout
    from scripts.archive_completed import _fingerprint
    from tests.test_batch_source import Frame

    source = tmp_path / "source"
    source.mkdir()
    (source / "One.mp4").write_bytes(b"fake video one")

    class Reader:
        def __init__(self, file, *, decode_threads=None):
            self.path = Path(file)
            self.info = SimpleNamespace(duration=10.0, time_base="1/1000", fps=24.0)

        @property
        def fingerprint(self):
            return _fingerprint(self.path)

        def iter_frames(self, start=0.0, end=None, max_side=960):
            return iter([Frame(Image.new("RGB", (96, 54), (120, 50, 30)), 7.001, 7001)])

        def export_pngs(self, destinations):
            for pts, paths in destinations.items():
                for path in paths if isinstance(paths, list) else [paths]:
                    Image.new("RGB", (40, 30), (pts % 255, 0, 0)).save(path, format="PNG")
            return set()

    seen = []
    real = batch_source.publish_image_set

    def spy(*args, **kwargs):
        seen.append(Path(kwargs["state_root"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(video_module, "VideoReader", Reader)
    monkeypatch.setattr(batch_source, "publish_image_set", spy)
    out = tmp_path / "outputs"
    assert batch_source.run([
        "--source", str(source), "--mode", "all", "--output-root", str(out),
        "--state-root", str(tmp_path / ".local" / "s"), "--cache-root", str(tmp_path / "cache"),
        "--no-logo-removal", "--no-archive",
    ]) == 0
    assert seen == [output_layout.publication_state_root(out)]


def test_model_free_batch_checks_legacy_batches_on_its_output_root(tmp_path, monkeypatch):
    import scripts.batch_source as batch_source

    source = tmp_path / "source"
    source.mkdir()
    (source / "One.mp4").write_bytes(b"fake")
    calls = []
    monkeypatch.setattr(exclusion, "ensure_no_legacy_batch", lambda root, output_root=None, **_: calls.append(output_root))
    monkeypatch.setattr(batch_source, "_run_locked", lambda *args, **kwargs: 0)
    out = tmp_path / "outputs"
    assert batch_source.run(["--source", str(source), "--mode", "all", "--output-root", str(out)]) == 0
    assert calls == [out.resolve()]


def test_ui_export_with_logo_removal_counts_as_gpu_work(tmp_path, monkeypatch):
    from anime_picker import server

    calls = []

    @contextlib.contextmanager
    def capture(video, *, output_root=None, gpu=True):
        calls.append(gpu)
        yield

    monkeypatch.setattr(server, "_video_exclusion", capture)
    monkeypatch.setattr(server, "_get_video", lambda _id: object())
    monkeypatch.setattr(server, "_export_job_exclusive", lambda *args, **kwargs: {})
    for logo in (True, False):
        job = type("J", (), {"video_id": "v", "settings": {"output_root": str(tmp_path / "out"), "logo_removal": logo}})()
        server._export_job(job, ["c"])
    assert calls == [True, False]
