"""Cross-process exclusion shared by the Web UI and ``scripts/batch_source.py``.

Three locks, each taken only for the stage that needs it:

* ``video`` lease: one video is processed by one process at a time (shared
  scan/identity caches and the public ``images`` directory). Different videos
  never wait on each other. Threads of one process share the lease.
* ``gpu`` gate: held only around a model call (and model load), never across
  CPU-only decoding, scanning or de-duplication. It serializes GPU compute
  only: weights a process has loaded stay resident in VRAM between calls, so
  the gate does not bound how much VRAM several processes hold together.
* ``publish`` lock: held while one video's public image set is switched.

Locks are OS byte-range locks on small files, so they vanish with the owning
process (including a crash or kill); nothing is cleaned up by timestamps.

Processes started before this module existed hold no lock. They are detected
read-only from the process list (``batch_source.py`` command lines) so that a
conflicting new start can be refused; they are never signalled or stopped.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterator

if os.name == "nt":
    import msvcrt
else:  # pragma: no cover - exercised on POSIX only
    import fcntl

# ``tag_many`` is the batched PixAI path (cute mode and automatic UI runs);
# ``_prepare_runtime`` touches CUDA directly when cute mode checks the detectors.
GPU_METHODS = frozenset({"detect", "embed", "distance", "tag", "tag_many", "process", "_prepare_runtime"})
POLL_SECONDS = 0.05
# ``is_held`` probes take a lock for an instant; real holders keep it. A
# fail-fast acquire waits this long so a probe never reads as "busy".
PROBE_GRACE_SECONDS = 1.0


class ExclusionBusy(RuntimeError):
    """Another process (or thread) holds the requested exclusion."""


def locks_directory(project_root: str | os.PathLike[str] | None = None) -> Path:
    """Machine-wide lock directory, identical for every checkout and state root.

    The locked resources (a video's content, the GPU, a public output
    directory) do not belong to one checkout, so the locks must not either.
    ``ANIME_PICKER_LOCK_DIR`` overrides the per-user default. ``project_root``
    is accepted for call compatibility and deliberately not used.
    """

    override = os.environ.get("ANIME_PICKER_LOCK_DIR")
    if override:
        return Path(override).resolve(strict=False)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    else:  # pragma: no cover
        base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return Path(base).resolve(strict=False) / "anime-picker" / "locks"


def _name(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}.lock"


_HELD: set[str] = set()
_HELD_MUTEX = threading.Lock()


def _try_os_lock(fd: int) -> bool:
    try:
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _os_unlock(fd: int) -> None:
    if os.name == "nt":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover
        fcntl.flock(fd, fcntl.LOCK_UN)


class FileLock:
    """Exclusive lock; a second acquire fails whoever (process or thread) holds it."""

    def __init__(self, path: str | os.PathLike[str], holder: str = "") -> None:
        self.path = Path(path)
        self.holder = holder
        self._fd: int | None = None
        self._key = os.path.normcase(os.fspath(self.path.resolve(strict=False)))

    def acquire(self, timeout: float = 0.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if self._try_once():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(POLL_SECONDS)

    def _try_once(self) -> bool:
        with _HELD_MUTEX:
            if self._key in _HELD:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
            if not _try_os_lock(fd):
                os.close(fd)
                return False
            _HELD.add(self._key)
            self._fd = fd
        # Diagnostics only; byte 0 is the lock, the description follows it.
        with contextlib.suppress(OSError):
            info = json.dumps({"pid": os.getpid(), "holder": self.holder, "since": time.time()}).encode("utf-8")
            os.lseek(fd, 1, os.SEEK_SET)
            os.write(fd, info)
        return True

    def release(self) -> None:
        with _HELD_MUTEX:
            fd, self._fd = self._fd, None
            if fd is None:
                return
            with contextlib.suppress(OSError):
                _os_unlock(fd)
            with contextlib.suppress(OSError):
                os.close(fd)
            _HELD.discard(self._key)

    def __enter__(self) -> "FileLock":
        if not self.acquire():
            raise ExclusionBusy(f"busy: {self.path.name}")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def describe(self) -> dict[str, Any]:
        """Best-effort description of the current holder (may be empty)."""

        try:
            with open(self.path, "rb") as handle:
                handle.seek(1)
                return json.loads(handle.read(512).decode("utf-8", "replace").strip("\x00") or "{}")
        except (OSError, ValueError):
            return {}


def is_held(path: str | os.PathLike[str]) -> bool:
    """True if some live process (or thread here) holds ``path``."""

    probe = FileLock(path)
    if probe.acquire():
        probe.release()
        return False
    return True


# --- video lease ---------------------------------------------------------

_LEASES: dict[str, list[Any]] = {}
_LEASE_MUTEX = threading.Lock()


def _video_path(project_root: Any, fingerprint: str) -> Path:
    return locks_directory(project_root) / _name("video", fingerprint)


def video_busy(project_root: Any, fingerprint: str) -> bool:
    """Whether a *different* process is working on this video."""

    path = _video_path(project_root, fingerprint)
    with _LEASE_MUTEX:
        if path in _LEASES:
            return False
    return is_held(path)


@contextlib.contextmanager
def video_lease(project_root: Any, fingerprint: str, holder: str = "") -> Iterator[None]:
    """Hold one video exclusively across processes; threads here share it."""

    path = _video_path(project_root, fingerprint)
    with _LEASE_MUTEX:
        entry = _LEASES.get(path)
        if entry is not None:
            entry[1] += 1
        else:
            lock = FileLock(path, holder or f"video:{os.getpid()}")
            if not lock.acquire(PROBE_GRACE_SECONDS):
                other = lock.describe()
                raise ExclusionBusy(
                    "this video is being processed by another process"
                    + (f" (pid {other['pid']})" if other.get("pid") else "")
                )
            _LEASES[path] = [lock, 1]
    try:
        yield
    finally:
        with _LEASE_MUTEX:
            entry = _LEASES[path]
            entry[1] -= 1
            if entry[1] == 0:
                del _LEASES[path]
                entry[0].release()


# --- GPU gate ------------------------------------------------------------

_GPU_THREAD_LOCK = threading.RLock()
_GPU_STATE = {"depth": 0, "lock": None}


def gpu_wait_seconds() -> float:
    try:
        return float(os.environ.get("ANIME_PICKER_GPU_WAIT_SECONDS", "3600"))
    except ValueError:
        return 3600.0


@contextlib.contextmanager
def gpu_gate(project_root: Any, timeout: float | None = None) -> Iterator[None]:
    """Exclusive GPU use for one model call; re-entrant within a thread."""

    wait = gpu_wait_seconds() if timeout is None else timeout
    deadline = time.monotonic() + wait
    if not _GPU_THREAD_LOCK.acquire(timeout=max(0.0, wait)):
        raise ExclusionBusy("timed out waiting for the GPU gate (another thread)")
    try:
        if _GPU_STATE["depth"] == 0:
            lock = FileLock(locks_directory(project_root) / "gpu.lock", f"gpu:{os.getpid()}")
            if not lock.acquire(max(0.0, deadline - time.monotonic())):
                raise ExclusionBusy("timed out waiting for the GPU gate (another process)")
            _GPU_STATE["lock"] = lock
        _GPU_STATE["depth"] += 1
        try:
            yield
        finally:
            _GPU_STATE["depth"] -= 1
            if _GPU_STATE["depth"] == 0:
                _GPU_STATE["lock"].release()
                _GPU_STATE["lock"] = None
    finally:
        _GPU_THREAD_LOCK.release()


def gpu_busy(project_root: Any) -> bool:
    if _GPU_STATE["depth"]:
        return False
    return is_held(locks_directory(project_root) / "gpu.lock")


class GuardedModel:
    """Model proxy whose GPU calls (only) run inside the GPU gate."""

    def __init__(self, model: Any, project_root: Any) -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_root", project_root)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._model, name)
        if name in GPU_METHODS and callable(value):
            root = self._root

            def guarded(*args: Any, **kwargs: Any) -> Any:
                with gpu_gate(root):
                    return value(*args, **kwargs)

            return guarded
        return value


def guard_model(model: Any, project_root: Any) -> Any:
    if model is None or isinstance(model, GuardedModel):
        return model
    return GuardedModel(model, project_root)


# --- publish lock --------------------------------------------------------

def publish_lock(output_root: str | os.PathLike[str], identity: str, timeout: float = 300.0) -> FileLock:
    """Acquire (blocking up to ``timeout``) the lock for one public video directory.

    Keyed by the resolved public output root and the video identity, never by
    a state root: writers with different state roots still share the images.
    """

    root = os.path.normcase(os.fspath(Path(output_root).resolve(strict=False)))
    lock = FileLock(locks_directory() / _name("publish", f"{root}|{identity}"), f"publish:{os.getpid()}")
    if not lock.acquire(timeout):
        raise ExclusionBusy(f"timed out waiting to publish {identity!r}; another publication is running")
    return lock


# --- legacy (lock-less) batch processes ----------------------------------

def register_batch_process(project_root: Any) -> FileLock:
    """Mark this process as a lock-aware batch for its lifetime."""

    lock = FileLock(locks_directory(project_root) / f"batch-{os.getpid()}.lock", f"batch:{os.getpid()}")
    if not lock.acquire(PROBE_GRACE_SECONDS):
        raise ExclusionBusy("batch registry entry is already held")
    return lock


def _is_batch_command(cmdline: list[str]) -> bool:
    for argument in cmdline[1:]:
        base = os.path.basename(argument.replace("\\", "/")).lower()
        if base in {"batch_source.py", "scripts.batch_source"}:
            return True
    return False


def _script_checkout(cmdline: list[str], base: Path) -> Path:
    for argument in cmdline[1:]:
        if os.path.basename(argument.replace("\\", "/")).lower() == "batch_source.py":
            return (base / argument).resolve(strict=False).parent.parent
    return base


def _arg_value(cmdline: list[str], flag: str) -> str | None:
    for index, argument in enumerate(cmdline):
        if argument == flag and index + 1 < len(cmdline):
            return cmdline[index + 1]
        if argument.startswith(flag + "="):
            return argument.split("=", 1)[1]
    return None


def legacy_batch_processes(
    project_root: Any,
    output_root: Any | None = None,
    *,
    process_iter: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Running ``batch_source.py`` processes that hold no lock (read-only scan).

    When ``output_root`` is given, only batches that write that public root
    qualify. Dry runs and processes related to a lock-aware batch (its venv
    launcher parent/children) or to this process are ignored.
    """

    if process_iter is None:
        try:
            import psutil
        except ImportError:
            return []
        process_iter = psutil.process_iter
    root = Path(project_root).resolve(strict=False)
    wanted_root = Path(output_root).resolve(strict=False) if output_root is not None else None
    registered: set[int] = set()
    registry = locks_directory(root)
    if registry.is_dir():
        for path in registry.glob("batch-*.lock"):
            try:
                pid = int(path.stem.split("-", 1)[1])
            except ValueError:
                continue
            if is_held(path):
                registered.add(pid)
    candidates: dict[int, dict[str, Any]] = {}
    for proc in process_iter(["pid", "ppid", "cmdline", "cwd"]):
        try:
            info = proc.info
            pid = int(info["pid"])
            candidates[pid] = {"pid": pid, "ppid": int(info.get("ppid") or 0), "cmdline": list(info.get("cmdline") or []), "cwd": info.get("cwd")}
        except Exception:
            continue
    # Ours: lock-aware batches and this process, their launcher parents
    # (a venv python.exe re-spawns the interpreter) and the descendants of
    # registered batches (not of this process: a server's children are not it).
    mine = registered | {os.getpid()}
    launchers = {item["ppid"] for pid, item in candidates.items() if pid in mine}
    descendants: set[int] = set()
    frontier = set(registered)
    while frontier:
        frontier = {pid for pid, item in candidates.items() if item["ppid"] in frontier and pid not in descendants and pid not in mine}
        descendants |= frontier
    related = mine | launchers | descendants
    found: list[dict[str, Any]] = []
    for pid, item in candidates.items():
        cmdline = item["cmdline"]
        if pid in related or not _is_batch_command(cmdline) or "--dry-run" in cmdline:
            continue
        if wanted_root is not None:
            value = _arg_value(cmdline, "--output-root")
            base = Path(item["cwd"]) if item.get("cwd") else root
            if value is None:
                # Default is <checkout>/outputs of the script that is running.
                theirs = _script_checkout(cmdline, base) / "outputs"
            else:
                theirs = (base / value).resolve(strict=False)
            if os.path.normcase(os.fspath(theirs)) != os.path.normcase(os.fspath(wanted_root)):
                continue
        found.append({"pid": pid, "cmdline": " ".join(cmdline)[:300]})
    return found


def ensure_no_legacy_batch(project_root: Any, output_root: Any | None = None, **kwargs: Any) -> None:
    found = legacy_batch_processes(project_root, output_root, **kwargs)
    if found:
        pids = ", ".join(str(item["pid"]) for item in found)
        raise ExclusionBusy(f"a batch started before cross-process locking is still running (pid {pids}); wait for it to finish")
