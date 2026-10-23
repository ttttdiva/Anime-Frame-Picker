from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "real_legacy_scan: keep the real process-list scan for lock-less batches")


@pytest.fixture(autouse=True)
def _isolated_lock_dir(tmp_path_factory, monkeypatch):
    """Machine-wide locks live in a per-test directory (children inherit it)."""

    monkeypatch.setenv("ANIME_PICKER_LOCK_DIR", str(tmp_path_factory.mktemp("locks")))
    yield


@pytest.fixture(autouse=True)
def _no_foreign_batch_scan(request, monkeypatch):
    """Tests must not depend on real batches running on this machine."""

    if request.node.get_closest_marker("real_legacy_scan") is None:
        from anime_picker import exclusion

        monkeypatch.setattr(exclusion, "ensure_no_legacy_batch", lambda *args, **kwargs: None)
    yield


@pytest.fixture(autouse=True)
def _isolated_run_manifests(tmp_path, monkeypatch):
    """Keep UI run manifests and candidate indexes per test (never shared)."""

    from anime_picker import server

    monkeypatch.setattr(server, "RUN_ROOT", tmp_path / "_runs")
    monkeypatch.setattr(server, "SUGGESTION_INDEX", {})
    # Run ids are content hashes, so a queued job left by another test must
    # never be "attached" to this test's identical selection.
    monkeypatch.setattr(server, "JOB_REGISTRY", {})
    monkeypatch.setattr(server, "RUN_OWNERS", {})
    monkeypatch.setattr(server, "RUN_LOCKS", {})
    yield
