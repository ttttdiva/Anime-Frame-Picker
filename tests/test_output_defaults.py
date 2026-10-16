from pathlib import Path

import pytest

from anime_picker import server
from scripts import bootstrap, doctor


def test_default_output_roots_are_repo_relative_and_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    expected = server.PACKAGE_ROOT / "outputs"

    assert server.DEFAULT_OUTPUT_ROOT == expected.resolve()
    assert bootstrap.DEFAULT_OUTPUT_ROOT == (bootstrap.ROOT / "outputs").resolve()
    assert doctor.DEFAULT_OUTPUT_ROOT == (doctor.ROOT / "outputs").resolve()
    assert server._resolve_output_root("outputs") == expected.resolve()


def test_explicit_output_root_override_is_preserved(tmp_path):
    external = tmp_path / "frame-exports"

    assert server._resolve_output_root(external) == external.resolve()


def test_blank_output_root_environment_uses_repo_default(monkeypatch, tmp_path):
    bootstrap_default = tmp_path / "bootstrap-default"
    monkeypatch.setattr(bootstrap, "DEFAULT_OUTPUT_ROOT", bootstrap_default)
    monkeypatch.setenv("ANIME_PICKER_OUTPUT_ROOT", "  ")
    monkeypatch.setenv("ANIME_PICKER_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("ANIME_PICKER_LOGO_MODEL_ROOT", str(tmp_path / "logo-model"))
    monkeypatch.setenv("ANIME_PICKER_LOGO_PROFILE_ROOT", str(tmp_path / "profiles"))

    environment = bootstrap._runtime_environment()

    assert Path(environment["ANIME_PICKER_OUTPUT_ROOT"]) == bootstrap_default
    assert doctor._path_from_environment("ANIME_PICKER_OUTPUT_ROOT", doctor.DEFAULT_OUTPUT_ROOT) == doctor.DEFAULT_OUTPUT_ROOT
    assert server._resolve_output_root("  ") == server.DEFAULT_OUTPUT_ROOT


@pytest.mark.parametrize("invalid", ["C:frame-exports", "C:", r"\frame-exports"])
def test_drive_relative_output_roots_are_rejected(monkeypatch, invalid):
    monkeypatch.setenv("ANIME_PICKER_OUTPUT_ROOT", invalid)

    with pytest.raises(ValueError):
        server._resolve_output_root(invalid)
    with pytest.raises(ValueError):
        bootstrap._resolve_output_root(invalid)
    with pytest.raises(ValueError):
        doctor._path_from_environment("ANIME_PICKER_OUTPUT_ROOT", doctor.DEFAULT_OUTPUT_ROOT)


def test_absolute_windows_output_override_is_preserved():
    absolute = Path(r"C:\frame-exports")

    assert server._resolve_output_root(absolute) == absolute.resolve(strict=False)


def test_bootstrap_runtime_environment_uses_explicit_output_root(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    output_root = tmp_path / "exports"
    logo_model_root = tmp_path / "logo-model"
    logo_profile_root = tmp_path / "profiles"
    monkeypatch.setenv("ANIME_PICKER_MODEL_ROOT", str(model_root))
    monkeypatch.setenv("ANIME_PICKER_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("ANIME_PICKER_LOGO_MODEL_ROOT", str(logo_model_root))
    monkeypatch.setenv("ANIME_PICKER_LOGO_PROFILE_ROOT", str(logo_profile_root))

    environment = bootstrap._runtime_environment()

    assert Path(environment["ANIME_PICKER_OUTPUT_ROOT"]) == output_root
    assert output_root.is_dir()
    assert Path(environment["ANIME_PICKER_MODEL_ROOT"]) == model_root
