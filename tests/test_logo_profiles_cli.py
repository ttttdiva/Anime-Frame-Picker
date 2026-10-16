from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.logo_profiles import main


def test_create_rejects_source_inside_target_profile_before_overwrite(tmp_path: Path, capsys):
    """--overwrite must never mutate an input used as a profile destination."""

    profile_root = tmp_path / "profiles"
    profile_dir = profile_root / "demo"
    profile_dir.mkdir(parents=True)
    source = profile_dir / "template.png"
    Image.new("RGB", (64, 48), (23, 41, 59)).save(source, format="PNG")
    before = source.read_bytes()

    code = main(
        [
            "--root",
            str(profile_root),
            "create",
            "--id",
            "demo",
            "--name",
            "Demo",
            "--image",
            str(source),
            "--roi",
            "0",
            "0",
            "64",
            "48",
            "--mask-rect",
            "8",
            "8",
            "32",
            "24",
            "--overwrite",
        ]
    )

    assert code == 2
    assert source.read_bytes() == before
    assert not (profile_dir / "reference.png").exists()
    assert "target profile directory" in capsys.readouterr().err
