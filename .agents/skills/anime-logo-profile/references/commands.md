# Local command reference

`scripts/logo_profiles.py` owns profile orchestration and uses
`anime_picker.logo_removal` for schema validation, matching, masks, and
inpainting.

The store root is `.local/logo-profiles` by default. Set
`ANIME_PICKER_LOGO_PROFILE_ROOT` or pass `--profile-root`/`--root` to point at
another local store. Profile IDs are safe path components only:

```text
^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$
```

The profile schema is version 1. `reference_size`, `roi`, and mask geometry are
full-frame reference pixels. `template_image` is normally an ROI-local PNG.
`custom_mask` may be ROI-local or full-frame. The detector settings are
`threshold`, `aspect_tolerance`, `edge_weight`, and `detail_weight`.

The commands are:

* `list`: load and print validated profiles without absolute asset paths.
* `create`: copy an unprocessed source PNG, derive/copy a local template and
  optional mask, and write `profile.json`.
* `calibrate`: score native image/video samples on CPU and report a proposed
  threshold; writing requires `--write --overwrite`.
* `preview`: decode an image or exact-PTS/time video frame and write
  `before.png`, `after.png`, `mask.png`, and `metadata.json`.
* `verify`: run detector-only present/absent and pixel-boundary checks on CPU.

`preview --dry`, `calibrate`, and `verify` never load the inpainting model.
Runtime model preparation is a separate explicit operation; no command should
download a model as a side effect of processing a frame.

`verify` replaces the inpainting boundary with deterministic dummy output. Its
pass means that detector expectations, native dimensions, mask geometry, and
pixel equality outside the mask hold. It does not establish Anime-LaMa visual
cleanup quality. Complete a profile only after a real GPU `preview`, visual
before/after/mask inspection, and an outside-mask audit. Never infer an absent
logo from a timestamp without inspecting that source frame first.
