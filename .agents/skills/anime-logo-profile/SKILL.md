---
name: anime-logo-profile
description: Create, calibrate, preview, and verify local fixed-logo profiles for Anime Frame Picker videos while preserving native PTS frames and every pixel outside the declared edit mask.
---

# Anime logo profiles

Use this repository skill when the user asks to remove a fixed video logo,
adapt the workflow to a new logo shape or location, or verify logo removal.
The workflow is local and profile based. It does not upload frames, call cloud
inference, import `D:\Dev\09_test`, or claim to detect an unseen logo without
a registered template/profile.

The command reference is [references/commands.md](references/commands.md), and
the repository schema/evidence guide is
[docs/logo-profiles.md](../../../docs/logo-profiles.md).

## Start from the repository

1. Discover the project directory from the current working directory and use
   its `.venv\Scripts\python.exe` on Windows when it exists. Run noninteractive
   helpers with hidden/no-console process settings; never open a visible
   Windows Terminal for a test, build, or server helper.
2. Read `docs/logo-profiles.md`, inspect the local source videos, and inspect
   existing profiles with:

   ```powershell
   .venv\Scripts\python.exe scripts\logo_profiles.py list
   ```

3. Decode an **unprocessed** native source frame with `VideoReader`. Use the
   exact PTS for a video sample (`get_frame(pts, max_side=None)`), or a local
   native PNG. Keep the original video and source PNGs; never use a processed
   preview/export as a profile input.

## Create or adapt a profile

Measure the model ROI/context region and logo edit mask separately. The ROI is
the surrounding region used for matching; the mask is the shape that may be
changed by inpainting. Rectangles and arbitrary polygons use full-frame
reference pixels. Multiple rectangles/polygons may be declared for one logo.

Create a profile from a native PNG:

```powershell
.venv\Scripts\python.exe scripts\logo_profiles.py create `
  --id hi-anime --name "hi!anime.at" --image .cache\qa\frame-20s.png `
  --roi 1440 0 1920 256 --mask-rect 1718 18 1910 72
```

Use `--mask-file` for an ROI-local or full-frame PNG mask, `--polygon` for
`x,y;x,y;...` or JSON points, and `--context-roi` to record a wider context
region without expanding the edit mask. `--threshold` and the detector
weights are profile settings. Assets are copied to
`.local/logo-profiles/<id>/` (or `ANIME_PICKER_LOGO_PROFILE_ROOT`) with
`profile.json`, `reference.png`, and `template.png`; this store is local and
ignored by Git. An existing profile is never replaced without explicit
`--overwrite`.

If the logo position, size, or shape changes, create a new profile or update
the mask/template from a new unprocessed native frame. Do not assume one
profile covers another logo. For multiple simultaneous logos, make one profile
per independently matched logo and verify their masks do not overlap
unexpectedly.

## Calibrate and preview

Calibration compares unprocessed positive and negative local frames through
the core detector on CPU. It does not load Anime-LaMa or reserve the GPU. Do
not label a frame as negative from its timestamp alone: inspect it first. In
the supplied Episode 1 samples, `hi!anime.at` is present at all checked
timestamps, so use those as positive samples. For a MUSE calibration, the
known sampled positive is 110 s and the checked negatives are 20 s and 70 s:

```powershell
.venv\Scripts\python.exe scripts\logo_profiles.py calibrate `
  --profile muse --video "source\The World's Strongest Witch Episode 1.mp4" `
  --positive-time 110 --negative-time 20 --negative-time 70
```

Use `--positive-pts`/`--negative-pts` for exact PTS samples or image paths with
`--positive`/`--negative`. Review the recommendation and pass both
`--write --overwrite` to update `profile.json`. A threshold is calibration
evidence for the supplied samples; it is not a universal detector guarantee.
For `hi-anime`, calibrate with the observed positive timestamps (20, 60, 70,
80, 100, 110 in the supplied episode) and obtain a negative only from a
separately inspected logo-absent image or an explicitly synthetic empty ROI.

Preview writes native before/after/mask PNGs and compact JSON metadata. The
video path is decoded at an exact PTS when `--pts` is supplied, or sampled at
the requested time with `VideoReader.sample`:

```powershell
.venv\Scripts\python.exe scripts\logo_profiles.py preview `
  --profile hi-anime --video source\episode.mp4 --time 20 `
  --out .cache\logo-preview\hi-anime-20s
```

The default preview requires the verified local Anime-LaMa model and the
runtime's normal GPU policy. `--dry` is a detector-only CPU preview; `--cpu`
is an explicit real CPU inpainting choice. Model preparation is a separate
operator action and is never triggered by runtime, calibration, or preview.

## Verify before integration

Run CPU-only checks across logo-present and logo-absent frames, different
shapes/backgrounds, several timestamps, and more than one resolution when
available. For the supplied MUSE profile:

```powershell
.venv\Scripts\python.exe scripts\logo_profiles.py verify `
  --profile muse --video "source\The World's Strongest Witch Episode 1.mp4" `
  --present-time 110 --absent-time 20 --absent-time 70 `
  --out .cache\logo-verify\muse --overwrite
```

The report must show that present samples match and absent samples skip the
profile, native dimensions remain unchanged, and every pixel outside the
returned edit mask is byte/pixel exact. CPU `verify` uses a deterministic
dummy inpainting seam: it validates detector hits, mask geometry, native-size
handling, and the outside-mask pixel contract, but it does not validate the
visual quality of Anime-LaMa cleanup. A profile is complete only after a real
GPU `preview` run, visual inspection of before/after/mask images with Codex
image view or Computer Use, and an outside-mask audit. Check that the mask
covers the logo and excludes nearby content, and that the original video and
input PNG remain unchanged.

Default UI behavior is logo removal enabled; expose an explicit user option
to disable it when integrating the profile processor. Keep that option
separate from profile calibration and preserve the unmodified source.

## Scope and claims

The profile detector supports user-chosen locations, sizes, multiple logo
shapes, bright/dark backgrounds, and threshold calibration. It is a local
template/profile matcher, not a universal unseen-logo detector. Use the core
`ProfileStore`/`LogoProcessor` API through `scripts/logo_profiles.py`; do not
duplicate the matching or Anime-LaMa engine in a new script.
