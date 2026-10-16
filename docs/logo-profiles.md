# Local logo profiles

Logo removal is a local, optional export-stage feature. A registered profile
describes a fixed logo with a reference image/template, a surrounding ROI for
matching, and one or more full-frame mask shapes. The mask defines the only
area that the inpainting step is allowed to edit. A frame with no profile hit
is returned unchanged and does not load Anime-LaMa.

The profile store defaults to `.local/logo-profiles` under the repository. Set
`ANIME_PICKER_LOGO_PROFILE_ROOT` or pass `--profile-root` to use another local
directory. Profile assets are ignored local files; the repository should not
contain raw source frames or logo crops. The original video and native input
PNG are retained outside the generated preview directory.

## Schema version 1

Each profile is `<store>/<id>/profile.json` with `template.png`,
`reference.png`, and optionally `mask.png` beside it. The core validator is
`anime_picker.logo_removal.ProfileStore`.

```json
{
  "version": 1,
  "id": "hi-anime",
  "name": "hi!anime.at",
  "reference_size": [1920, 1080],
  "roi": [1440, 0, 1920, 256],
  "mask_coordinates": "full_frame",
  "mask": {
    "rectangles": [[1718, 18, 1910, 72]],
    "polygons": []
  },
  "template_image": "template.png",
  "detector": {
    "threshold": 0.62,
    "aspect_tolerance": 0.025,
    "edge_weight": 0.65,
    "detail_weight": 0.35
  }
}
```

`reference_size`, `roi`, rectangles, and polygon points use full-frame pixel
coordinates. The template is usually an ROI-local PNG; a full-frame template
is accepted and cropped by the core. A custom mask can be ROI-local or
full-frame. `context-roi` is optional metadata for a wider matching context;
it does not enlarge the edit mask.

Create, list, calibrate, preview, and verify profiles with one CLI:

```powershell
$py = '.venv\Scripts\python.exe'
& $py scripts\logo_profiles.py list
& $py scripts\logo_profiles.py create `
  --id hi-anime --name 'hi!anime.at' `
  --image .cache\qa\episode_1_020s.png `
  --roi 1440 0 1920 256 --mask-rect 1718 18 1910 72
& $py scripts\logo_profiles.py calibrate `
  --profile muse --video 'source\The World''s Strongest Witch Episode 1.mp4' `
  --positive-time 110 --negative-time 20 --negative-time 70
& $py scripts\logo_profiles.py preview `
  --profile hi-anime --video 'source\episode.mp4' --time 20 `
  --out .cache\logo-preview\hi-anime-20
& $py scripts\logo_profiles.py verify `
  --profile muse --video 'source\The World''s Strongest Witch Episode 1.mp4' `
  --present-time 110 --absent-time 20 --absent-time 70 `
  --out .cache\logo-verify\muse --overwrite
```

`calibrate` and `verify` are CPU-only detector tests. They use unprocessed
native source images/frames and do not require or download the inpainting
model. The verifier uses deterministic dummy inpainting, so it checks the
detector/mask contract and exact pixels outside the mask; it does not prove
Anime-LaMa cleanup quality. Use `calibrate --write --overwrite` only after
reviewing the proposed threshold. `preview --dry` also avoids model
inference; a normal preview uses the prepared local model and the runtime GPU
policy. A profile is complete only after a real GPU preview, visual
before/after/mask inspection, and an outside-mask audit. Model preparation is
a separate explicit operator command.

## Known source coordinates

For the supplied 1920×1080 source episodes, the reconnoitered positions are:

| Profile | Model ROI | Recommended mask | Notes |
| --- | --- | --- | --- |
| `hi-anime` | `[1440, 0, 1920, 256]` | `[1718, 18, 1910, 72]` | `hi!anime.at`, present in sampled dark/bright scenes |
| `muse` | `[1440, 824, 1920, 1080]` | `[1740, 915, 1905, 1070]` | MUSE badge/text, present in the sampled Episode 1 110 s frame and absent in other checked samples |

These coordinates are starting measurements for those source dimensions. If a
release changes the logo location, size, aspect ratio, or shape, create or
adapt a profile from an unprocessed native frame and recalibrate with positive
and negative samples. Do not claim coverage for unseen logos or all scenes
from a small calibration set.

## Verification evidence

For each profile, check logo-present and logo-absent frames, bright and dark
backgrounds, several timestamps, and multiple resolutions/aspect-compatible
scales. Inspect every expected label from the image itself before passing it
to the verifier; a timestamp is not evidence that a logo is absent. In the
supplied Episode 1 set, hi!anime.at is present at all sampled timestamps, and
the MUSE profile is present at 110 s and absent at the checked 20 s and 70 s
samples. The verifier checks:

* profile detection status matches the expected present/absent label;
* native image dimensions and exact video PTS metadata are retained;
* every pixel outside the returned edit mask is equal to the input;
* an absent logo leaves the whole frame pixel exact; and
* the original source remains available and is never used as an output target.

Open generated before/after/mask artifacts with Codex image view or the
Computer Use UI to confirm visually that the mask is tightly shaped around the
logo and excludes nearby content. Keep logo removal default-enabled in the UI,
with an explicit option to turn it off for users who need the untouched
export. A profile matcher cannot identify an arbitrary unseen logo without a
registered local template/profile.

## Recorded integration verification (2026-10-16)

- The original inpainting algorithm was copied into this repository's
  `anime_picker/logo_removal.py`; no runtime import or path refers to the old
  workspace. Verified existing Anime-LaMa weights can be prepared with network
  access blocked, without downloading them again.
- Native 1920×1080 samples at Episode 1 times 20/60/70 seconds matched only
  `hi-anime`; at 110 seconds, both `hi-anime` and `muse` matched. CUDA
  inpainting removed the registered logos. Before/after images were visually
  reviewed and pixels outside the returned edit mask matched exactly.
- MUSE-only absent samples stayed pixel-identical without loading the model.
  At 960×540, some faint hi!anime.at samples conservatively skipped at the
  native profile's threshold. A smaller or fainter logo needs its own inspected
  calibration samples; thresholds were not lowered globally to force a match.
- Browser job `37b0e13c5c4c4b989146ce853cd0350d` confirmed default ON, automatic
  and explicit profiles, processed/skipped previews, and ON/OFF export.
  The ON export retained a source-identical `originals/frame_0001.png`; its
  cleaned PNG changed 10,669 pixels, all inside the declared edit mask. The
  OFF PNG was source-identical. Both remained 1920×1080 at PTS 6305490.
- External connections/model downloads were blocked during browser validation;
  both attempt counters remained zero. Local help loaded successfully, and no
  browser console errors were observed.

Private images, calibrated profiles, and machine-readable evidence are under
`.local/logo-profiles/` and `.cache/logo-qa/` and are excluded from Git. The
Codex Skill and reusable CLI are versioned; a fresh checkout needs its own
local profile assets.
