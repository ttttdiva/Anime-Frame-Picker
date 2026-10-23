# AI agent headless batch operations

This is the required runbook for `scripts/batch_source.py`. It processes the
top-level MP4/MKV inputs in `source/` sequentially and does not require the Web
UI. A full source-video batch is a separate action from implementation and
fixture verification; do not start it unless the user explicitly asks for the
production extraction.

## Four exclusive modes

- `--mode cute` is the default. It starts from resumable scene/coverage scan
  candidates, runs the GPU-backed anime person/head detectors and local PixAI
  tagger, excludes frames without character evidence, and ranks expression,
  face visibility, composition, and sharpness. The default cut-quality floor
  is `0.55` and is configurable with `--cut-quality-min`; it selects by score
  and does not cap the result count. The floor is part of the cute run config
  hash, so a run completed with another floor is never skipped as complete. Global pHash/dHash neighbor search
  nominates visual pairs without a time cutoff; whole-frame and 4x3 local
  hashes, pixel/SSIM, registration, expression, and person-count evidence make
  the final decision.
  Strong opposing expression changes and material person-count changes are
  protected. Character presence uses general PixAI person tags at score `0.17`
  or higher, strong person/head boxes as supporting evidence, and a content
  gate against near-empty or watermark-only frames. Cute does not run CCIP or
  form identity clusters. PixAI inference requires CUDA PyTorch and detector
  inference requires CUDA ONNX Runtime; either stage fails without a CPU retry.
- `--mode all` is an explicit comprehensive scene-candidate alternative. It
  selects scene anchors, coverage anchors, and local visual-change peaks across
  the whole video. It has no identity-model stage and does not infer which
  characters appear. Candidate output has no fixed count cap; coverage spacing
  controls regular anchors, while scene boundaries and local-change peaks add
  candidates.
- `--mode character` (legacy alias `manual`) uses explicit reference crops to
  find matching candidates. Every video needs an exact-stem reference in
  `source/refs` or a shared `--reference` image. Missing references are an
  error; this mode never falls back to automatic character selection.
- `--mode auto-topn` (legacy alias `auto`) opts into frequency-ranked recurring
  character groups. `--top-n N` (legacy alias `--characters-per-video`) sets
  the maximum number of selected groups and defaults to 2. It is a bounded
  top-N identity subset, not comprehensive identity coverage. Any positive N
  is accepted; there is no arbitrary 12-character cap. Identity grouping and
  frequency are heuristics, not identity guarantees.

The scan decodes every presentation frame sequentially at up to 960px long edge
and retains small per-frame summaries, including 48x27 grayscale thumbnails.
Those thumbnails help find scene/local-change candidates and skip obviously
dissimilar frames during duplicate checks; they cannot by themselves reject a
candidate as a near-duplicate. Cute mode checks the staged candidate image (up
to 960px long edge) using whole-frame and local structure/color evidence,
small-translation registration, and expression/person-count protections. A
low visual-information gate blocks empty frames that receive spurious person
tags. Brightness and contrast are recorded as diagnostics and are not absolute
exclusion gates. The `0.55` cut-quality floor combines expression, face
visibility, composition, presence, and technical sharpness; it is a practical
ranking filter, not a universal aesthetic or anatomy detector. Scene cuts do
not by themselves imply a good image.

## Dry run and execution

```powershell
Set-Location 'D:\Dev\92_Anime_Frame_Picker'
.\.venv\Scripts\python.exe scripts\batch_source.py --dry-run
```

The dry run lists the actual files and selected mode. It does not decode video,
load models, write output, or archive inputs. For ordinary character-focused
curation, run:

```powershell
.\.venv\Scripts\python.exe -u scripts\batch_source.py
```

Explicit alternatives:

```powershell
# Explicit comprehensive scene/coverage/local-change candidates.
.\.venv\Scripts\python.exe -u scripts\batch_source.py --mode all

# Character-presence, cut-quality, and visual near-duplicate curation (default).
.\.venv\Scripts\python.exe -u scripts\batch_source.py --mode cute

# Raise or lower the cut-quality floor without imposing a result-count cap.
.\.venv\Scripts\python.exe -u scripts\batch_source.py --mode cute --cut-quality-min 0.55

# Explicit references only; every video needs an exact-stem reference crop.
.\.venv\Scripts\python.exe -u scripts\batch_source.py --mode character

# Opt-in frequency-ranked top-N subset; select three groups per video.
.\.venv\Scripts\python.exe -u scripts\batch_source.py --mode auto-topn --top-n 3

# One common, tightly cropped identity reference for every video.
.\.venv\Scripts\python.exe -u scripts\batch_source.py --mode character --reference 'D:\refs\character.png'

# Change coverage spacing without limiting the number of results.
.\.venv\Scripts\python.exe -u scripts\batch_source.py --candidate-spacing 0.8

# Limit per-video FFmpeg/PyAV decoder work while other CPU jobs are active.
.\.venv\Scripts\python.exe -u scripts\batch_source.py --decode-threads 1
```

PixAI conditions use exact supported general vocabulary, supplied as
`--want subject:smile`, `--avoid frame:multiple_people`, and similar repeated
options. Subject conditions require character or auto-topn; cute uses full-frame
tags for presence and does not treat them as character crops. Frame conditions
can be used in any mode. Logo removal is enabled by default; use
`--no-logo-removal` to export untouched source pixels only.

## Shared analysis cache and output resume

`.cache/batch/<source-sha>/scan/<scan-key>/` stores sequential low-resolution
frame summaries keyed by video fingerprint and scan preprocessing revision.
`.cache/batch/<source-sha>/identity/<identity-key>/` stores selected-PTS
detections and CCIP features keyed by scan key, model revision,
detector/CCIP preprocessing revision, and chunk PTS. `.cache/batch/<source-sha>/tags/`
stores PixAI results by pixel digest and tag revision. Cute detections are
cached under `detections/<detector-revision>/pts_<pts>.json`; each entry records
the exact source fingerprint, candidate image digest, model revisions, and
preprocessing revision. Cute mode also retains staged analysis PNGs and a
reasoned `curation-report.json` in the private run folder, separate from public
extracted images. Compatible identity evidence is reused between `character`
and `auto-topn`; cute intentionally does not import bounded legacy identity
groups. Curation reports and output metadata retain only person/expression/
framing tags; full PixAI score maps remain in the private tag cache. When a new
curation revision uses the same source fingerprint and selected PTS list, its
candidate PNGs are linked from an earlier private run instead of being decoded
again. Completed checkpoints are written atomically and validated before reuse;
a changed input, candidate PTS list, model, or preprocessing revision rebuilds
only the affected chunk set. The same scan and selected-PTS evidence is reused
rather than reopening a second scene-detection pass.


Public PNGs are under `outputs/<video name>/images/` for both the UI and batch
runner. Candidate filenames preserve exact PTS (`frame_pts_<pts>.png`). If a
different image already occupies that PTS name, a stable content-hash suffix
is added; existing images are never overwritten. Sanitized or colliding video
names use a stable short source-identity suffix. Batch run manifests and
reports are under `.local/anime-picker/batch/` by default; `--state-root`
overrides that private location and `--output-root` selects the public root.
The state root must stay outside the public root and on the same volume so the
runner can switch complete image sets atomically.
The scan/cache data remains in `.cache/batch/`. Selected PTS values are
decoded in one ordered pass at native resolution. Existing valid partial
outputs and analysis chunks are reused on restart; no random retry folders are
created under the public images directory. `--force` creates a separate
private run and publishes only a fully verified image set. A superseded public
set is moved intact under `outputs/_archive/` before the new set becomes active.

Existing historical outputs, including earlier capped per-character runs, are
preserved as legacy data. They are not acceptance criteria for comprehensive
candidate extraction.

Each invocation writes `batch_run_*.json` to the private state root; each video
has a private `batch-result.json`, cute mode also writes `curation-report.json`,
and each group has private `metadata.json`. The public `images/` directory
contains PNGs only. Cute exports only character-present representatives;
`curation-report.json` records no-character exclusions and lower-ranked
near-duplicates with their reason and representative ID. Reports include scan
coverage, candidate reasons, cache hits, PTS, model calls and stage timings, process-local PyTorch
allocated/reserved GPU memory, and cleanup observations. PyTorch allocator
counters exclude ONNX Runtime and other processes; `empty_cache` cannot free
live tensors and is not a total VRAM cap.

For isolated runs, pass separate `--output-root`, `--state-root`, and
`--cache-root` paths; `--model-cache-root` moves Hugging Face, imgutils, and
Torch runtime cache writes out of the prepared model bundle. `--model-root`
continues to point at the read-only prepared weights.

## Cross-process exclusion (UI and batch)

The Web UI and `batch_source.py` share OS-level locks in one machine-wide
directory (`%LOCALAPPDATA%\anime-picker\locks`, or `ANIME_PICKER_LOCK_DIR`),
independent of checkout and `--state-root`; they are released automatically if
a process dies:

- **Video lease** - one process works on a given video (by fingerprint) at a
  time. A batch that finds a video held by the UI or another batch records that
  video as `busy`, leaves it in `source/`, and continues with the others. The
  UI answers HTTP 409. Different videos never wait on each other.
- **GPU gate** - taken only around a model call or model load, never across
  decoding, scanning, de-duplication or export. It serializes GPU compute, not
  VRAM residency: each process keeps the weights it loaded resident between
  calls, so check free VRAM before starting a second model-using process.
- **Publication lock** - keyed by the resolved public output root and video,
  one public image set per video is switched at a time, for both writers.

Public publication state (folder-name registry, active manifests, history)
lives in `<output-root parent>\.local\anime-picker-<root hash>` for every
writer of that output root, whatever its checkout or `--state-root`; private
run state (`--state-root`, UI exports) stays per caller.

A batch started before these locks existed holds none. Lock-aware processes
find it read-only from the process list (`batch_source.py` command lines;
dry runs ignored) and refuse conflicting starts: any GPU start (including a UI export with logo
removal), and model-free batches or exports to the same output root. They never stop or signal it. A new batch that needs a
model exits with code 3 while such a batch runs.

## Completion and archive rules

Only videos with a verified nonempty set of valid output PNGs are marked
completed. After the final batch report is written, completed video files and
exact-stem `.vtt`, `.srt`, `.ass`, or `.ssa` subtitles move from `source/` to
sibling `completed/`. Failed, interrupted, or no-match inputs remain in
`source/`. Existing destination files are never overwritten. Archive status and
collisions are recorded in the final report. `--no-archive` disables movement.

Before any production GPU run, inspect existing GPU work and avoid contention.
Do not interrupt another process. Never delete or overwrite source videos,
subtitles, older outputs, model files, or existing cache data during an
implementation or fixture test. Do not stage any of those private/generated
files in Git.

## Verification

Use CPU-only fixtures and synthetic videos for code checks. Do not invoke the
full source batch, even as a test. A successful dry run is not a completed
extraction. For production work, verify the final report, every saved PNG, the
exact-PTS metadata, and the `archive` receipt before reporting completion.
