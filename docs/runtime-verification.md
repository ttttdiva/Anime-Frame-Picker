# Real-video runtime verification

`scripts/verify_real_video.py` is the bounded smoke check for a real source
file. It exercises the model, frame reader, search pipeline, native PNG export,
tag scope plumbing, and pixel-only duplicate review without uploading source
footage. The fixture is local Episode 1 and the generated files stay under
`.cache/qa`.

The fixture reference is the blue-haired, yellow-eyed central character at
about 70 seconds. The default search interval is 55–85 seconds, which includes
the blonde knight scene near 60 seconds and the red-haired scene near 80
seconds. The interval is a QA fixture chosen for wrong-character and
multi-person coverage; it is not a production rule or a claim about the
source's other scenes.

Before a GPU run, coordinate with the parent agent so another model process is
not using the GPU. The command requires `require_gpu=True` and does not open a
console window or start a server:

```powershell
D:\Dev\92_Anime_Frame_Picker\.venv\Scripts\python.exe `
  D:\Dev\92_Anime_Frame_Picker\scripts\verify_real_video.py `
  --model-root D:\path\to\model-root
```

The script discovers `anime_picker.models:ModelService` by default. A different
module can be supplied with `--model-service module:ModelService`, or through
`ANIME_PICKER_MODEL_SERVICE`. `ANIME_PICKER_MODEL_ROOT` can provide the model
root instead of `--model-root`.

The first search has no tag conditions and reuses one cache root for the
optional subject-scope pass. By default the second pass requests
`subject:smile`; disable it with `--no-scope-pass`. Explicit conditions can be
added to the primary pass, for example:

```powershell
...\python.exe ...\verify_real_video.py `
  --model-root D:\path\to\model-root `
  --wanted subject:smile `
  --avoided frame:multiple_people
```

Each candidate's `pts` is read again with `VideoReader.get_frame(pts,
max_side=None)` and exported with `VideoReader.export_png`. The report compares
the exported PNG's RGB array with the exact `get_frame` RGB array and compares
the candidate timestamp with the source frame timestamp. A candidate preview
is required to exist, but its JPEG pixels are not treated as a native-source
comparison because the pipeline samples previews at `max_side=960` and stores
them as JPEG.

The report also records the image shapes passed into `detect`, `embed`, and
`tag`. The identity input audit flags both native full-frame dimensions and
the full analysis-frame dimensions seen by `detect` (normally `960x540`),
which would indicate that CCIP evidence was taken from a whole multi-person
frame instead of a detector crop. Candidate normalized boxes are checked for
whole-frame boxes. The optional scope pass verifies that returned tag data has distinct
`subject` and `frame` fields; the model call log shows crop-sized and
full-frame-sized tag inputs separately. Even when the primary run has no tag
conditions, the script makes one diagnostic tag call on a returned crop and
one on its exact full source frame, so the two input shapes are visible in the
report. The primary stats also preserve the pipeline's
`multi_person_full_frame` rejection count for the mixed-person portion of the
interval.

Default negative crops are a blonde character around 60 seconds and the
left-side red-haired character around 80 seconds. Their distances are reported
against the 70-second blue-haired reference. A lower positive distance is
useful evidence for this fixture, but the script does not turn two negatives
into a precision or recall guarantee. The boxes can be replaced when a source
cut changes:

```powershell
...\python.exe ...\verify_real_video.py `
  --model-root D:\path\to\model-root `
  --negative blonde@60:0.14,0.06,0.38,0.99 `
  --negative red@80:0.00,0.00,0.47,0.96
```

The initial source contact sheet is at
`.cache/qa/contact.jpg`; it labels frames at 20, 30, …, 130 seconds. After a
successful run, inspect `.cache/qa/candidates-contact.jpg` manually for
wrong-character leakage and near-duplicate expressions. The machine-readable
results are in `.cache/qa/real-video-verification.json`, with native reference
and candidate exports next to it. A successful status means the reference
export and every returned candidate export matched RGB exactly and candidate
timestamps matched source PTS times, while identity inputs and returned boxes
also stayed crop-scoped. It does not mean that every visually good frame was
found. Any candidates within 0.55 seconds are listed as
`nearby_candidate_pairs` and require that contact-sheet review even when the
pixel-hash heuristic finds no exact duplicate.

Standard output is a compact summary so tag vocabularies do not flood the
terminal or CI log; the full tag maps remain in the ignored JSON report. The
recorded run also has a compact summary at `.cache/qa/real-video-summary.log`
and a raw progress/trace log at `.cache/qa/real-video.log`.

The script exits with code 2 when the source or model root is unavailable, code
1 for a runtime or evidence failure, and code 0 for a passing bounded smoke
check. Synthetic VFR and B-frame behavior belongs to `tests/test_video.py` and
is intentionally not duplicated here.

The recorded GPU smoke on 2026-10-16 used `D:\AI\models\Cold_F\image` (the
runtime resolved the existing junction to `F:\AI\models\Cold_F\image`) with
the manual reference box `0.30,0.0,0.72,0.9`. The primary and `subject:smile`
passes each sampled 250 frames and returned six candidates. All six primary
candidate exports matched the exact source RGB arrays and all candidate times
matched their source PTS times. The target reference distance was `0.01862`;
the blonde-knight and red-haired fixture negatives were `0.48967` and
`0.43369`. The run recorded two `multi_person_full_frame` rejections, no
whole-frame identity inputs, and distinct crop/frame tag inputs. Cache reuse in
the scope pass was visible as 1,233 hits and 30 misses. The candidate contact
sheet shows four close-in-time pairs for manual review (70.04/70.20 and three
near 76 seconds); the pixel-hash heuristic alone is not a duplicate-quality
pass. The process used the pre-follow-up pipeline import, so these artifacts
are source-proof for that run and do not validate a later same-PTS collapse
change.

## Final integration checks (2026-10-16, JST)

- Browser job `927da0bc34f2432ca5bd708e586fba07` searched 68–82 seconds,
  displayed 20 candidates, compared two, and exported two 1920×1080 PNGs to
  the historical destination
  `D:\Work\Download\The World's Strongest Witch Episode 1_20261016_125515_4ebb70ea`.
  Both exported RGB arrays matched exact source PTS 7227090 and 6312690.
  Cancellation stayed pending until the worker actually stopped.
- The final short browser job `3e911b82d2634baea73d4506a4b42409` ran with
  external sockets and Hugging Face downloads blocked. It analyzed 11 frames
  from 70–70.4 seconds, tagged one pixel-diverse candidate, returned one
  candidate and reported 19 missing. It reused 18 cache entries. The complete
  public job response was 3,148 bytes. External socket/download attempt
  counters both remained zero.
- Single-frame enlargement, target rectangle/crop, Escape and arrow controls
  worked. The exported PNG at PTS 6305490 was 1920×1080 and RGB-identical to
  the decoded source. Output:
  the historical destination
  `D:\Work\Download\The World's Strongest Witch Episode 1_20261016_130211_b1588063`.
  Evidence is in `.cache/qa/ui-final-job.json`, `ui-final-export-check.json`,
  `offline-runtime.json`, and `ui-final-preview.png`.
- The two historical browser-export directories above were not migrated. New
  exports use the repository-root `outputs/` directory; the current full-run
  result directories are `outputs/Anime_Frame_Picker_full_Episode1_20261016`
  and `outputs/Anime_Frame_Picker_full_Episode2_20261016`.
- The CUDA bfloat16 tagger check used the same crop as a float32 baseline.
  `smile` changed from 0.86146 to 0.86328. Its first call took 13.10 seconds,
  the warm call 0.178 seconds; PyTorch peak allocated/reserved memory was
  3.07/3.28 GB. This is one-image inference evidence, not a video throughput
  benchmark. See `.cache/qa/bf16-smoke.json`.
- After the final head/body context fix, an offline one-frame
  `subject:upper_body` search evaluated both a 458×380 head crop and a
  919×540 body crop. Scores were 0.16699 and 0.56641; at the 0.17 threshold,
  the body candidate correctly survived. See
  `.cache/qa/pose-scope-check.json`.
- MP4 and MKV synthetic VFR/B-frame decoding/export checks both passed.
- A final condition-preservation fix removes only exact pixel duplicates before
  requested tag conditions are evaluated. The two-frame GPU check then tagged
  all four head/body alternatives, returned one frame, reported 19 missing,
  and produced a source-identical native PNG. External networking/downloads
  were blocked; CUDAExecutionProvider and bfloat16 were active. See
  `.cache/qa/final-conditional-check.json`.
- The final project virtual environment test run passed all 35 tests.

The full-video precision and anatomy/eye-state judgment remain heuristic.
These bounded checks do not establish performance or accuracy for every cut,
character, or video. All generated evidence and source footage are excluded
from Git.
