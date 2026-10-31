# Web migration guide: B v1 -> E13 region080 v2

## 1. Replace the complete model unit

Keep these resources together in their own directory:

```text
endolens_e13_region080_v2/
  endolens_model/
  model.json
  weights/e13_resnet50_joint_seed42_epoch94.pt
```

Load `ModelAdapter(model_dir="/absolute/path/endolens_e13_region080_v2", device="auto")`
once per backend process. Add that directory to `sys.path` before importing
`endolens_model`. Restart the process: Python may otherwise retain the already
imported old ResNet34 module. There is no need for the training repository,
ImageNet checkpoint, data archive, or original F: drive path on the recipient's PC.

The selected ResNet50 architecture differs from ResNet34. Renaming the new weight
to the old filename is not a migration and will fail strict state loading.

## 2. Use the new final output

```python
from endolens_model import ModelAdapter, MODEL_VERSION
adapter = ModelAdapter(device="auto")

result = adapter.predict(rgb_uint8, MODEL_VERSION)
mask_01 = result.binary_mask               # canonical FILTERED uint8 native mask
candidate = result.is_candidate           # exactly bool(mask_01.any())
regions = result.regions                  # retained regions only
contours = result.contours                # extracted from the filtered mask
score = result.postprocess_result.foreground_score  # largest retained region; null if empty
```

Replace the old `mask = (result.probability_map >= 0.5)` expression. The raw score
map remains available for compatibility/diagnostics, but thresholding it directly
reintroduces the rejected regions. If the existing backend already calls
`postprocess(result.probability_map)`, the v2 helper defaults to the same released
mean0.8 rule. Prefer the attached `postprocess_result` to avoid duplicate processing.

All geometry uses original frame pixels, origin top-left. Save `mask_01*255` as a
single-channel PNG. Use lossless saved source frames when exact playback/overlay
alignment is required. Video seeking can land on a different source frame.

The filter is **per connected region**, not per frame: do not retain all regions
because one region's mean is high; do not reject a small high-score region based
on full-frame mean; do not substitute the maximum pixel score. Scores are compared
before rounding. Use8-connectivity and the native probability map.

## 3. Existing frontend

Set the request version to:

```text
e13-r50-joint-r94-r4-region080-v2
```

The optional reference backend preserves:

| Action | Endpoint |
|---|---|
| Health / model description | GET `/api/v1/health`, GET `/api/v1/models` |
| Upload / stream | POST `/api/v1/videos`, GET `/api/v1/videos/{id}/stream` |
| Start / poll analysis | POST `/api/v1/analyses`, GET `/api/v1/analyses/{task}` |
| Frame results | GET `/api/v1/analyses/{task}/frames` |
| Candidate-only list | Same frames endpoint with `candidate_only=true` |
| Keyframes / report | GET `/api/v1/analyses/{task}/keyframes`, `/report` |
| Mask and source image | Returned `mask_url`, `source_frame_url`, `thumbnail_url` |
| Human review | PUT `/api/v1/frames/{frame_id}/review` |
| CSV/JSON export | POST `/api/v1/analyses/{task}/exports` |

Example analysis body:

```json
{
  "video_id": "ID_RETURNED_BY_UPLOAD",
  "sampling_mode": "interval",
  "interval_seconds": 1.0,
  "model_version": "e13-r50-joint-r94-r4-region080-v2"
}
```

Each task snapshots `input_size=320`, `tta=rotate4`, pixel0.5, region mean0.8,
area gate0, weight hashes and postprocessing version. Each completed frame includes
`raw_component_count`, `removed_component_count`, retained `component_count`,
`is_candidate`, `foreground_score`, `regions` and `contours`.

When all regions are rejected, the frame is still successfully processed. Display
the source image, clear previous mask/contour overlays, and omit it from the candidate
list. The API returns an all-zero mask PNG, `is_candidate=false`, `regions=[]`,
`contours=[]`, `foreground_score=null`. A failed/unsampled frame is a different
state; failed results have `is_candidate=null` and must not be counted as normal.

The browser must use server `is_candidate` and filtered assets. Do not derive
candidate status from raw probability maxima or reuse a previous frame's overlay.
Merge `frontend_examples/endpoints.ts` and `FrameOverlay.tsx` into the existing
React project. The canvas example clears its bitmap on frame change and uses the
native mask, including holes. No complete UI rebuild is needed.

## 4. Reference backend setup

Install dependencies using the README, then run `start_backend.cmd` from a terminal.
Use one worker and no reload. Optional environment values:

```powershell
$env:ENDO_DEVICE = "cuda:0"    # auto or cpu are also supported
$env:ENDO_STORAGE_DIR = "D:\EndoLensWeb\runtime_v2"
$env:ENDO_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
.\start_backend.cmd
```

Use a fresh storage directory. Old stored AI masks are immutable records of the
old model; start a new analysis to get v2 predictions. Nonzero
`ENDO_MIN_REGION_AREA_RATIO` now fails at startup rather than silently changing
the released model. There is no environment override for the region0.8 rule.

Backend device, model metadata and postprocessing rules can be checked through
`/api/v1/models`. `inference_time_ms` includes four forwards and region filtering;
`processing_fps` additionally includes queue-independent task/video/storage work.
Neither field is a promise of real-time performance. UI demo FPS values must be
replaced with actual task measurements.

## 5. Local replay and acceptance

```powershell
python verify_bundle.py
python run_tests.py
python examples/predict_image.py --image examples/synthetic_frame.png --output runtime_data/check.png
```

For HTTP/video checks, start the server and run `python examples/api_smoke.py`.
The synthetic fixture is labeled and contains no patient imagery. The smoke test
checks the real model, video upload/PTS sampling, mask/candidate consistency, assets,
candidate counts, keyframes and CSV download. Compare local CPU output to the
included synthetic probability map and filtered mask; hardware can cause small
floating-point differences near a cutoff.

Common issues:

| Symptom | Correction |
|---|---|
| 422 old model ID | Update frontend/body to the v2 ID and restart backend |
| State-dict mismatch | Replace architecture, model.json and weights together |
| Every frame has an old-looking mask | Use binary_mask, not raw probability >=0.5; reanalyze video |
| Candidate=false but overlay remains | Clear the canvas on every frame; use the current filtered mask |
| OpenCV import fails | Install requirements-model.txt or requirements.txt in the web environment |
| CUDA unavailable | Check wheel/device selection; explicit cpu is supported |
| Area-setting startup error | Remove old nonzero ENDO_MIN_REGION_AREA_RATIO |
| Slower than v1 | This release uses four rotations; do not remove TTA and reuse its scores |
| Video/size/task limit | Transcode supported H264 MP4, shorten clip or increase sampling interval |

Temporal confirmation and area filtering were not selected for this release. Any
later addition changes the model pipeline and should have its own version/evidence.
