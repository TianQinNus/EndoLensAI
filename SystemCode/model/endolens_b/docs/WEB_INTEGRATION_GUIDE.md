# Web integration guide

English repository edition,22 October2026. The original user-provided v1 specification is retained as INPUT_INTEGRATION_SPEC_v1_DRAFT.md and remains proposed. Its polyp-pvt-v1, area0.01 and example counts/FPS are placeholders, not facts about this release.

## First acceptance target

Upload recorded MP4 -> create asynchronous analysis -> poll progress -> retrieve sampled frames -> overlay exact source/mask -> consistent candidate timeline/keyframes -> JSON report. The model segments RGB frames; the backend owns PTS, task records, postprocessing and statistics; the frontend consumes those saved records.

Also implemented: cancellation, separate PNG review revisions/conflict checking and JSON/CSV metadata exports. Not implemented/validated: full brush/undo UI, PDF, production auth/queue, clinical diagnosis, unique-lesion tracking or live endoscope support.

## Mapping the existing page

| UI element | Actual field and semantics |
|---|---|
| Model selection | GET /api/v1/models; only unet-b-expanded-seed42-r98-v1; disable E0/ensemble placeholders |
| Live Analysis | Label Recorded Video Analysis for this release |
| Candidate Polyps | Rename Candidate Frames; use status.progress.candidate_frames |
| FPS | performance.processing_fps is successful sampled frames / task elapsed time, not video playback FPS |
| Main mask | Pause on selected saved source_frame_url and mask_url; do not leave an old mask over continuous unrelated video frames |
| Detection list | One FrameResult per row, not one independent polyp |
| Confidence | Rename Foreground Score; foreground_score, null shown as a dash |
| Time | timestamp_seconds; retain exact value, format mm:ss only for display |
| Frame x/total | frame_index is0-based; display+1 if desired, preserve underlying index; unknown total stays unknown |
| Timeline | Only analyzed frames have decisions; failed/null and unsampled are unknown, not normal |
| Keyframes | GET /keyframes: up to60 chronological uniform samples, not lesion deduplication |
| Review | Separate original AI mask and edited_mask_url, clearly label source |
| Reports/counts | Use saved backend report/status, not current page length |

foreground_score is the mean model foreground score inside the largest predicted connected component. It is not frame-level disease probability, pathology or calibrated diagnosis confidence. Without GT, do not invent Dice/IoU/accuracy in the UI.

## Merge with the existing React app

Merge frontend_examples/endpoints.ts into the existing src/utils/endpoints.ts, preserving unrelated functions. Place FrameOverlay.tsx in src/components and adjust its import to ../utils/endpoints. Keep deterministic mock mode clearly separate from real results. Do not replace or reorder the existing UI just to use these examples.

For Vite, set VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1 in .env.local and restart Vite. Same-origin proxy setups can use /api/v1. Resolve returned resource paths using assetUrl; avoid duplicated /api/v1 prefixes.

~~~ts
const uploaded = await uploadVideo(file);
videoElement.src = assetUrl(uploaded.video_url);
const task = await startAnalysis(uploaded.video_id, 1.0);
localStorage.setItem("endolensTaskId", task.task_id);
const controller = new AbortController();
void followAnalysis(task.task_id, (status, frames) => {
  setProgress(status.progress);
  setFrames(frames);
}, controller.signal).catch(showRequestError);
// On unmount: controller.abort().
// User-requested server cancellation: cancelAnalysis(task.task_id).
~~~

Restore task_id after refresh and GET saved status/frames. Aborting frontend polling does not cancel the backend task. After editing, refetch frame records/revision_id; incremental creation cursors do not report modifications to an old frame. No actual teammate React build/browser E2E has been run with these snippets.

## Pagination, timing and geometry

GET /api/v1/analyses/{task_id}/frames?after=42&limit=100&candidate_only=false.

next_cursor is null on the current final page. Added last_cursor retains the highest frame-index watermark for later polling; has_more indicates existing additional pages. Merge by frame_id and sort by frame_index. Reset the cursor when changing filters.

Indexing has unknown target_frames/percent, not a false0% or failure. The backend sequentially decodes twice, first to establish exact sampling targets and then for inference. Timestamps are frame PTS*time_base minus container start time, not frame_index/fps. Missing or non-increasing PTS is rejected. Variable-rate tests cover this distinction.

Coordinates are original-frame pixels with origin top-left. Display contours are simplified, capped at100 contours/256 points and may omit geometry; display_geometry_truncated indicates truncation. RETR_LIST has no hole hierarchy. The native binary mask is authoritative, not a filled set of simplified contours.

The canvas example composites RGB/mask at native bitmap dimensions and CSS-scales afterward, preserving mask holes. Exact review uses saved source frames because browser video seeks can land on an adjacent frame.

## SDK and configuration snapshots

~~~python
from endolens_model import ModelAdapter, MODEL_VERSION
adapter = ModelAdapter(model_dir="/absolute/path/endolens_b", device="cpu")
result = adapter.predict(frame_rgb_uint8, MODEL_VERSION)
~~~

Load once. Module-level predict also provides a lazy singleton. Output is native float32[H,W] plus model_version, inference_time_ms and device. RGB uint8 is strict; convert BGR explicitly.

Preprocessing is Pillow bilinear352 square stretch, /255, ImageNet mean/std; FP32 logits are resized natively before sigmoid. No min-max, AMP inference, TTA, fusion, letterbox or mask filtering is added.

inference_time_ms covers adapter preprocessing, transfers, forward, native resize/sigmoid and CPU output copy, excluding queue/decode/postprocessing/storage. processing_fps includes indexing/decode/storage task overhead. They are not interchangeable.

Task configuration stores model version, source checkpoint SHA, exported weight SHA, preprocessing, actual device and postprocessing version/thresholds. A changed promoted weight needs a new identity/export, never a silent overwrite under the same version.

## Candidate gate and incomplete results

Only segmentation threshold0.5 preserves the retained research protocol. The proposed area0.01 was not validated and could suppress small targets, so it is not adopted as a scientific decision. Default min_region_area_ratio0 treats any nonempty component as a candidate; empty remains false. ENDO_MIN_REGION_AREA_RATIO can set a demo rule before startup, frozen per task and explicitly marked UNVALIDATED_ENGINEERING_DEMO. It never deletes the stored mask.

No negative-frame specificity validation exists. candidate=false means the engineering rule did not trigger, not confirmed normal. Failed/null and unsampled are unknown; failed frames retain errors and completed tasks may warn HAS_FAILED_FRAMES. All-frame failures are failed, not success.

## Human reviews and exports

PUT /api/v1/frames/{frame_id}/review uses multipart edited_mask, review_status reviewed/rejected and notes<=2,000 characters. Masks must be native-size single-channel binary PNG with0/1 or0/255. First base_revision_id is empty; subsequent saves must send the current revision. Stale base returns409; reload and ask the reviewer, do not overwrite silently.

AI mask_url never changes. Human revisions have new immutable URLs and records; edits do not propagate to neighbors. AI report statistics remain based on original AI predictions; human_reviews are separate. include_reviewed_masks in exports means revision references, not embedding all PNGs in CSV/JSON. PDF and binary ZIP export remain Web-team work.

## Deployment limits and errors

Single Uvicorn worker, no reload; storage is protected against concurrent owners. Default local CORS origins include localhost5173/3000 and127.0.0.1:5173; ENDO_CORS_ORIGINS accepts explicit origins. ENDO_STORAGE_DIR defaults to package runtime_data and must not point into training runs.

422 wrong model_version: use the exact B identifier. Integrity failure: retrieve the complete selected binary weight and verify, do not disable checks. CUDA unavailable: select CPU or install matching official wheels. Sampling limit: use longer intervals/shorter clips, never silently truncate the remainder. Codec/rotation rejection: transcode upright square-pixel H264 first. Restarted running/queued tasks are marked failed with partial records; create a new task. Storage full: preserve results and choose controlled new storage, not untracked deletion of AI assets.

The reference API has no production auth, TLS, tenant separation, hostile-media decoding sandbox, persistent distributed queue, automatic retries or retention system. Multipart is parsed before the application's size bound; use reverse-proxy limits in a real deployment. Do not publish this local research service on the internet or process identifiable clinical video.

## Evidence and remaining acceptance

Original evidence: three historical GPU score maps matched frozen B exactly; real CPU synthetic fixture;15 API/model/edge tests; real B+synthetic video complete API smoke; actual Uvicorn HTTP upload/Range206/assets/report/CSV; cancellation, failure-null, persistence/restart, immutable review conflict and variable-PTS tests.

Controlled failure/cancel API tests use explicit synthetic stubs; the separate real-model smoke is not a clinical accuracy/FPS benchmark. Repository English migration tests are additional, not a reinterpretation of those original records. The existing React project has not yet been merged/built/tested end-to-end with this release.

Web team must confirm target device, UI label changes, upload limits, editor/PDF scope and production security. Research confirmation/negative-frame safety and any candidate threshold need separate audited experiments.

Implementation references: [FastAPI uploads](https://fastapi.tiangolo.com/tutorial/request-files/), [lifespan](https://fastapi.tiangolo.com/advanced/events/), [PyAV frame timestamps](https://pyav.org/docs/stable/api/frame.html), [Starlette responses](https://www.starlette.dev/responses/). These describe software behavior, not clinical validity.
