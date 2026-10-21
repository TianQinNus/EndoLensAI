# EndoLens B model integration package

English repository edition,22 October2026. Research/teaching integration prototype, **not clinical software**. First release processes recorded videos, not live endoscope streams.

The only released version is **unet-b-expanded-seed42-r98-v1**, original frozen E8 B. It is an ImageNet-pretrained ResNet34-U-Net selected at round98. It is not incomplete E11, E0, an ensemble or a teammate's scratch U-Net++ model. The export contains all inference parameters; no ImageNet download is needed at model initialization.

## Contents

- weights/ and model.json: selected portable parameters, exact identity/preprocessing and checksums.
- endolens_model/: standalone RGB frame adapter and native-resolution postprocessing.
- backend/: FastAPI reference API for upload, async tasks, frames, immutable AI masks, human revisions and JSON/CSV.
- frontend_examples/: React request/polling and mask-overlay examples to merge into the existing app, not a complete frontend.
- examples/: explicitly synthetic image/video, actual model output and standalone/HTTP smoke commands.
- verification/: original cache parity and software test evidence, not new medical performance estimates.
- docs/: [integration guide](docs/WEB_INTEGRATION_GUIDE.md), [model card](docs/MODEL_CARD.md), OpenAPI and original proposed integration specification.

## Clone with the actual weight

The complete97,893,041-byte selected weight is committed as a regular binary Git file. This public fork rejected new Git LFS objects, so normal git clone retrieves the actual weight without an LFS step. The package verifier must pass; do not disable it. This directory can then be copied as a complete standalone package.

## CPU quick start on Windows

Enter this directory. Use a new environment rather than changing an active training environment:

~~~cmd
conda create -n endolens-web python=3.11 -y
conda activate endolens-web
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python verify_bundle.py
python run_tests.py
start_backend.cmd
~~~

Open http://127.0.0.1:8000/docs or /openapi.json. Swagger assets may need network; installed API/model inference does not. In a second terminal with this environment and directory:

~~~cmd
python examples/api_smoke.py
~~~

Success validates actual HTTP upload, video Range206, asynchronous processing, mask assets, report counts and CSV download on synthetic graphics. It does not measure clinical accuracy or deployment FPS.

For an appropriate CUDA12.8 setup, replace the CPU torch installation with the official cu128 index. Do not install both variants. ENDO_DEVICE defaults to auto; set ENDO_DEVICE=cpu to fix CPU, or cuda:0 explicitly. Actual device is reported. Cross-hardware floating-point equality is not guaranteed.

## Existing backend only needs the SDK

Keep endolens_model/, model.json and weights/ together, and load once at application startup:

~~~python
from endolens_model import ModelAdapter, MODEL_VERSION
adapter = ModelAdapter(device="auto")
result = adapter.predict(rgb_uint8_hwc_frame, MODEL_VERSION)
probability = result.probability_map  # float32[H,W], native resolution
mask = (probability >= 0.5).astype("uint8")
~~~

An explicit package root can be provided through model_dir. OpenCV BGR input must be converted to RGB. No training project, optimizer state or dataset is needed.

## Important boundaries

- Segmentation threshold0.5 preserves B's protocol. Candidate area gating is unvalidated, default0 with empty masks always false; it affects the candidate flag, not the stored AI mask. Snapshot every task's setting.
- Candidate counts are frames, not distinct polyps. Foreground scores are not calibrated clinical confidence. Negative-frame specificity, tracking, pathology and real-time performance are unvalidated.
- Native frame index is0-based and timestamps use source PTS. During indexing, target_frames/percent are null. Failed candidate flags are null; unsampled frames are unknown.
- Exact overlays use saved source_frame_url, not an adjacent HTML video seek frame. Original decoded RGB is saved losslessly as PNG; human masks are separate immutable revisions.
- Single process/one worker/no reload. SQLite persists records; interrupted queued/running tasks become failed on restart with partial results retained, not automatically resumed.
- Localhost-only reference server, no auth/TLS/user isolation. Do not expose publicly or process identifiable clinical records.
- Initial limits: actual H264 MP4/MOV-family container with .mp4 upload name, upright square pixels, at most1080p pixel count,100MiB,15 minutes,60,000 decoded frames and600 sampled frames. Use shorter authorized test clips or longer intervals; other codecs/rotation need prior transcoding.
- Approximate2GiB storage budget; lifecycle cleanup is not implemented. Multipart parsing occurs before application size rejection, so production request limits need a reverse proxy.
- Review API accepts native binary PNG and rejects revision conflicts409. The package does not supply a complete brush/undo UI, PDF, production queue or binary mask-bundle export.

Read [the model card](docs/MODEL_CARD.md) and [third-party notices](THIRD_PARTY_NOTICES.md). This English edition changes documentation, not the selected weight/inference protocol. Historical verification evidence is labeled by scope; new repository checks are recorded separately.
