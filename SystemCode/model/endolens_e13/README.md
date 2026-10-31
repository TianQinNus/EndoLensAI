# EndoLens AI: final team model handover v2

This standalone bundle contains the user-selected **E13 ResNet50 U-Net, epoch94,
with native connected-region mean score >=0.8**. It replaces the old ResNet34 B
handover. No training repository, raw dataset, original ImageNet weights, or Mac
paths are required. All code and documentation in this bundle are in English.

## Repository installation and external weights

This directory is the canonical SDK for the integrated repository backend.
The 124.37 MiB weight is external and intentionally absent from Git.
Download both unchanged parts from the [Google Drive release folder](https://drive.google.com/drive/folders/12nddUWrat2Ih4sHpa4c8M-DuO76tDEKO),
keeping the filenames in `weights_download.json`. From this directory run:

```powershell
python install_weights.py --parts-dir "$HOME/Downloads"
python install_weights.py --check
```

The standard-library helper verifies each part and the merged original weight.
If you already have the complete original file, use `--source /path/to/file.pt`.
Do not unzip the parts or load them separately. No weight conversion occurs.
Keep `endolens_model/`, `model.json` and `weights/` together. Do not load the E13
weight into the historical ResNet34 architecture.

For the integrated application, follow the [root startup guide](../../../README.md#4-installation-and-startup)
and [modular backend guide](../../backend/README.md). The `backend/` directory
inside this SDK is an optional standalone reference server with its own tests.
Run one server, not both on port 8000.

```text
model_version:          e13-r50-joint-r94-r4-region080-v2
postprocessing_version: native-cc8-pixel050-mean080-v1
```

## The exact released output rule

1. RGB uint8 native frame -> Pillow bilinear resize320 x320 -> /255.
2. The model applies ImageNet normalization internally, exactly once.
3. Predict rotations0/90/180/270; inverse-align and average **logits**.
4. Resize logits to original frame dimensions; sigmoid once, FP32.
5. Threshold pixels at probability >=0.5; identify8-connected regions.
6. Compute the original probability mean over each region's pixels. **Delete a
   region from the output mask if its unrounded mean is <0.8.**
7. `is_candidate` is true exactly when the filtered mask has foreground.

There is **no minimum-area filter, largest-only rule or temporal confirmation**
in this release. A tiny region with a sufficiently high mean is retained. Rejected
regions are deleted from mask PNGs, contours, region lists and candidate statistics,
not merely hidden from one label. A completely rejected frame returns an all-zero
native-size mask, `is_candidate=false`, empty contours/regions and a null score.

## Quick integration into an existing Python backend

Install the frame SDK dependencies in the web backend environment, then keep
`endolens_model/`, `model.json` and `weights/` together. Add the extracted package
root to Python's import path. Restart the old backend process before importing v2.

```python
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, "/absolute/path/endolens_e13_region080_v2")
from endolens_model import ModelAdapter, MODEL_VERSION

adapter = ModelAdapter(device="auto")  # load once at application startup

def analyze_frame(rgb_frame: np.ndarray):
    result = adapter.predict(rgb_frame, MODEL_VERSION)
    Image.fromarray(result.binary_mask * 255, mode="L").save("mask.png")
    return {
        "model_version": result.model_version,
        "is_candidate": result.is_candidate,
        "contours": result.contours,
        "regions": result.regions,
        "foreground_score": result.postprocess_result.foreground_score,
    }
```

**Use `result.binary_mask` for display and storage.** `result.probability_map` is
the raw native score map retained for diagnostics and legacy adapter compatibility.
The old expression `(result.probability_map >= 0.5)` bypasses the chosen filter and
must be replaced. OpenCV input is BGR: convert with `cv2.COLOR_BGR2RGB` first.
PyAV input should use `frame.to_ndarray(format="rgb24")`.

## Environment installation

Use Python3.11. The reference versions match the tested runtime; do not modify
an active training environment just to run this web package. Model loading is
offline after dependency installation.

```powershell
conda create -n endolens-web-v2 python=3.11 -y
conda activate endolens-web-v2
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python verify_bundle.py
python run_tests.py
```

For an existing compatible NVIDIA CUDA12.8 setup, use the matching official CUDA
wheels **instead of** the CPU wheel command:

```powershell
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

For SDK-only use, install `requirements-model.txt` instead of the complete backend
requirements. OpenCV is mandatory because the final SDK output includes connected
region filtering. `device="auto"` chooses CUDA when available, otherwise CPU.
CPU/GPU numerical differences are possible; the complete raw and filtered CPU
fixture is provided for local replay.

## Run the reference backend

The included backend preserves the old `/api/v1` video/task/frame interfaces.
It is optional if the teammate already has a backend. To run it from the package
root on Windows:

```powershell
.\start_backend.cmd
```

Use a single worker, no reload. Default address is `http://127.0.0.1:8000`.
`GET /api/v1/health` and `GET /api/v1/models` expose the model/rule identity.
Open `/docs`, or read `docs/openapi.json` without starting the server.

In another terminal with the same environment and package root:

```powershell
python examples/api_smoke.py
```

This uploads the included clearly synthetic clip, starts actual model inference,
checks filtered mask/candidate consistency and downloads frame assets and reports.
It is an integration test, not an accuracy or FPS benchmark.

For a single image:

```powershell
python examples/predict_image.py --image examples/synthetic_frame.png --output runtime_data/mask.png --device cpu
```

## Existing web project migration

Read `docs/WEB_INTEGRATION_GUIDE.md`. The main changes are:

- Change the request's `model_version` to the v2 ID above; the old ID is rejected.
- Replace old SDK architecture/weights as one package, not one file.
- Use the filtered mask and candidate flag; clear the old overlay on every new
  frame, including empty masks. Do not regenerate a mask from raw probabilities.
- Update inference metadata to input320/rotate4 and mean-score filtering.
- Use a fresh storage directory/new analysis tasks. Old saved masks do not update
  when the model changes. Remove any old nonzero `ENDO_MIN_REGION_AREA_RATIO`.

`frontend_examples/` contains mergeable endpoint and native-coordinate canvas
examples. It is not a replacement frontend application.

## Frozen evaluation record

E14 reused C6 exploration:83 positive images,1,760 verified negative images,
input320/rotate4/native masks. The **confidence-only0.8** row is included in
`validation_results.json`:

| Metric | Unfiltered E13 | Released rule |
|---|---:|---:|
| Positive mean Dice | 0.85882717 | 0.85607267 |
| Positive mean IoU | 0.79539512 | 0.79299139 |
| Negative frames with nonempty mask | 542/1760 (30.80%) | 369/1760 (20.97%) |
| Positive Dice=0 | 3 | 3 |

These are existing exploratory research results, not new independent validation
from exporting the model. The user selected this as the final project handover.
There is no claim that every normal frame becomes empty or that the region score
is a calibrated probability of disease. Candidate counts count frames, not unique
polyps. Real video throughput must be measured; rotate4 performs four forward passes.

## Contents

- `weights/`, `model.json`: complete exported segmentation weights and identities.
- `endolens_model/`: portable offline SDK, architecture and fixed output rule.
- `backend/`: optional recorded-video FastAPI reference server.
- `examples/`: synthetic frame/video, actual CPU outputs, CLI and HTTP smoke test.
- `tests/`, `verification/`: regression tests and export/evaluation parity evidence.
- `docs/`: migration, model card and exported OpenAPI.
- `MANIFEST.json`, `verify_bundle.py`: file integrity and CPU fixture replay.

Reference server limits remain100MiB/H264 MP4, up to1080p/15min, at most600 sampled
frames per task and roughly2GiB storage. Default sampling is one frame per second.
Use a longer interval or shorter clip when needed. No temporal tracking is enabled.
The reference server is a local team prototype without authentication; public
hosting/security controls belong to the web deployment project. See the model card
and source notices for research scope and data provenance.
