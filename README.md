# EndoLens AI

**Polyp segmentation and interactive colonoscopy video analysis**

EndoLens AI combines model development with timestamped video review. The React
frontend uploads a recording, starts an analysis task and displays native-size
segmentation masks, candidate frames, human review revisions and reports through
a versioned FastAPI backend. This repository is a research and teaching prototype.

**Current model, 31 October 2026:** E13 joint positive/negative ResNet50 U-Net,
epoch 94, with four-rotation inference and a native connected-region mean score
of at least **0.8**. This replaces ResNet34 B as the current deployment choice.
Polyp-PVT stays a frozen comparison model. The final model uses one network,
not multi-model fusion.

## 1. Project structure

```text
SystemCode/
  frontend/                    React / TypeScript / Vite application
  backend/                     Modular FastAPI application and regression tests
  model/
    endolens_e13/              Current portable SDK, model identity and weight installer
    endolens_b/                Historical ResNet34 B baseline, preserved separately
    docs/                     Research protocols and final release record
    e0_mac/, e1_*/, e2_*/, ... Historical experiments, not the web startup path
ProjectReport/                Earlier experiment reports and research roadmap
Miscellaneous/
  model_results/2026-10-31/    Compact matched metrics and model timing evidence
  model_artifacts/            Historical split/protocol/source registries
Video/                        Demonstration guidance
```

The canonical current SDK lives in
[`SystemCode/model/endolens_e13`](SystemCode/model/endolens_e13/README.md).
`backend.endolens_model` forwards imports to that SDK, so the application and
standalone model use the same architecture, weight and output rule.

## 2. Model development and final selection

Pretrained residual encoders improved the U-Net development trajectory. Additional
PolypGen positive images improved the ResNet34 baseline. Fusion experiments could
improve some averages but did not satisfy all declared safeguards. Whole-frame
negative examples then addressed the false foreground observed during video use.

E13 starts from an ImageNet ResNet50 V2 encoder and a random GroupNorm U-Net
decoder. It samples 50% positive and 50% negative images from epoch one. The
training pool contains 2,269 positives and 1,907 negatives. The decoder uses
bilinear upsampling, skip concatenation and two convolution/GroupNorm/ReLU
operations per block. Encoder and decoder train together, with encoder BatchNorm
running statistics fixed. There are 32,520,977 parameters.

The selected epoch-94 checkpoint is the best-positive fallback. **No E13 checkpoint
passed all original positive-protection guards.** The team explicitly selected a
configuration with fewer negative-frame false positives at a positive-Dice cost.
See the [complete final release record](SystemCode/model/docs/FINAL_MODEL_RELEASE.md)
for the recipe, data accounting, selection limitations and timing boundary.

## 3. Matched evaluation results

All rows below use 320 input, FP32, four-rotation logit averaging and native-size
scoring on **83 C6 positive images and 1,760 negatives**. Dice and IoU average
positive images only. Negative-frame FPR counts negative images with any retained
foreground. Training histories differ, so this is not an architecture-only ablation.

| Configuration | Positive Dice | Positive IoU | Negative-frame FPR | Zero-Dice positives |
|---|---:|---:|---:|---:|
| Frozen Polyp-PVT | 0.84357 | 0.78978 | 66.19% | 4 |
| Frozen ResNet34 B | 0.84647 | 0.78440 | 56.02% | 3 |
| Positive-only ResNet50 | **0.88389** | **0.82727** | 89.72% | 2 |
| E13, before region filtering | 0.85883 | 0.79540 | 30.80% | 3 |
| **Final E13, region mean >=0.8** | **0.85607** | **0.79299** | **20.97%** | 3 |

The region rule removes predictions from 173 additional negative frames, reducing
FPR from 542/1,760 to 369/1,760. Positive Dice decreases by 0.00275. The final
model still has residual false alarms and seven positives with Dice below 0.5.
Positive-only comparators were not tested at their own best region-filtered
operating points. C6 was reused, including postprocessing selection, so these
results describe development performance rather than an untouched test.

[Machine-readable metrics](Miscellaneous/model_results/2026-10-31/model_comparison.csv)
and [metric definitions and limitations](SystemCode/model/docs/FINAL_MODEL_RELEASE.md)
are included. Earlier 352-input and historical 798-image results remain in their
original reports and must not be substituted for these matched rows.

## 4. Installation and startup

### 4.1 System requirements and model download

The walkthrough targets **Windows 10/11 with PowerShell**. On macOS/Linux, use the
same Python commands and normal shell environment-variable syntax. The final SDK
supports CPU and NVIDIA CUDA. An Apple MPS backend is not part of this release.

- Git and access to this repository.
- Conda with Python **3.11** for the backend.
- Node.js **24.x** and npm for the current frontend. Vite 8 requires Node.js
  **20.19+ or 22.12+** and fails on older versions.
- The external deployment weight **`e13_resnet50_joint_seed42_epoch94.pt`**,
  130,413,723 bytes, approximately **124.37 MiB**. It is not stored in Git.
- Optional: an NVIDIA GPU with a driver compatible with the PyTorch CUDA 12.8
  wheels. Without it, `ENDO_DEVICE=auto` selects CPU, which is slower.

**[Download the final E13 weights from Google Drive](https://drive.google.com/drive/folders/12nddUWrat2Ih4sHpa4c8M-DuO76tDEKO).**

Download both files into the same folder, preserving their names:

| File | Size | Download |
|---|---:|---|
| `e13_resnet50_joint_seed42_epoch94.pt.part01` | 67,108,864 bytes | [Part 1](https://drive.google.com/file/d/1pw_dTRumCR9gzV01QwTre7jnBrnS1EQF/view?usp=drivesdk) |
| `e13_resnet50_joint_seed42_epoch94.pt.part02` | 63,304,859 bytes | [Part 2](https://drive.google.com/file/d/1jK8p2FNagEAzYvKJ5aNW9XcCnKJ6BESa/view?usp=drivesdk) |

This is a **lossless split** for the upload service's 100 MiB file limit. The
installer checks both part hashes, combines them in order and checks the complete
weight. It does not quantise, retrain or otherwise convert the model. Do not unzip
the parts, load an individual part with PyTorch, or use the old ResNet34 weight.

```powershell
git clone --branch master https://github.com/TianQinNus/EndoLensAI.git
cd EndoLensAI
python SystemCode/model/endolens_e13/install_weights.py --parts-dir "$HOME/Downloads"
```

If you already have the complete original file, install it with:

```powershell
python SystemCode/model/endolens_e13/install_weights.py --source "C:/path/to/e13_resnet50_joint_seed42_epoch94.pt"
```

The installed location is
`SystemCode/model/endolens_e13/weights/e13_resnet50_joint_seed42_epoch94.pt`.
SHA256 of the complete file:

```text
dc6c9092baf68cf9e66264f632fae76642ff621282aaa7f65dbf8a899f168e68
```

### 4.2 Backend environment

From the repository root:

```powershell
conda create -n endolens-web python=3.11 -y
conda activate endolens-web
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r SystemCode/backend/requirements.txt
python SystemCode/model/endolens_e13/install_weights.py --check
```

For a compatible NVIDIA setup, use the CUDA command **instead of** the CPU wheel
command above:

```powershell
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

Load the complete `.pt` only through the matching E13 architecture. Dependency
installation needs internet access, but loading this SDK does not download
ImageNet weights or contact an external model service.

### 4.3 Start the backend

```powershell
cd SystemCode
$env:ENDO_DEVICE = "auto"
$env:ENDO_STORAGE_DIR = Join-Path (Get-Location) "runtime_data_e13"
$env:ENDO_MIN_REGION_AREA_RATIO = "0"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Use one worker and one process per storage directory. Do not use auto-reload for
inference. Keep the area setting at zero: the final model uses a region-score
rule, not an additional area filter. A new storage directory prevents old B masks
from being mistaken for new E13 results. Model changes do not recompute old tasks.

- API health: `http://127.0.0.1:8000/api/v1/health`
- Model identity and output rule: `http://127.0.0.1:8000/api/v1/models`
- Interactive API reference: `http://127.0.0.1:8000/docs`

`/health` must report `e13-r50-joint-r94-r4-region080-v2` and
`native-cc8-pixel050-mean080-v1`. The backend uses explicit CORS origins, configured
through `ENDO_CORS_ORIGINS`. See the [backend guide](SystemCode/backend/README.md).

### 4.4 Start the frontend

Open a second terminal from the repository root:

```powershell
cd SystemCode/frontend
npm ci
if (!(Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm run dev
```

Keep an existing `.env.local` if you already configured it. Its API setting is
`VITE_API_BASE_URL=http://127.0.0.1:8000/api/v1`. Open the local URL printed by Vite.
The frontend analysis request now uses the E13 version. The deployment changes
preserve the existing interface and endpoint structure.

### 4.5 Verify the SDK and application contracts

From the repository root with the backend environment active:

```powershell
cd SystemCode/model/endolens_e13
python verify_bundle.py
python run_tests.py
```

In a separate terminal from the repository root:

```powershell
cd SystemCode
python -m unittest discover -s backend/tests -p "test*.py" -v
```

The model fixture uses real CPU inference on a clearly synthetic image. API tests
also use injected test adapters to check masks, candidate counts, failures,
cancellation, reviews and exports. They are regression tests, not a clinical
accuracy or throughput benchmark.

## 5. The released input/output contract

1. Accept native `RGB uint8[H,W,3]`. Convert OpenCV BGR to RGB explicitly.
2. Pillow bilinear resize to 320 x 320, divide by 255. The network applies
   ImageNet normalisation internally, exactly once.
3. Predict 0/90/180/270-degree rotations using one model, inverse-align and
   average **logits**.
4. Bilinearly restore logits to the original dimensions, then apply sigmoid once.
5. Threshold pixels at 0.5 and find **8-connected components at native size**.
6. Retain a component only if its unrounded mean foreground score is >=0.8.
7. Use the filtered `binary_mask` for storage, overlays, contours and candidate
   status. An empty mask means `is_candidate=false`.

`probability_map` is a raw diagnostic output. Re-thresholding it on the frontend
bypasses the chosen rule. The release uses no largest-only rule, minimum-area
filter, temporal confirmation or separate classification head. Region scores are
uncalibrated model outputs, not disease probabilities. See the
[SDK integration guide](SystemCode/model/endolens_e13/docs/WEB_INTEGRATION_GUIDE.md).

## 6. Data, evaluation and runtime limits

| Partition | Positive | Negative | Total |
|---|---:|---:|---:|
| Training | 2,269 | 1,907 | 4,176 |
| Validation | 494 | 733 | 1,227 |
| C6 evaluation | 83 | 1,760 | 1,843 |

Positive training uses 1,151 original images and 1,118 PolypGen C1-C4 images.
Validation combines 290 original positives with 204 C5 positives. Negative
sequences 1-10 train, 11-13 validate and 14-23 evaluate. Verified empty single-frame
annotations follow centre assignments. Missing labels alone never define negatives.

Each E13 epoch draws 1,160 samples **with replacement**. Positive/negative draw
probabilities are 50/50. Within positives, the original source receives 50% and
each additional centre 12.5%. This is not one traversal of the entire pool.

Historical 798-image, FAR, C5 and C6 data have been inspected during development.
C6 also informed the final rule. No unseen final-test or clinical-validation claim
is made. Candidate counts are frames, not distinct polyps. Unprocessed or failed
frames are unknown rather than confirmed normal. Human revisions stay separate
from immutable original AI statistics.

On the controlled RTX 5060 Laptop GPU benchmark, mean model forward time was
18.41 ms, four-view network time 67.25 ms and full frame-SDK time 114.75 ms.
The SDK p95 was 136.76 ms. These batch-one FP32 measurements exclude video decoding,
HTTP, task queues, persistence and browser display. They do not establish live
application FPS. See [the exact boundary](SystemCode/model/docs/FINAL_MODEL_RELEASE.md).

## 7. Research branches, reports and attribution

- `master`: integrated application and current E13 release.
- `model_exploration`: historical E5-E11 implementations, protocols and research
  provenance. This release does not rewrite those executed source snapshots.
- `frontend` and `backend`: teammate development branches. Current code in master
  may differ from these branch snapshots.

Earlier reports remain in [ProjectReport](ProjectReport/README.md). The
[final release record](SystemCode/model/docs/FINAL_MODEL_RELEASE.md) supersedes
old statements that B is the current deployment model. E11 remains incomplete.

Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the
[model notices](SystemCode/model/endolens_e13/THIRD_PARTY_NOTICES.md). Existing
architectures, datasets and upstream assets retain their original terms. No
blanket commercial or clinical licence is granted. Do not commit raw datasets,
clinical videos, credentials or downloaded E13 weights. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the historical source-preservation rules.
