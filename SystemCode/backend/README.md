# EndoLens AI modular backend

The current backend uses the final **E13 ResNet50 U-Net**, epoch 94, input 320,
four-rotation logit averaging and native connected-region mean score >=0.8.
Its model version is `e13-r50-joint-r94-r4-region080-v2`.

## Model location

Canonical source, configuration, installer and the external weight live in
[`../model/endolens_e13`](../model/endolens_e13/README.md).
`backend.endolens_model` forwards imports to that SDK. The backend does not load
the old ResNet34 model or a separate weight in `backend/weights`.
`model.json` in this folder is a compatibility pointer, not a second model config.

From the repository root, download and install the two Drive parts as described in
[README section 4.1](../../README.md#41-system-requirements-and-model-download).
Install Python 3.11, matching PyTorch/torchvision wheels and `requirements.txt`.

## Run

From the repository root with the backend environment active:

```powershell
cd SystemCode
$env:ENDO_DEVICE = "auto"
$env:ENDO_STORAGE_DIR = Join-Path (Get-Location) "runtime_data_e13"
$env:ENDO_MIN_REGION_AREA_RATIO = "0"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

On macOS/Linux, use `export NAME=value` for environment settings. CPU is supported.
One process and one Uvicorn worker must own a storage directory. Avoid reload.
Use a fresh storage directory and new tasks after upgrading the model: old masks
do not update automatically. The old B model version is rejected in new requests.

Check `/api/v1/health` and `/api/v1/models` for both frozen release identities.
API documentation is available at `/docs`, with a matching snapshot in `docs/openapi.json`.
`ENDO_CORS_ORIGINS` accepts explicit frontend origins, not wildcard `*`.
Uvicorn also accepts its normal `UVICORN_HOST`/`UVICORN_PORT` settings.
The `.env.example` is a configuration reference, not an automatically loaded file.

## Filtered output contract

The adapter returns the canonical native `binary_mask` and its associated
`postprocess_result`. The inference service uses that object directly.
For injected test adapters that return raw probabilities only, the service applies
the same frozen rule. Region means below 0.8 are deleted from masks, contours,
regions and candidate counts. An empty final mask is not a candidate.

`probability_map` is diagnostic. Do not threshold it again in the frontend.
No area gate or temporal confirmation is enabled. Nonzero
`ENDO_MIN_REGION_AREA_RATIO` is rejected. Region scores are not calibrated disease
probabilities and candidate counts are frames, not unique lesions.

The existing boundaries remain: `core`, `schemas`, `repositories`, `services`,
`workers` and `api/v1`. Uploads, PTS sampling, cancellation, immutable AI masks,
human revisions, reports and CSV/JSON exports retain their existing endpoints.

## Test

```powershell
cd SystemCode
python -m unittest discover -s backend/tests -p "test*.py" -v
```

Run this command from the repository root in a new terminal. The real CPU fixture
requires the installed E13 weight. Other tests use synthetic adapters to check
HTTP and storage contracts. They do not evaluate clinical performance or FPS.

The earlier [technical specification](docs/BACKEND_TECHNICAL_SPEC.md) records the
historical B integration. Current model details above, the OpenAPI snapshot and
[final release record](../model/docs/FINAL_MODEL_RELEASE.md) supersede its old
352-input, B-weight and candidate-area-only statements.
