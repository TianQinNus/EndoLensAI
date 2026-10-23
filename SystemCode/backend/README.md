# EndoLens video-analysis backend

This package integrates the frozen `unet-b-expanded-seed42-r98-v1` model and
the recorded-video FastAPI service into the main EndoLensAI project.

The backend follows the formal boundaries in the technical specification:
`core` contains configuration, `schemas` contains Pydantic contracts,
`repositories` owns persistence adapters, `services` owns video and inference
logic, `workers` owns task lifecycle adapters, and `api/v1` is the versioned
route registration boundary. `main.py` is the application entrypoint; `app.py`
remains a compatibility import for existing commands.

It is a research/teaching integration service, not a clinical or public-facing
deployment. The service supports H264 MP4 uploads, PTS-based sampling,
asynchronous frame inference, AI masks, human mask review, reports, and CSV or
JSON metadata exports.

## Run

From the repository root, after installing the dependencies in this directory,
run the one-time configuration below. Uvicorn reads `UVICORN_HOST` and
`UVICORN_PORT` before loading the application, so these two settings cannot be
overridden by FastAPI code after startup.

```bash
conda env config vars set -n endolens \
  UVICORN_HOST=127.0.0.1 \
  UVICORN_PORT=8000 \
  ENDO_DEVICE=auto \
  ENDO_STORAGE_DIR=runtime_data \
  ENDO_MIN_REGION_AREA_RATIO=0 \
  ENDO_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000
```

Close and reopen the terminal after changing conda environment variables.
After that, the normal startup command is:

```bash
cd SystemCode
conda activate endolens
uvicorn backend.main:app --workers 1
```

For a LAN-only development server, set `UVICORN_HOST=0.0.0.0` once instead.
Do not expose this research service directly to the public internet.

The complete project-local template is
[`backend/.env.example`](.env.example). The backend only accepts explicit CORS
origins; wildcard `*` is rejected to prevent accidental cross-origin exposure.
The backend must run with one Uvicorn worker and one process per storage
directory.

## Test

```bash
cd SystemCode
python -m unittest discover -s backend/tests -p 'test*.py' -v
```

The CPU model tests use the committed model contract and weight asset. The
weight file is ignored by the repository's existing large-artifact policy, so
deployment must provision `weights/unet_b_expanded_seed42.pt` alongside this
package.
