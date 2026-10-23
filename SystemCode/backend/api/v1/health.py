from fastapi import APIRouter, Request

from ...endolens_model import MODEL_VERSION
from ..dependencies import get_runtime

router = APIRouter()


@router.get("/health")
def health(request: Request):
    runtime = get_runtime(request)
    description = runtime.adapter.describe()
    return {
        "status": "ready",
        "model_version": MODEL_VERSION,
        "device": description["device"],
        "clinical_use": False,
    }
