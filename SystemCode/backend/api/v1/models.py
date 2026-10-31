from fastapi import APIRouter, Request

from ...core.config import WARNINGS
from ..dependencies import get_runtime

router = APIRouter()


@router.get("/models")
def models(request: Request):
    runtime = get_runtime(request)
    return {
        "items": [runtime.adapter.describe()],
        "warnings": WARNINGS,
        "postprocessing": {
            "segmentation_threshold": 0.5,
            "min_region_area_ratio": runtime.area_gate,
            "min_region_mean_score": 0.8,
            "mask_filtering": True,
            "connectivity": 8,
            "temporal_confirmation": False,
            "validation_status": "C6_DEVELOPMENT_REUSED_NOT_INDEPENDENT",
        },
    }
