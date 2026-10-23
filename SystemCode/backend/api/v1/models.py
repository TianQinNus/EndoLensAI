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
            "validation_status": "engineering_demo_unvalidated_area_gate",
        },
    }
