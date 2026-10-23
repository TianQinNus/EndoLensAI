"""Aggregated version-one API router."""

from fastapi import APIRouter

from .analyses import router as analyses_router
from .exports import router as exports_router
from .frames import router as frames_router
from .health import router as health_router
from .models import router as models_router
from .videos import router as videos_router

API_PREFIX = "/api/v1"
router = APIRouter()

router.include_router(health_router)
router.include_router(models_router)
router.include_router(videos_router)
router.include_router(analyses_router)
router.include_router(frames_router)
router.include_router(exports_router)
