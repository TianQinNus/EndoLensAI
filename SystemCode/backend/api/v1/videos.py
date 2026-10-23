import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from ...core.config import MAX_UPLOAD
from ...services.video_service import VideoService
from ..dependencies import get_record, get_runtime, identity

router = APIRouter()


@router.post("/videos", status_code=201)
async def upload(request: Request, file: UploadFile = File(...)):
    runtime = get_runtime(request)
    try:
        if not (file.filename or "").lower().endswith(".mp4"):
            raise HTTPException(415, "Only H264 .mp4 accepted")
        content = await file.read(MAX_UPLOAD + 1)
        if not content or len(content) > MAX_UPLOAD:
            raise HTTPException(413, "Empty file or upload exceeds100MiB")
        video_id = identity()
        try:
            path = runtime.store.write(f"videos/{video_id}.mp4", content)
            metadata = VideoService.probe(path)
        except Exception as error:
            raise HTTPException(422, f"Video rejected:{error}") from error
        value = {
            "video_id": video_id,
            "filename": Path((file.filename or "").replace("\\", "/")).name[:120],
            "video_url": f"/api/v1/videos/{video_id}/stream",
            "stored_path": f"videos/{video_id}.mp4",
            **metadata,
        }
        runtime.store.put("video", video_id, value)
        return {key: value[key] for key in value if key != "stored_path"}
    finally:
        await file.close()


@router.get("/videos/{video_id}/stream")
def stream(video_id: str, request: Request):
    runtime = get_runtime(request)
    value = get_record(runtime, "video", video_id)
    byte_range = request.headers.get("range", "")
    if byte_range and not re.fullmatch(r"bytes=(\d{1,18}-\d{0,18}|-\d{1,18})", byte_range):
        raise HTTPException(416, "Only a single byte range is supported")
    return FileResponse(runtime.store.path(value["stored_path"]), media_type="video/mp4")
