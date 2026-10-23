import io
import re
from typing import Literal

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from ..dependencies import get_frame_rows, get_record, get_runtime, identity, image_bytes, now

router = APIRouter()


@router.get("/analyses/{task_id}/frames")
def frames(task_id: str, request: Request, after: str | None = None, limit: int = Query(100, ge=1, le=200), candidate_only: bool = False):
    rows = get_frame_rows(get_runtime(request), task_id)
    cursor = -1
    if after is not None:
        if not re.fullmatch(r"\d{1,10}", after):
            raise HTTPException(422, "after must be a frame_index cursor")
        cursor = int(after)
    rows = [row for row in rows if row["frame_index"] > cursor and (not candidate_only or row["is_candidate"] is True)]
    items = rows[:limit]
    return {
        "task_id": task_id,
        "items": items,
        "next_cursor": str(items[-1]["frame_index"]) if len(rows) > limit else None,
        "last_cursor": str(items[-1]["frame_index"]) if items else after,
        "has_more": len(rows) > limit,
    }


@router.get("/analyses/{task_id}/keyframes")
def keyframes(task_id: str, request: Request):
    rows = [row for row in get_frame_rows(get_runtime(request), task_id) if row["status"] == "completed"]
    indexes = np.linspace(0, len(rows) - 1, min(60, len(rows)), dtype=int) if rows else []
    return {
        "task_id": task_id,
        "keyframes": [
            {key: rows[index][key] for key in ("frame_id", "timestamp_seconds", "thumbnail_url", "is_candidate")}
            for index in indexes
        ],
    }


@router.get("/assets/{kind}/{name}")
def asset(kind: Literal["frames", "thumbs", "masks", "reviews"], name: str, request: Request):
    pattern = r"[a-f0-9]{32}\.png" if kind == "reviews" else r"[a-f0-9]{32}-f\d{8}\.(png|jpg)"
    if not re.fullmatch(pattern, name):
        raise HTTPException(404, "Asset not found")
    path = get_runtime(request).store.path(f"{kind}/{name}")
    if not path.is_file():
        raise HTTPException(404, "Asset not found")
    return FileResponse(path, media_type="image/jpeg" if name.endswith(".jpg") else "image/png")


@router.put("/frames/{frame_id}/review")
async def review(frame_id: str, request: Request, edited_mask: UploadFile = File(...), base_revision_id: str | None = Form(None), review_status: Literal["reviewed", "rejected"] = Form("reviewed"), notes: str = Form("")):
    runtime = get_runtime(request)
    content = await edited_mask.read(16 * 1024**2 + 1)
    await edited_mask.close()
    if len(content) > 16 * 1024**2 or len(notes) > 2000:
        raise HTTPException(413, "Review size limit")
    frame = get_record(runtime, "frame", frame_id)
    if frame["status"] != "completed":
        raise HTTPException(409, "Cannot review failed frame")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "PNG" or image.mode not in ("1", "L") or image.size != (frame["width"], frame["height"]):
                raise ValueError("Require native-resolution single-channel binary PNG")
            mask = np.asarray(image)
            if not np.isin(mask, (0, 1, 255)).all():
                raise ValueError("Mask must contain only0/1 or0/255")
            encoded = image_bytes(Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L"))
    except Exception as error:
        raise HTTPException(422, str(error)) from error
    with runtime.store.lock:
        frame = get_record(runtime, "frame", frame_id)
        if (base_revision_id or None) != frame["revision_id"]:
            raise HTTPException(409, "Revision conflict; reload frame")
        revision = identity()
        runtime.store.write(f"reviews/{revision}.png", encoded)
        row = {
            "frame_id": frame_id,
            "revision_id": revision,
            "review_status": review_status,
            "edited_mask_url": f"/api/v1/assets/reviews/{revision}.png",
            "notes": notes,
            "saved_at": now(),
            "base_revision_id": base_revision_id or None,
        }
        runtime.store.put("review", revision, row, frame["task_id"], frame["frame_index"])
        frame.update(review_status=review_status, revision_id=revision, edited_mask_url=row["edited_mask_url"])
        runtime.store.put("frame", frame_id, frame, frame["task_id"], frame["frame_index"])
    return row
