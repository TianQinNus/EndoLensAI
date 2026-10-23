import csv
import io
import json

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from ...schemas.requests import ExportInput
from ..dependencies import get_frame_rows, get_record, get_runtime, identity
from .analyses import build_report

router = APIRouter()


@router.post("/analyses/{task_id}/exports")
def export(task_id: str, body: ExportInput, request: Request):
    runtime = get_runtime(request)
    summary = build_report(runtime, task_id)
    rows = get_frame_rows(runtime, task_id)
    if not body.include_reviewed_masks:
        summary.pop("human_reviews", None)
    export_id = identity()
    if body.format == "json":
        data = json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    else:
        text = io.StringIO()
        keys = ["frame_id", "frame_index", "timestamp_seconds", "status", "is_candidate", "max_region_area_ratio", "foreground_score"]
        if body.include_reviewed_masks:
            keys += ["review_status", "revision_id", "edited_mask_url"]
        writer = csv.DictWriter(text, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)
        data = text.getvalue().encode("utf-8-sig")
    relative = f"exports/{export_id}.{body.format}"
    runtime.store.write(relative, data)
    runtime.store.put("export", export_id, {"path": relative, "format": body.format})
    return {
        "export_id": export_id,
        "status": "completed",
        "download_url": f"/api/v1/exports/{export_id}/download",
        "reviewed_masks": "references_only_not_binary_bundle" if body.include_reviewed_masks else "excluded",
    }


@router.get("/exports/{export_id}/download")
def download(export_id: str, request: Request):
    runtime = get_runtime(request)
    row = get_record(runtime, "export", export_id)
    return FileResponse(runtime.store.path(row["path"]), filename=f"endolens-{export_id}.{row['format']}")
