import time

from fastapi import APIRouter, HTTPException, Request

from ...core.config import WARNINGS
from ...endolens_model import MODEL_VERSION
from ...schemas.requests import AnalysisInput
from ..dependencies import get_frame_rows, get_record, get_runtime, now, identity

router = APIRouter()


@router.post("/analyses", status_code=202)
def analyze(body: AnalysisInput, request: Request):
    runtime = get_runtime(request)
    get_record(runtime, "video", body.video_id)
    task_id = identity()
    description = runtime.adapter.describe()
    config = {
        **body.model_dump(),
        "segmentation_threshold": 0.5,
        "min_region_area_ratio": runtime.area_gate,
        "candidate_rule_status": "UNVALIDATED_ENGINEERING_DEMO",
        "source_checkpoint_sha256": description["source_checkpoint_sha256"],
        "model_weights_sha256": description.get("weights_sha256", "TEST_ONLY"),
        "model_device": description["device"],
        "preprocessing": description.get("input", {}),
        "postprocessing_version": "candidate-area-demo-v1",
    }
    value = {
        "task_id": task_id,
        "video_id": body.video_id,
        "status": "queued",
        "configuration": config,
        "created_at": now(),
        "target_frames": None,
        "processed_frames": 0,
        "failed_frames": 0,
        "candidate_frames": 0,
        "elapsed_seconds": 0.0,
        "cancel_requested": False,
        "phase": "queued",
        "warnings": WARNINGS,
    }
    with runtime.store.lock:
        if runtime.jobs.full():
            raise HTTPException(429, "Queue full; wait for an existing task")
        runtime.store.put("analysis", task_id, value)
        runtime.jobs.put_nowait(task_id)
    return {key: value[key] for key in ("task_id", "video_id", "status", "configuration", "created_at")}


@router.get("/analyses/{task_id}")
def status(task_id: str, request: Request):
    runtime = get_runtime(request)
    row = get_record(runtime, "analysis", task_id)
    saved = get_frame_rows(runtime, task_id)
    row["processed_frames"] = sum(item["status"] == "completed" for item in saved)
    row["failed_frames"] = sum(item["status"] == "failed" for item in saved)
    row["candidate_frames"] = sum(item["is_candidate"] is True for item in saved)
    total = row["target_frames"]
    attempted = len(saved)
    return {
        **row,
        "progress": {
            "target_frames": total,
            "processed_frames": row["processed_frames"],
            "candidate_frames": row["candidate_frames"],
            "failed_frames": row["failed_frames"],
            "percent": 100 * attempted / total if total else None,
        },
        "performance": {
            "elapsed_seconds": row["elapsed_seconds"],
            "processing_fps": row["processed_frames"] / row["elapsed_seconds"] if row["elapsed_seconds"] else 0.0,
        },
    }


@router.post("/analyses/{task_id}/cancel", status_code=202)
def cancel(task_id: str, request: Request):
    runtime = get_runtime(request)
    row = get_record(runtime, "analysis", task_id)
    if row["status"] in ("queued", "running"):
        row = runtime.store.update(task_id, cancel_requested=True)
    return {"task_id": task_id, "status": row["status"], "cancel_requested": row["cancel_requested"]}


def build_report(runtime, task_id: str):
    task = get_record(runtime, "analysis", task_id)
    rows = get_frame_rows(runtime, task_id)
    good = [row for row in rows if row["status"] == "completed"]
    candidates = [row for row in good if row["is_candidate"]]
    import numpy as np

    return {
        "task_id": task_id,
        "video_id": task["video_id"],
        "status": task["status"],
        "model_version": MODEL_VERSION,
        "configuration": task["configuration"],
        "video": {key: value for key, value in get_record(runtime, "video", task["video_id"]).items() if key != "stored_path"},
        "sampling": {
            "mode": task["configuration"]["sampling_mode"],
            "interval_seconds": task["configuration"]["interval_seconds"],
        },
        "summary": {
            "analyzed_frames": len(good),
            "candidate_frames": len(candidates),
            "failed_frames": len(rows) - len(good),
            "candidate_frame_ratio": len(candidates) / len(good) if good else None,
            "reviewed_frames": sum(row["review_status"] != "unreviewed" for row in rows),
            "processing_time_seconds": task["elapsed_seconds"],
            "mean_segmented_area_ratio": float(np.mean([row["segmented_area_ratio"] for row in good])) if good else None,
        },
        "candidate_timestamps_seconds": [row["timestamp_seconds"] for row in candidates],
        "representative_frames": good[:8],
        "human_reviews": runtime.store.list("review", task_id),
        "statistics_basis": "original_AI_masks; reviews are separate, never silently overwrite AI statistics",
        "warnings": task["warnings"],
        "generated_at": now(),
    }


@router.get("/analyses/{task_id}/report")
def report(task_id: str, request: Request):
    return build_report(get_runtime(request), task_id)
