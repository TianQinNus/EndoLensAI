"""Application factory and serial analysis runtime.

HTTP endpoints live in ``api.v1``. This module owns only process lifecycle,
model lifetime, and the single background inference consumer.
"""

import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.v1.router import API_PREFIX, router as api_router
from .core.config import (
    WARNINGS,
    area_gate,
    cors_headers,
    cors_methods,
    cors_origins,
    device as configured_device,
    storage_dir as configured_storage_dir,
)
from .endolens_model import ModelAdapter
from .services.inference_service import InferenceService, image_bytes
from .services.video_service import VideoService
from .storage import Store


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Runtime:
    def __init__(self, store, adapter, area_gate_value):
        self.store = store
        self.adapter = adapter
        self.area_gate = area_gate_value
        self.inference = InferenceService(adapter, area_gate_value)
        self.jobs = queue.Queue(maxsize=4)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.loop, name="endolens-serial-worker", daemon=True)
        self.thread.start()

    def cancelled(self, task_id: str) -> bool:
        return self.stopping.is_set() or self.store.get("analysis", task_id).get("cancel_requested", False)

    def loop(self):
        while not self.stopping.is_set():
            try:
                task_id = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.run(task_id)
            except Exception as error:
                self.store.update(task_id, status="failed", error=f"{type(error).__name__}:{error}", finished_at=now())
            finally:
                self.jobs.task_done()

    def run(self, task_id: str):
        current = self.store.get("analysis", task_id)
        if self.cancelled(task_id):
            self.store.update(task_id, status="cancelled", finished_at=now())
            return
        started = time.perf_counter()
        self.store.update(task_id, status="running", phase="indexing", started_at=now())
        config = current["configuration"]
        video = self.store.get("video", current["video_id"])
        path = self.store.path(video["stored_path"])
        targets = VideoService.schedule(
            path,
            video["timestamp_origin_seconds"],
            config["sampling_mode"],
            config["interval_seconds"],
            lambda: self.cancelled(task_id),
        )
        if targets is None:
            self.store.update(task_id, status="cancelled", finished_at=now())
            return
        if not targets:
            raise ValueError("No scheduled frames")
        self.store.update(task_id, phase="inference", target_frames=len(targets), source_frames_indexed=True)
        wanted = {row["frame_index"]: row["timestamp_seconds"] for row in targets}
        processed = failed = candidates = 0
        for index, seconds, frame in VideoService.decoded(path, video["timestamp_origin_seconds"]):
            if self.cancelled(task_id):
                self.store.update(task_id, status="cancelled", finished_at=now(), elapsed_seconds=time.perf_counter() - started)
                return
            if index not in wanted:
                continue
            if abs(seconds - wanted[index]) > 1e-6:
                raise ValueError("PTS changed between indexing and inference")
            frame_id = f"{task_id}-f{index:08d}"
            row = {
                "frame_id": frame_id,
                "task_id": task_id,
                "frame_index": index,
                "timestamp_seconds": seconds,
                "width": frame.width,
                "height": frame.height,
                "model_version": config["model_version"],
                "status": "failed",
                "is_candidate": None,
                "max_region_area_ratio": None,
                "source_frame_url": None,
                "thumbnail_url": None,
                "mask_url": None,
                "contours": [],
                "review_status": "unreviewed",
                "revision_id": None,
                "foreground_score": None,
            }
            try:
                rgb = frame.to_ndarray(format="rgb24")
                source = self.inference.source_image(rgb)
                self.store.write(f"frames/{frame_id}.png", image_bytes(source))
                self.store.write(f"thumbs/{frame_id}.jpg", image_bytes(self.inference.thumbnail(source), "JPEG"))
                row.update(
                    source_frame_url=f"/api/v1/assets/frames/{frame_id}.png",
                    thumbnail_url=f"/api/v1/assets/thumbs/{frame_id}.jpg",
                )
                result, post = self.inference.predict(rgb, config["model_version"])
                self.store.write(f"masks/{frame_id}.png", image_bytes(self.inference.mask_image(post)))
                row.update(
                    status="completed",
                    is_candidate=post.is_candidate,
                    max_region_area_ratio=post.max_region_area_ratio,
                    mask_url=f"/api/v1/assets/masks/{frame_id}.png",
                    contours=post.contours,
                    regions=post.regions,
                    component_count=post.component_count,
                    display_geometry_truncated=post.display_geometry_truncated,
                    foreground_score=post.foreground_score,
                    score_type="mean_foreground_score_of_largest_component",
                    segmented_area_ratio=post.segmented_area_ratio,
                    inference_time_ms=result.inference_time_ms,
                    device=result.device,
                )
                processed += 1
                candidates += int(post.is_candidate)
            except Exception as error:
                row["error"] = f"{type(error).__name__}:{error}"
                failed += 1
            self.store.put("frame", frame_id, row, task_id, index)
            self.store.update(
                task_id,
                processed_frames=processed,
                failed_frames=failed,
                candidate_frames=candidates,
                elapsed_seconds=time.perf_counter() - started,
                warnings=([*WARNINGS, "HAS_FAILED_FRAMES"] if failed else WARNINGS),
            )
        if processed + failed != len(targets):
            raise ValueError("Video truncated:scheduled frame missing during inference")
        self.store.update(
            task_id,
            status="completed" if processed else "failed",
            phase="done",
            finished_at=now(),
            error=None if processed else "ALL_FRAMES_FAILED",
            elapsed_seconds=time.perf_counter() - started,
        )

    def close(self):
        self.stopping.set()
        self.thread.join()


def create_app(storage_dir=None, adapter=None, device=None):
    area_gate_value = area_gate()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = Store(storage_dir or configured_storage_dir())
        try:
            selected = adapter or ModelAdapter(device=device or configured_device())
            app.state.runtime = Runtime(store, selected, area_gate_value)
            yield
        finally:
            if hasattr(app.state, "runtime"):
                app.state.runtime.close()
            store.close()

    app = FastAPI(title="EndoLens B integration starter", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_methods=cors_methods(),
        allow_headers=cors_headers(),
        expose_headers=["Accept-Ranges", "Content-Length", "Content-Range"],
        allow_credentials=False,
    )
    app.include_router(api_router, prefix=API_PREFIX)
    return app


app = create_app()
