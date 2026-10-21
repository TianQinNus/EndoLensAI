"""Local recorded-video integration starter. One process/one worker; no auth."""
import csv
import io
import json
import math
import os
import queue
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from endolens_model import ModelAdapter, MODEL_VERSION
from endolens_model.postprocess import postprocess
from .storage import Store
from .video import decoded, probe, schedule

ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD = 100*1024**2
WARNINGS = ["RESEARCH_DEMO_NOT_CLINICAL", "POSITIVE_ONLY_MODEL:NO_NEGATIVE_FRAME_SPECIFICITY_CLAIM",
            "AREA_GATE_UNVALIDATED_ENGINEERING_RULE", "CANDIDATE_COUNT_IS_FRAMES_NOT_DISTINCT_POLYPS",
            "FOREGROUND_SCORE_NOT_CALIBRATED_CLINICAL_PROBABILITY"]


def now():
    return datetime.now(timezone.utc).isoformat()


def identity():
    return uuid.uuid4().hex


def image_bytes(image, format="PNG"):
    buffer = io.BytesIO(); image.save(buffer, format=format)
    return buffer.getvalue()


class AnalysisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    sampling_mode: Literal["interval", "every_frame"] = "interval"
    interval_seconds: float | None = Field(default=1., ge=.1, le=900, allow_inf_nan=False)
    model_version: Literal[MODEL_VERSION] = MODEL_VERSION

    @model_validator(mode="after")
    def consistent(self):
        if self.sampling_mode == "interval" and self.interval_seconds is None:
            raise ValueError("interval_seconds required for interval mode")
        if self.sampling_mode == "every_frame":
            self.interval_seconds = None
        return self


class ExportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["csv", "json"] = "csv"
    include_reviewed_masks: bool = True


class Runtime:
    def __init__(self, store, adapter, area_gate):
        self.store, self.adapter, self.area_gate = store, adapter, area_gate
        self.jobs, self.stopping = queue.Queue(maxsize=4), threading.Event()
        self.thread = threading.Thread(target=self.loop, name="endolens-serial-worker", daemon=True)
        self.thread.start()

    def cancelled(self, task):
        return self.stopping.is_set() or self.store.get("analysis", task).get("cancel_requested",False)

    def loop(self):
        while not self.stopping.is_set():
            try:
                task = self.jobs.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                self.run(task)
            except Exception as error:
                self.store.update(task, status="failed", error=f"{type(error).__name__}:{error}", finished_at=now())
            finally:
                self.jobs.task_done()

    def run(self, task):
        current = self.store.get("analysis", task)
        if self.cancelled(task):
            self.store.update(task,status="cancelled",finished_at=now()); return
        started = time.perf_counter()
        self.store.update(task,status="running",phase="indexing",started_at=now())
        config = current["configuration"]
        video = self.store.get("video", current["video_id"])
        path = self.store.path(video["stored_path"])
        targets = schedule(path,video["timestamp_origin_seconds"],config["sampling_mode"],config["interval_seconds"],lambda:self.cancelled(task))
        if targets is None:
            self.store.update(task,status="cancelled",finished_at=now()); return
        if not targets:
            raise ValueError("No scheduled frames")
        self.store.update(task,phase="inference",target_frames=len(targets),source_frames_indexed=True)
        wanted = {row["frame_index"]:row["timestamp_seconds"] for row in targets}
        processed = failed = candidates = 0
        for index, seconds, frame in decoded(path,video["timestamp_origin_seconds"]):
            if self.cancelled(task):
                self.store.update(task,status="cancelled",finished_at=now(),elapsed_seconds=time.perf_counter()-started); return
            if index not in wanted:
                continue
            if abs(seconds-wanted[index]) > 1e-6:
                raise ValueError("PTS changed between indexing and inference")
            fid = f"{task}-f{index:08d}"
            row = {"frame_id":fid,"task_id":task,"frame_index":index,"timestamp_seconds":seconds,
                   "width":frame.width,"height":frame.height,"model_version":MODEL_VERSION,
                   "status":"failed","is_candidate":None,"max_region_area_ratio":None,
                   "source_frame_url":None,"thumbnail_url":None,"mask_url":None,"contours":[],
                   "review_status":"unreviewed","revision_id":None,"foreground_score":None}
            try:
                rgb = frame.to_ndarray(format="rgb24")
                source = Image.fromarray(rgb)
                self.store.write(f"frames/{fid}.png",image_bytes(source))
                thumbnail=source.copy();thumbnail.thumbnail((320,180),Image.Resampling.BILINEAR)
                self.store.write(f"thumbs/{fid}.jpg",image_bytes(thumbnail,"JPEG"))
                row.update(source_frame_url=f"/api/v1/assets/frames/{fid}.png",thumbnail_url=f"/api/v1/assets/thumbs/{fid}.jpg")
                result=self.adapter.predict(rgb,config["model_version"])
                post=postprocess(result.probability_map,config["segmentation_threshold"],config["min_region_area_ratio"])
                self.store.write(f"masks/{fid}.png",image_bytes(Image.fromarray(post.binary_mask*255,mode="L")))
                row.update(status="completed",is_candidate=post.is_candidate,max_region_area_ratio=post.max_region_area_ratio,
                           mask_url=f"/api/v1/assets/masks/{fid}.png",contours=post.contours,regions=post.regions,
                           component_count=post.component_count,display_geometry_truncated=post.display_geometry_truncated,
                           foreground_score=post.foreground_score,score_type="mean_foreground_score_of_largest_component",
                           segmented_area_ratio=post.segmented_area_ratio,inference_time_ms=result.inference_time_ms,
                           device=result.device)
                processed+=1;candidates+=int(post.is_candidate)
            except Exception as error:
                row.update(error=f"{type(error).__name__}:{error}")
                failed+=1
            self.store.put("frame",fid,row,task,index)
            self.store.update(task,processed_frames=processed,failed_frames=failed,candidate_frames=candidates,
                              elapsed_seconds=time.perf_counter()-started,warnings=([*WARNINGS,"HAS_FAILED_FRAMES"] if failed else WARNINGS))
        if processed+failed != len(targets):
            raise ValueError("Video truncated:scheduled frame missing during inference")
        self.store.update(task,status="completed" if processed else "failed",phase="done",finished_at=now(),
                          error=None if processed else "ALL_FRAMES_FAILED",elapsed_seconds=time.perf_counter()-started)

    def close(self):
        self.stopping.set();self.thread.join()


def create_app(storage_dir=None, adapter=None, device=None):
    area_gate = float(os.getenv("ENDO_MIN_REGION_AREA_RATIO","0"))
    if not math.isfinite(area_gate) or not 0 <= area_gate <= 1:
        raise ValueError("Invalid ENDO_MIN_REGION_AREA_RATIO")
    @asynccontextmanager
    async def lifespan(app):
        store = Store(storage_dir or os.getenv("ENDO_STORAGE_DIR",str(ROOT/"runtime_data")))
        try:
            selected = adapter or ModelAdapter(device=device or os.getenv("ENDO_DEVICE","auto"))
            app.state.runtime = Runtime(store,selected,area_gate)
            yield
        finally:
            if hasattr(app.state,"runtime"):
                app.state.runtime.close()
            store.close()
    app = FastAPI(title="EndoLens B integration starter",version="1.0.0",lifespan=lifespan)
    origins=os.getenv("ENDO_CORS_ORIGINS","http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000").split(",")
    app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=["GET","POST","PUT"],allow_headers=["Content-Type","Range"],expose_headers=["Content-Range","Accept-Ranges"],allow_credentials=False)

    def rt():return app.state.runtime
    def record(kind, key):
        try:return rt().store.get(kind,key)
        except KeyError:raise HTTPException(404,"Record not found")
    def frame_rows(task):
        record("analysis",task)
        return rt().store.list("frame",task)

    @app.get("/api/v1/health")
    def health():
        description=rt().adapter.describe()
        return {"status":"ready","model_version":MODEL_VERSION,"device":description["device"],"clinical_use":False}

    @app.get("/api/v1/models")
    def models():
        return {"items":[rt().adapter.describe()],"warnings":WARNINGS,
                "postprocessing":{"segmentation_threshold":.5,"min_region_area_ratio":area_gate,"validation_status":"engineering_demo_unvalidated_area_gate"}}

    @app.post("/api/v1/videos",status_code=201)
    async def upload(file:UploadFile=File(...)):
        try:
            if not (file.filename or "").lower().endswith(".mp4"):
                raise HTTPException(415,"Only H264 .mp4 accepted")
            content=await file.read(MAX_UPLOAD+1)
            if not content or len(content)>MAX_UPLOAD:
                raise HTTPException(413,"Empty file or upload exceeds100MiB")
            vid=identity()
            try:
                path=rt().store.write(f"videos/{vid}.mp4",content)
                metadata=probe(path)
            except Exception as error:
                raise HTTPException(422,f"Video rejected:{error}")
            value={"video_id":vid,"filename":Path(file.filename.replace("\\","/")).name[:120],
                   "video_url":f"/api/v1/videos/{vid}/stream","stored_path":f"videos/{vid}.mp4",**metadata}
            rt().store.put("video",vid,value)
            return {k:v for k,v in value.items() if k!="stored_path"}
        finally:
            await file.close()

    @app.get("/api/v1/videos/{video_id}/stream")
    def stream(video_id:str,request:Request):
        value=record("video",video_id)
        byte_range=request.headers.get("range","")
        if byte_range and not re.fullmatch(r"bytes=(\d{1,18}-\d{0,18}|-\d{1,18})",byte_range):
            raise HTTPException(416,"Only a single byte range is supported")
        return FileResponse(rt().store.path(value["stored_path"]),media_type="video/mp4")

    @app.post("/api/v1/analyses",status_code=202)
    def analyze(body:AnalysisInput):
        record("video",body.video_id)
        task=identity()
        config={**body.model_dump(),"segmentation_threshold":.5,"min_region_area_ratio":area_gate,
                "candidate_rule_status":"UNVALIDATED_ENGINEERING_DEMO","source_checkpoint_sha256":rt().adapter.describe()["source_checkpoint_sha256"],
                "model_weights_sha256":rt().adapter.describe().get("weights_sha256","TEST_ONLY"),
                "model_device":rt().adapter.describe()["device"],"preprocessing":rt().adapter.describe().get("input",{}),
                "postprocessing_version":"candidate-area-demo-v1"}
        value={"task_id":task,"video_id":body.video_id,"status":"queued","configuration":config,"created_at":now(),
               "target_frames":None,"processed_frames":0,"failed_frames":0,"candidate_frames":0,
               "elapsed_seconds":0.,"cancel_requested":False,"phase":"queued","warnings":WARNINGS}
        # Queue consumer takes the same store lock before reading the committed task.
        with rt().store.lock:
            if rt().jobs.full():raise HTTPException(429,"Queue full; wait for an existing task")
            rt().store.put("analysis",task,value)
            rt().jobs.put_nowait(task)
        return {k:value[k] for k in ("task_id","video_id","status","configuration","created_at")}

    @app.get("/api/v1/analyses/{task_id}")
    def status(task_id:str):
        row=record("analysis",task_id)
        saved=rt().store.list("frame",task_id)
        row["processed_frames"]=sum(r["status"]=="completed" for r in saved)
        row["failed_frames"]=sum(r["status"]=="failed" for r in saved)
        row["candidate_frames"]=sum(r["is_candidate"] is True for r in saved)
        attempted=len(saved)
        total=row["target_frames"]
        return {**row,"progress":{"target_frames":total,"processed_frames":row["processed_frames"],"candidate_frames":row["candidate_frames"],
                                   "failed_frames":row["failed_frames"],"percent":100*attempted/total if total else None},
                "performance":{"elapsed_seconds":row["elapsed_seconds"],"processing_fps":row["processed_frames"]/row["elapsed_seconds"] if row["elapsed_seconds"] else 0.}}

    @app.post("/api/v1/analyses/{task_id}/cancel",status_code=202)
    def cancel(task_id:str):
        row=record("analysis",task_id)
        if row["status"] in ("queued","running"):
            row=rt().store.update(task_id,cancel_requested=True)
        return {"task_id":task_id,"status":row["status"],"cancel_requested":row["cancel_requested"]}

    @app.get("/api/v1/analyses/{task_id}/frames")
    def frames(task_id:str,after:str|None=None,limit:int=Query(100,ge=1,le=200),candidate_only:bool=False):
        cursor=-1
        if after is not None:
            if not re.fullmatch(r"\d{1,10}",after):raise HTTPException(422,"after must be a frame_index cursor")
            cursor=int(after)
        rows=[r for r in frame_rows(task_id) if r["frame_index"]>cursor and (not candidate_only or r["is_candidate"] is True)]
        items=rows[:limit]
        # next_cursor is conventional pagination; last_cursor lets polling keep
        # its high-water mark when the current final page has no next page yet.
        return {"task_id":task_id,"items":items,"next_cursor":str(items[-1]["frame_index"]) if len(rows)>limit else None,
                "last_cursor":str(items[-1]["frame_index"]) if items else after,"has_more":len(rows)>limit}

    @app.get("/api/v1/analyses/{task_id}/keyframes")
    def keyframes(task_id:str):
        rows=[r for r in frame_rows(task_id) if r["status"]=="completed"]
        indexes=np.linspace(0,len(rows)-1,min(60,len(rows)),dtype=int) if rows else []
        return {"task_id":task_id,"keyframes":[{k:rows[i][k] for k in ("frame_id","timestamp_seconds","thumbnail_url","is_candidate")} for i in indexes]}

    @app.get("/api/v1/assets/{kind}/{name}")
    def asset(kind:Literal["frames","thumbs","masks","reviews"],name:str):
        pattern=r"[a-f0-9]{32}\.png" if kind=="reviews" else r"[a-f0-9]{32}-f\d{8}\.(png|jpg)"
        if not re.fullmatch(pattern,name):raise HTTPException(404,"Asset not found")
        path=rt().store.path(f"{kind}/{name}")
        if not path.is_file():raise HTTPException(404,"Asset not found")
        return FileResponse(path,media_type="image/jpeg" if name.endswith(".jpg") else "image/png")

    @app.put("/api/v1/frames/{frame_id}/review")
    async def review(frame_id:str,edited_mask:UploadFile=File(...),base_revision_id:str|None=Form(None),
                     review_status:Literal["reviewed","rejected"]=Form("reviewed"),notes:str=Form("")):
        content=await edited_mask.read(16*1024**2+1);await edited_mask.close()
        if len(content)>16*1024**2 or len(notes)>2000:raise HTTPException(413,"Review size limit")
        frame=record("frame",frame_id)
        if frame["status"]!="completed":raise HTTPException(409,"Cannot review failed frame")
        try:
            with Image.open(io.BytesIO(content)) as im:
                if im.format!="PNG" or im.mode not in ("1","L") or im.size!=(frame["width"],frame["height"]):
                    raise ValueError("Require native-resolution single-channel binary PNG")
                mask=np.asarray(im)
                if not np.isin(mask,(0,1,255)).all():raise ValueError("Mask must contain only0/1 or0/255")
                encoded=image_bytes(Image.fromarray((mask>0).astype(np.uint8)*255,mode="L"))
        except Exception as error:raise HTTPException(422,str(error))
        with rt().store.lock:
            frame=record("frame",frame_id)
            if (base_revision_id or None)!=frame["revision_id"]:raise HTTPException(409,"Revision conflict; reload frame")
            revision=identity()
            rt().store.write(f"reviews/{revision}.png",encoded)
            row={"frame_id":frame_id,"revision_id":revision,"review_status":review_status,"edited_mask_url":f"/api/v1/assets/reviews/{revision}.png",
                 "notes":notes,"saved_at":now(),"base_revision_id":base_revision_id or None}
            rt().store.put("review",revision,row,frame["task_id"],frame["frame_index"])
            frame.update(review_status=review_status,revision_id=revision,edited_mask_url=row["edited_mask_url"])
            rt().store.put("frame",frame_id,frame,frame["task_id"],frame["frame_index"])
        return row

    @app.get("/api/v1/analyses/{task_id}/report")
    def report(task_id:str):
        task=record("analysis",task_id);rows=frame_rows(task_id)
        good=[r for r in rows if r["status"]=="completed"]
        candidates=[r for r in good if r["is_candidate"]]
        return {"task_id":task_id,"video_id":task["video_id"],"status":task["status"],"model_version":MODEL_VERSION,
                "configuration":task["configuration"],"video":{k:v for k,v in record("video",task["video_id"]).items() if k!="stored_path"},
                "sampling":{"mode":task["configuration"]["sampling_mode"],"interval_seconds":task["configuration"]["interval_seconds"]},
                "summary":{"analyzed_frames":len(good),"candidate_frames":len(candidates),"failed_frames":len(rows)-len(good),
                           "candidate_frame_ratio":len(candidates)/len(good) if good else None,
                           "reviewed_frames":sum(r["review_status"]!="unreviewed" for r in rows),
                           "processing_time_seconds":task["elapsed_seconds"],
                           "mean_segmented_area_ratio":float(np.mean([r["segmented_area_ratio"] for r in good])) if good else None},
                "candidate_timestamps_seconds":[r["timestamp_seconds"] for r in candidates],
                "representative_frames":good[:8],"human_reviews":rt().store.list("review",task_id),
                "statistics_basis":"original_AI_masks; reviews are separate, never silently overwrite AI statistics",
                "warnings":task["warnings"],"generated_at":now()}

    @app.post("/api/v1/analyses/{task_id}/exports")
    def export(task_id:str,body:ExportInput):
        summary=report(task_id);rows=frame_rows(task_id)
        if not body.include_reviewed_masks:summary.pop("human_reviews",None)
        eid=identity()
        if body.format=="json":
            data=json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2).encode("utf-8")
        else:
            text=io.StringIO();keys=["frame_id","frame_index","timestamp_seconds","status","is_candidate","max_region_area_ratio","foreground_score"]
            if body.include_reviewed_masks:keys += ["review_status","revision_id","edited_mask_url"]
            writer=csv.DictWriter(text,fieldnames=keys);writer.writeheader()
            writer.writerows({k:r.get(k) for k in keys} for r in rows)
            data=text.getvalue().encode("utf-8-sig")
        relative=f"exports/{eid}.{body.format}"
        rt().store.write(relative,data)
        rt().store.put("export",eid,{"path":relative,"format":body.format})
        return {"export_id":eid,"status":"completed","download_url":f"/api/v1/exports/{eid}/download",
                "reviewed_masks":"references_only_not_binary_bundle" if body.include_reviewed_masks else "excluded"}

    @app.get("/api/v1/exports/{export_id}/download")
    def download(export_id:str):
        row=record("export",export_id)
        return FileResponse(rt().store.path(row["path"]),filename=f"endolens-{export_id}.{row['format']}")
    return app


app=create_app()
