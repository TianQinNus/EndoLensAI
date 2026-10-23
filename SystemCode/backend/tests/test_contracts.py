import io
import tempfile
import time
import threading
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import create_app, image_bytes
from backend.storage import Store
from backend.video import probe, decoded, schedule
from backend.endolens_model import MODEL_VERSION, PredictionResult
from backend.endolens_model.postprocess import postprocess
from backend.tests.make_demo_video import make_video

ROOT=Path(__file__).resolve().parent.parent


class StubAdapter:
    """Synthetic API-test double; never used by the shipped server default."""
    def __init__(self,fail_call=0,block=False):
        self.calls=0;self.fail_call=fail_call;self.block=block
        self.entered=threading.Event();self.release=threading.Event()
    def describe(self):return {"device":"synthetic-test-stub","source_checkpoint_sha256":"TEST_ONLY","model_version":MODEL_VERSION}
    def predict(self,frame,version):
        self.calls+=1
        if self.block and self.calls==1:
            self.entered.set();self.release.wait(timeout=10)
        if self.calls==self.fail_call:raise RuntimeError("Synthetic inference failure")
        p=np.zeros(frame.shape[:2],np.float32);p[20:40,30:50]=.8
        return PredictionResult(p,MODEL_VERSION,1.,"synthetic-test-stub")


class PostprocessingTests(unittest.TestCase):
    def test_empty_is_not_candidate_at_zero_area_gate(self):
        p=np.zeros((20,30),np.float32);result=postprocess(p)
        self.assertFalse(result.is_candidate);self.assertIsNone(result.foreground_score)
        self.assertEqual(result.max_region_area_ratio,0.)

    def test_area_gate_never_deletes_small_mask(self):
        p=np.zeros((100,100),np.float32);p[30:32,50:52]=.9
        result=postprocess(p,min_region_area_ratio=.01)
        self.assertFalse(result.is_candidate);self.assertEqual(int(result.binary_mask.sum()),4)
        self.assertAlmostEqual(result.foreground_score,.9,places=6)
        self.assertEqual(result.binary_mask.dtype,np.uint8)

    def test_components_coordinates_holes_and_limits(self):
        p=np.zeros((40,70),np.float32);p[5:20,30:50]=.8;p[8:12,34:38]=0
        result=postprocess(p)
        self.assertEqual(result.regions[0]["bbox_xywh"],[30,5,20,15])
        self.assertEqual(result.binary_mask[9,35],0)
        self.assertTrue(all(0<=x<70 and 0<=y<40 for c in result.contours for x,y in c))
        many=np.zeros((90,90),np.float32);many[::3,::3]=1
        self.assertTrue(postprocess(many).display_geometry_truncated)

    def test_invalid_probability_and_threshold(self):
        for value in (np.zeros((2,2),np.float64),np.full((2,2),np.nan,np.float32)):
            with self.assertRaises(ValueError):postprocess(value)
        with self.assertRaises(ValueError):postprocess(np.zeros((2,2),np.float32),min_region_area_ratio=float("nan"))


class VideoTests(unittest.TestCase):
    def test_actual_pts_not_index_divided_by_fps(self):
        with tempfile.TemporaryDirectory(prefix="_test_",dir=ROOT) as folder:
            path=make_video(Path(folder)/"vfr.mp4",[0.,.1,.7,.8,1.4,1.8])
            metadata=probe(path)
            frames=list(decoded(path,metadata["timestamp_origin_seconds"]))
            chosen=schedule(path,metadata["timestamp_origin_seconds"],"interval",.5)
            self.assertEqual([r["frame_index"] for r in chosen],[0,2,4,5])
            self.assertAlmostEqual(chosen[1]["timestamp_seconds"],frames[2][1])
            self.assertNotAlmostEqual(frames[2][1],2/metadata["fps"],places=2)
            self.assertIsNone(schedule(path,metadata["timestamp_origin_seconds"],"interval",.5,lambda:True))


class APITests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="_test_",dir=ROOT)
        self.folder=Path(self.temporary.name)
        self.video=make_video(self.folder/"clip.mp4",[0.,.1,.7,.8,1.4,1.8])
        self.client=None
        self.adapter=StubAdapter()

    def tearDown(self):
        self.adapter.release.set()
        if self.client:self.client.__exit__(None,None,None)
        self.temporary.cleanup()

    def start(self,adapter=None):
        self.adapter=adapter or self.adapter
        self.client=TestClient(create_app(self.folder/"storage",adapter=self.adapter))
        self.client.__enter__()
        return self.client

    def upload(self):
        response=self.client.post("/api/v1/videos",files={"file":("demo.mp4",self.video.read_bytes(),"video/mp4")})
        self.assertEqual(response.status_code,201,response.text)
        return response.json()

    def task(self,vid,every=False):
        response=self.client.post("/api/v1/analyses",json={"video_id":vid,"sampling_mode":"every_frame" if every else "interval",
                                                       "interval_seconds":None if every else .5,"model_version":MODEL_VERSION})
        self.assertEqual(response.status_code,202,response.text)
        return response.json()["task_id"]

    def wait(self,task,terminal=True):
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            row=self.client.get(f"/api/v1/analyses/{task}").json()
            if row["status"] in ("completed","cancelled","failed"):return row
            time.sleep(.02)
        self.fail("Worker did not reach terminal status")

    def test_upload_stream_range_and_validation(self):
        self.start();video=self.upload()
        response=self.client.get(video["video_url"],headers={"Range":"bytes=0-7"})
        self.assertEqual(response.status_code,206);self.assertEqual(response.content,self.video.read_bytes()[:8])
        self.assertEqual(self.client.get(video["video_url"],headers={"Range":"bytes=0-1,4-5"}).status_code,416)
        self.assertEqual(self.client.post("/api/v1/videos",files={"file":("x.avi",b"not-video")}).status_code,415)
        self.assertEqual(self.client.post("/api/v1/videos",files={"file":("x.mp4",b"not-video")}).status_code,422)
        response=self.client.post("/api/v1/analyses",json={"video_id":video["video_id"],"model_version":"polyp-pvt-v1"})
        self.assertEqual(response.status_code,422)

    def test_pipeline_pagination_statistics_assets_and_exports(self):
        self.start();video=self.upload();task=self.task(video["video_id"])
        status=self.wait(task);self.assertEqual(status["status"],"completed",status)
        self.assertEqual(status["progress"]["target_frames"],4)
        self.assertEqual(status["progress"]["processed_frames"],4)
        page=self.client.get(f"/api/v1/analyses/{task}/frames?limit=2").json()
        self.assertTrue(page["has_more"]);self.assertIsNotNone(page["next_cursor"])
        second=self.client.get(f"/api/v1/analyses/{task}/frames?after={page['next_cursor']}&limit=2").json()
        self.assertFalse(second["has_more"]);self.assertIsNone(second["next_cursor"])
        self.assertEqual([r["frame_index"] for r in page["items"]+second["items"]],[0,2,4,5])
        row=page["items"][0]
        with Image.open(io.BytesIO(self.client.get(row["mask_url"]).content)) as mask:
            self.assertEqual(mask.size,(320,240));self.assertEqual(set(np.unique(np.asarray(mask))),{0,255})
        self.assertEqual(self.client.get(row["source_frame_url"]).status_code,200)
        report=self.client.get(f"/api/v1/analyses/{task}/report").json()
        self.assertEqual(report["summary"]["candidate_frames"],status["progress"]["candidate_frames"])
        self.assertEqual(report["summary"]["analyzed_frames"],4)
        self.assertEqual(len(self.client.get(f"/api/v1/analyses/{task}/keyframes").json()["keyframes"]),4)
        exported=self.client.post(f"/api/v1/analyses/{task}/exports",json={"format":"csv"}).json()
        self.assertEqual(self.client.get(exported["download_url"]).status_code,200)
        self.assertIn("FOREGROUND_SCORE_NOT_CALIBRATED_CLINICAL_PROBABILITY",status["warnings"])

    def test_failed_frames_are_null_not_negative(self):
        self.start(StubAdapter(fail_call=2));video=self.upload();task=self.task(video["video_id"])
        status=self.wait(task)
        self.assertEqual(status["progress"]["failed_frames"],1)
        rows=self.client.get(f"/api/v1/analyses/{task}/frames").json()["items"]
        failed=next(r for r in rows if r["status"]=="failed")
        self.assertIsNone(failed["is_candidate"]);self.assertIsNone(failed["max_region_area_ratio"])
        self.assertIsNone(failed["mask_url"])
        report=self.client.get(f"/api/v1/analyses/{task}/report").json()
        self.assertEqual(report["summary"]["analyzed_frames"],3)
        self.assertEqual(report["summary"]["candidate_frame_ratio"],1.)

    def test_review_revision_conflict_and_ai_mask_unchanged(self):
        self.start();video=self.upload();task=self.task(video["video_id"]);self.wait(task)
        frame=self.client.get(f"/api/v1/analyses/{task}/frames").json()["items"][0]
        original=self.client.get(frame["mask_url"]).content
        edit=image_bytes(Image.new("L",(320,240),0))
        url=f"/api/v1/frames/{frame['frame_id']}/review"
        response=self.client.put(url,files={"edited_mask":("edit.png",edit,"image/png")},data={"review_status":"reviewed"})
        self.assertEqual(response.status_code,200,response.text)
        revision=response.json()["revision_id"]
        self.assertEqual(self.client.put(url,files={"edited_mask":("edit.png",edit)},data={"review_status":"reviewed"}).status_code,409)
        response2=self.client.put(url,files={"edited_mask":("edit.png",edit)},data={"base_revision_id":revision})
        self.assertEqual(response2.status_code,200)
        self.assertEqual(self.client.get(frame["mask_url"]).content,original)
        wrong=image_bytes(Image.new("L",(5,5),0))
        self.assertEqual(self.client.put(url,files={"edited_mask":("edit.png",wrong)}).status_code,422)
        report=self.client.get(f"/api/v1/analyses/{task}/report").json()
        self.assertEqual(report["summary"]["candidate_frames"],4)
        self.assertEqual(report["summary"]["reviewed_frames"],1)
        self.assertEqual(len(report["human_reviews"]),2)

    def test_cancel_during_inference_retains_partial_results(self):
        stub=StubAdapter(block=True);self.start(stub)
        video=self.upload();task=self.task(video["video_id"],every=True)
        self.assertTrue(stub.entered.wait(5))
        self.client.post(f"/api/v1/analyses/{task}/cancel");stub.release.set()
        status=self.wait(task)
        self.assertEqual(status["status"],"cancelled")
        self.assertEqual(status["progress"]["processed_frames"],1)

    def test_page_refresh_and_server_restart_recovery(self):
        self.start();video=self.upload();task=self.task(video["video_id"]);self.wait(task)
        self.client.__exit__(None,None,None);self.client=None
        store=Store(self.folder/"storage")
        previous=store.get("analysis",task)
        previous["task_id"]="a"*32;previous["status"]="running"
        store.put("analysis","a"*32,previous);store.close()
        self.start()
        self.assertEqual(self.client.get(f"/api/v1/analyses/{task}").json()["progress"]["processed_frames"],4)
        stopped=self.client.get("/api/v1/analyses/"+"a"*32).json()
        self.assertEqual(stopped["status"],"failed");self.assertIn("SERVER_RESTARTED",stopped["error"])


class StorageTests(unittest.TestCase):
    def test_quota_immutable_asset_and_path_escape(self):
        with tempfile.TemporaryDirectory(prefix="_test_",dir=ROOT) as folder:
            store=Store(folder)
            try:
                store.write("frames/a.bin",b"abc")
                with self.assertRaises(FileExistsError):store.write("frames/a.bin",b"xyz")
                with self.assertRaises(ValueError):store.path("../outside")
                store.max_bytes=store.used_bytes
                with self.assertRaises(ValueError):store.write("frames/b.bin",b"x")
            finally:store.close()


if __name__=="__main__":unittest.main(verbosity=2)
