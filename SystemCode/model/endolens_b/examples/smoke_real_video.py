"""Real B+FastAPI+video smoke on synthetic content; no clinical metrics."""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from fastapi.testclient import TestClient
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from backend.app import create_app
from endolens_model import ModelAdapter, MODEL_VERSION
ROOT=Path(__file__).resolve().parent.parent


def main(output=None):
    adapter=ModelAdapter(device="cpu")
    with tempfile.TemporaryDirectory(prefix="_test_real_",dir=ROOT) as folder:
        with TestClient(create_app(folder,adapter=adapter)) as client:
            uploaded=client.post("/api/v1/videos",files={"file":("synthetic.mp4",(ROOT/"examples/synthetic_demo.mp4").read_bytes(),"video/mp4")})
            uploaded.raise_for_status()
            task=client.post("/api/v1/analyses",json={"video_id":uploaded.json()["video_id"],"sampling_mode":"interval",
                "interval_seconds":.5,"model_version":MODEL_VERSION})
            task.raise_for_status();tid=task.json()["task_id"]
            deadline=time.monotonic()+60
            while True:
                status=client.get(f"/api/v1/analyses/{tid}").json()
                if status["status"] in ("completed","failed","cancelled"):break
                if time.monotonic()>deadline:raise TimeoutError("Real video smoke timeout")
                time.sleep(.05)
            assert status["status"]=="completed",status
            assert status["progress"]["failed_frames"]==0,status
            rows=client.get(f"/api/v1/analyses/{tid}/frames").json()["items"]
            assert len(rows)==status["progress"]["processed_frames"]>0
            for row in rows:
                for key in ("source_frame_url","thumbnail_url","mask_url"):
                    assert client.get(row[key]).status_code==200
            report=client.get(f"/api/v1/analyses/{tid}/report").json()
            assert report["summary"]["candidate_frames"]==status["progress"]["candidate_frames"]
            proof={"success":True,"content":"SYNTHETIC_NONCLINICAL","actual_model":MODEL_VERSION,"device":"cpu",
                   "progress":status["progress"],"performance":status["performance"],
                   "frame_indices":[r["frame_index"] for r in rows],"timestamps_seconds":[r["timestamp_seconds"] for r in rows],
                   "assets_downloaded":len(rows)*3,"report_counts_match":True,
                   "warning":"This proves integration only; neither FPS benchmark nor clinical detection performance."}
    # End-user reruns must not modify the frozen release evidence/manifest.
    destination=Path(output) if output else ROOT/"runtime_data/real_video_smoke_local.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(proof,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path)
    main(parser.parse_args().output)
