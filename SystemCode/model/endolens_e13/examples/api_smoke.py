"""Validate the running reference server over real HTTP; no clinical metrics."""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import urljoin
import httpx

ROOT=Path(__file__).resolve().parent.parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--base-url",default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--video",type=Path,default=ROOT/"examples/synthetic_demo.mp4")
    parser.add_argument("--timeout",type=float,default=120.)
    parser.add_argument("--output",type=Path,help="Optional local JSON evidence file")
    args=parser.parse_args();base=args.base_url.rstrip("/")
    with httpx.Client(timeout=30.) as client:
        def call(method,path,**kwargs):
            response=client.request(method,base+path,**kwargs)
            response.raise_for_status();return response.json()
        health=call("GET","/health")
        models=call("GET","/models")
        # Do not hardcode a different model version from the server selection.
        version="e13-r50-joint-r94-r4-region080-v2"
        assert health['model_version']==version
        assert models['postprocessing']['min_region_mean_score']==.8
        assert models['postprocessing']['mask_filtering'] is True
        with args.video.open("rb") as video:
            uploaded=call("POST","/videos",files={"file":(args.video.name,video,"video/mp4")})
        stream=client.get(urljoin(base,uploaded["video_url"]),headers={"Range":"bytes=0-15"})
        assert stream.status_code==206 and len(stream.content)==16,"HTTP Range failed"
        task=call("POST","/analyses",json={"video_id":uploaded["video_id"],"sampling_mode":"interval",
                                           "interval_seconds":.5,"model_version":version})
        tid=task["task_id"];deadline=time.monotonic()+args.timeout
        while True:
            status=call("GET",f"/analyses/{tid}")
            if status["status"] in ("completed","failed","cancelled"):break
            if time.monotonic()>deadline:raise TimeoutError(f"Task {tid} is still running; inspect server; not automatically cancelled")
            time.sleep(.25)
        assert status["status"]=="completed",status
        rows=[];cursor=None
        while True:
            page=call("GET",f"/analyses/{tid}/frames",params={"limit":100,**({"after":cursor} if cursor else {})})
            rows.extend(page["items"])
            if not page["has_more"]:break
            cursor=page["next_cursor"]
        completed=[r for r in rows if r["status"]=="completed"]
        assert completed,"No usable frame results"
        from PIL import Image
        import io,numpy as np
        for row in completed:
            response=client.get(urljoin(base,row['mask_url']));response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as image:mask=np.asarray(image)
            assert row['is_candidate']==bool(mask.any())
            assert all(r['foreground_score']>=.8 for r in row['regions'])
            if not row['is_candidate']:assert not row['contours'] and row['foreground_score'] is None
        for key in ("source_frame_url","thumbnail_url","mask_url"):
            response=client.get(urljoin(base,completed[0][key]));response.raise_for_status()
            assert response.content
        report=call("GET",f"/analyses/{tid}/report")
        assert report["summary"]["analyzed_frames"]==status["progress"]["processed_frames"]==len(completed)
        assert report["summary"]["candidate_frames"]==sum(r["is_candidate"] is True for r in rows)
        keyframes=call("GET",f"/analyses/{tid}/keyframes")
        exported=call("POST",f"/analyses/{tid}/exports",json={"format":"csv","include_reviewed_masks":True})
        csv=client.get(urljoin(base,exported["download_url"]));csv.raise_for_status()
        assert "frame_id" in csv.text
        proof={"success":True,"transport":"real HTTP","task_id":tid,"model_version":version,
                          "progress":status["progress"],"keyframes":len(keyframes["keyframes"]),
                          "range_206":True,"assets_ok":True,"report_counts_match":True,"csv_download_ok":True,
               "warning":"Integration only. Default video is synthetic; not a clinical/FPS evaluation."}
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(proof,indent=2))


if __name__=="__main__":main()
