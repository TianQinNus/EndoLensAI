"""Clearly synthetic nonclinical clip; no patient data redistribution."""
import argparse
import sys
from fractions import Fraction
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import av


def synthetic_frame(index=0):
    image=Image.new("RGB",(320,240),(25,35,55))
    draw=ImageDraw.Draw(image)
    draw.rectangle((10,45,310,225),fill=(40,70,90))
    x=70+index*8
    draw.ellipse((x,90,x+60,150),fill=(190,80,70))
    draw.text((12,10),"SYNTHETIC INTEGRATION FIXTURE",fill="white")
    draw.text((12,28),"NOT A CLINICAL VIDEO",fill="white")
    return np.asarray(image,np.uint8)


def make_video(path,timestamps=None):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    timestamps=timestamps if timestamps is not None else [i/6 for i in range(12)]
    with av.open(str(path),mode="w",format="mp4") as out:
        stream=out.add_stream("libx264",rate=30)
        stream.width=320;stream.height=240;stream.pix_fmt="yuv420p"
        stream.options={"preset":"ultrafast","crf":"22","bf":"0"}
        for i,seconds in enumerate(timestamps):
            frame=av.VideoFrame.from_ndarray(synthetic_frame(i),format="rgb24")
            frame.pts=round(seconds*30000);frame.time_base=Fraction(1,30000)
            for packet in stream.encode(frame):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    return path


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="synthetic_demo.mp4")
    print(make_video(parser.parse_args().output).resolve())
