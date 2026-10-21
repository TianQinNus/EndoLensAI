"""Portable integrity and single-frame replay check, no research workspace access."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from endolens_model import ModelAdapter
from endolens_model.adapter import sha256
ROOT=Path(__file__).resolve().parent


def verify(integrity_only=False):
    manifest=json.loads((ROOT/"MANIFEST.json").read_text(encoding="utf-8"))
    for name,expected in manifest["files"].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path)!=expected:raise ValueError("Bundle file changed:"+name)
    if not integrity_only:
        adapter=ModelAdapter(device="cpu")
        with Image.open(ROOT/"examples/synthetic_frame.png") as im:frame=np.asarray(im.convert("RGB"),np.uint8)
        actual=adapter.predict(frame).probability_map
        reference=np.load(ROOT/"examples/synthetic_probability.npy",allow_pickle=False)
        if actual.shape!=reference.shape or not np.allclose(actual,reference,rtol=0,atol=1e-5):raise ValueError("Synthetic CPU replay differs")
    print("BUNDLE_VERIFIED",manifest["model_version"],"integrity_only",integrity_only)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--integrity-only",action="store_true")
    verify(parser.parse_args().integrity_only)
