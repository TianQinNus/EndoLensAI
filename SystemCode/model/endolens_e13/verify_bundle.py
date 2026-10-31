"""Verify a standalone extracted bundle and replay its filtered CPU fixture."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from endolens_model import ModelAdapter,MODEL_VERSION
from endolens_model.adapter import sha256
ROOT=Path(__file__).resolve().parent

def verify(integrity_only=False):
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    if manifest['model_version']!=MODEL_VERSION:raise ValueError('Manifest version mismatch')
    for name,expected in manifest['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path)!=expected:raise ValueError('Bundle file changed: '+name)
    if not integrity_only:
        adapter=ModelAdapter(device='cpu')
        with Image.open(ROOT/'examples/synthetic_frame.png') as image:rgb=np.asarray(image.convert('RGB'),np.uint8)
        actual=adapter.predict(rgb)
        if not np.allclose(actual.probability_map,np.load(ROOT/'examples/synthetic_probability.npy',allow_pickle=False),rtol=0,atol=1e-5):raise ValueError('Raw CPU replay mismatch')
        with Image.open(ROOT/'examples/synthetic_mask.png') as image:expected=np.asarray(image)>0
        if not np.array_equal(actual.binary_mask>0,expected):raise ValueError('Filtered mask replay mismatch')
        if actual.is_candidate!=bool(expected.any()):raise ValueError('Candidate status mismatch')
    print('BUNDLE_VERIFIED',MODEL_VERSION,'integrity_only',integrity_only)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--integrity-only',action='store_true');verify(p.parse_args().integrity_only)
