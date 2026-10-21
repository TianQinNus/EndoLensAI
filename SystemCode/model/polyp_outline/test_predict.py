import importlib.util
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from predict import PolypOutline, mask_edge

class Tests(unittest.TestCase):
    def test_geometry(self):
        m=np.zeros((9,9),bool);m[2:7,2:7]=True
        self.assertEqual(int(mask_edge(m,1).sum()),16)
        self.assertFalse(mask_edge(m,1)[4,4])
        self.assertFalse(mask_edge(np.zeros((9,9),bool)).any())
        self.assertEqual(int(mask_edge(np.ones((9,9),bool),1).sum()),32)
        m[4,4]=False;self.assertTrue(mask_edge(m,1)[3,4])
    def test_pixels(self):
        r=object.__new__(PolypOutline);m=np.zeros((30,40),bool);m[5:25,10:30]=True
        r.predict_mask=lambda im:m
        a=np.random.default_rng(42).integers(0,256,(30,40,3),dtype=np.uint8)
        b=np.asarray(r.annotate(Image.fromarray(a)));e=mask_edge(m)
        np.testing.assert_array_equal(a[~e],b[~e]);self.assertTrue(np.all(b[e]==(0,255,0)))
    def test_parity(self):
        p=Path(__file__).resolve().parents[1]/'unetpp_context/train.py'
        if not p.exists():self.skipTest('Original training source not included')
        r=PolypOutline();s=importlib.util.spec_from_file_location('original',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        original=m.UNet(32).eval();original.load_state_dict(r.model.state_dict())
        with torch.inference_mode():
            x=torch.rand(1,3,64,64);torch.testing.assert_close(original(x),r.model(x),rtol=0,atol=0)
        self.assertEqual(r.predict_mask(Image.new('RGB',(101,73))).shape,(73,101))
if __name__=='__main__':unittest.main()
