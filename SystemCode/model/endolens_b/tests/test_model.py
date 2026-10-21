import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from endolens_model import ModelAdapter, MODEL_VERSION
from endolens_model.adapter import preprocess

ROOT=Path(__file__).resolve().parent.parent


class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.adapter=ModelAdapter(device="cpu")

    def test_real_exported_cpu_model_replays_fixture(self):
        with Image.open(ROOT/"examples/synthetic_frame.png") as im:rgb=np.asarray(im.convert("RGB"),np.uint8)
        result=self.adapter.predict(rgb,MODEL_VERSION)
        self.assertEqual(result.probability_map.shape,rgb.shape[:2]);self.assertEqual(result.probability_map.dtype,np.float32)
        np.testing.assert_allclose(result.probability_map,np.load(ROOT/"examples/synthetic_probability.npy",allow_pickle=False),rtol=0,atol=1e-6)
        self.assertGreater(result.inference_time_ms,0.)
        self.assertEqual(result.device,"cpu")

    def test_strict_rgb_and_version_and_no_silent_fallback(self):
        for value in (np.zeros((20,30),np.uint8),np.zeros((20,30,4),np.uint8),np.zeros((20,30,3),np.float32)):
            with self.assertRaises(ValueError):self.adapter.predict(value)
        with self.assertRaises(ValueError):self.adapter.predict(np.zeros((20,30,3),np.uint8),"polyp-pvt-v1")

    def test_preprocessing_matches_fixed_rgb_normalization(self):
        rgb=np.zeros((12,19,3),np.uint8);rgb[:,:,0]=255
        x=preprocess(rgb)
        self.assertEqual(tuple(x.shape),(1,3,352,352))
        expected=torch.tensor([(1-.485)/.229,-.456/.224,-.406/.225])
        torch.testing.assert_close(x[0,:,0,0],expected,rtol=1e-6,atol=1e-6)


if __name__=="__main__":unittest.main(verbosity=2)
