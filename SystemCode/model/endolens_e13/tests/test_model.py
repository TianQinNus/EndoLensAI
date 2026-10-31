import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from endolens_model import ModelAdapter,MODEL_VERSION
from endolens_model.adapter import preprocess
ROOT=Path(__file__).resolve().parent.parent

class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('torch.hub.download_url_to_file',side_effect=AssertionError('Offline loading required')):
            cls.adapter=ModelAdapter(device='cpu')
    def test_offline_cpu_replays_raw_and_filtered_fixture(self):
        with Image.open(ROOT/'examples/synthetic_frame.png') as im:rgb=np.asarray(im.convert('RGB'),np.uint8)
        result=self.adapter.predict(rgb)
        np.testing.assert_allclose(result.probability_map,np.load(ROOT/'examples/synthetic_probability.npy',allow_pickle=False),rtol=0,atol=1e-6)
        with Image.open(ROOT/'examples/synthetic_mask.png') as im:expected=np.asarray(im)>0
        np.testing.assert_array_equal(result.binary_mask>0,expected)
        self.assertEqual(result.is_candidate,bool(expected.any()))
        self.assertTrue(all(r['foreground_score']>=.8 for r in result.regions))
        self.assertEqual(result.probability_map.dtype,np.float32);self.assertEqual(result.binary_mask.dtype,np.uint8)
    def test_preprocessing_is320_rgb_div255_only(self):
        rgb=np.zeros((12,19,3),np.uint8);rgb[:,:,0]=255;x=preprocess(rgb)
        self.assertEqual(tuple(x.shape),(1,3,320,320))
        torch.testing.assert_close(x[0,:,0,0],torch.tensor([1.,0.,0.]),rtol=0,atol=0)
    def test_input_and_old_model_version_rejected(self):
        for value in [np.zeros((2,3),np.uint8),np.zeros((2,3,4),np.uint8),np.zeros((2,3,3),np.float32)]:
            with self.assertRaises(ValueError):self.adapter.predict(value)
        with self.assertRaises(ValueError):self.adapter.predict(np.zeros((10,12,3),np.uint8),'unet-b-expanded-seed42-r98-v1')
    def test_changed_release_rule_rejected(self):
        with tempfile.TemporaryDirectory(prefix='_test_',dir=ROOT) as folder:
            config=json.loads((ROOT/'model.json').read_text());config['candidate_rule']['min_component_mean_score']=.9
            (Path(folder)/'model.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):ModelAdapter(model_dir=folder,device='cpu')

if __name__=='__main__':unittest.main()
