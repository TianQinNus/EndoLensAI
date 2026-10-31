import io,tempfile,time,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from endolens_model import MODEL_VERSION,PredictionResult,postprocess
from backend.app import create_app
from examples.make_demo_video import make_video
ROOT=Path(__file__).resolve().parent.parent

class RegionRuleTests(unittest.TestCase):
    def test_low_mean_region_is_deleted_from_every_display_output(self):
        p=np.zeros((30,40),np.float32);p[3:10,5:12]=.79
        out=postprocess(p)
        self.assertEqual(out.raw_component_count,1);self.assertEqual(out.removed_component_count,1)
        self.assertFalse(out.is_candidate);self.assertFalse(out.binary_mask.any())
        self.assertEqual(out.contours,[]);self.assertEqual(out.regions,[]);self.assertIsNone(out.foreground_score)
    def test_mean_is_not_max_pixel_or_whole_frame_score(self):
        p=np.zeros((20,20),np.float32);p[1,1]=.99;p[1,2]=.6
        self.assertFalse(postprocess(p).is_candidate)
        p[10,10]=.99
        out=postprocess(p);self.assertEqual(int(out.binary_mask.sum()),1)
        self.assertEqual(out.binary_mask[10,10],1);self.assertEqual(out.binary_mask[1,1],0)
    def test_equality_small_regions_and_holes(self):
        p=np.zeros((20,20),np.float32);p[3:10,3:10]=.8;p[5:7,5:7]=0;p[15,15]=.9
        out=postprocess(p);self.assertTrue(out.is_candidate);self.assertEqual(out.component_count,2)
        self.assertEqual(out.binary_mask[5,5],0);self.assertEqual(out.binary_mask[15,15],1)
        self.assertTrue(all(r['foreground_score']>=.8 for r in out.regions))
    def test_8_connected_diagonal_region_uses_combined_mean(self):
        p=np.zeros((10,10),np.float32);p[1,1]=.99;p[2,2]=.6
        out=postprocess(p);self.assertEqual(out.raw_component_count,1);self.assertFalse(out.is_candidate)
    def test_fixed_settings_and_invalid_scores(self):
        p=np.zeros((10,10),np.float32)
        for kwargs in [{'segmentation_threshold':.7},{'min_region_mean_score':.9},{'min_region_area_ratio':.001}]:
            with self.assertRaises(ValueError):postprocess(p,**kwargs)
        with self.assertRaises(ValueError):postprocess(np.full((3,3),np.nan,np.float32))

class LowScoreAdapter:
    def describe(self):return {'device':'synthetic-test-stub','model_version':MODEL_VERSION,'source_checkpoint_sha256':'TEST_ONLY'}
    def predict(self,frame,version):
        p=np.zeros(frame.shape[:2],np.float32);p[20:40,30:50]=.79
        return PredictionResult(p,MODEL_VERSION,1.,'synthetic-test-stub')

class FilteredAPITests(unittest.TestCase):
    def test_video_masks_candidates_report_and_model_rule_agree(self):
        with tempfile.TemporaryDirectory(prefix='_test_',dir=ROOT) as folder:
            folder=Path(folder);clip=make_video(folder/'demo.mp4',[0.,.2,.7,.9])
            with TestClient(create_app(folder/'storage',adapter=LowScoreAdapter())) as client:
                info=client.get('/api/v1/models').json()['postprocessing']
                self.assertEqual(info['min_region_mean_score'],.8);self.assertTrue(info['mask_filtering'])
                uploaded=client.post('/api/v1/videos',files={'file':('demo.mp4',clip.read_bytes(),'video/mp4')}).json()
                response=client.post('/api/v1/analyses',json={'video_id':uploaded['video_id'],'sampling_mode':'every_frame','model_version':MODEL_VERSION})
                self.assertEqual(response.status_code,202,response.text);task=response.json()['task_id']
                deadline=time.monotonic()+15
                while True:
                    status=client.get('/api/v1/analyses/'+task).json()
                    if status['status'] in ['completed','failed']:break
                    self.assertLess(time.monotonic(),deadline);time.sleep(.02)
                self.assertEqual(status['status'],'completed');self.assertEqual(status['candidate_frames'],0)
                rows=client.get(f'/api/v1/analyses/{task}/frames').json()['items']
                self.assertEqual(len(rows),4)
                for row in rows:
                    self.assertFalse(row['is_candidate']);self.assertEqual(row['removed_component_count'],1)
                    self.assertEqual(row['contours'],[]);self.assertEqual(row['regions'],[])
                    self.assertIsNone(row['foreground_score'])
                    with Image.open(io.BytesIO(client.get(row['mask_url']).content)) as mask:self.assertFalse(np.asarray(mask).any())
                self.assertEqual(client.get(f'/api/v1/analyses/{task}/frames?candidate_only=true').json()['items'],[])
                self.assertEqual(client.get(f'/api/v1/analyses/{task}/report').json()['summary']['candidate_frames'],0)

if __name__=='__main__':unittest.main()
