import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import run_e7a2 as m


class RevisionTests(unittest.TestCase):
    def test_identity_preserves_endpoints(self):
        p=np.array([0.,0.1,0.5,1.],dtype=np.float32)
        np.testing.assert_array_equal(m.transform(p,1.,0.),p)

    def test_mapped_threshold_equivalence(self):
        raw=np.linspace(0.001,0.999,1000,dtype=np.float32)
        for slope,intercept in ((0.4,-1.9),(1.,0.),(2.,0.3)):
            p=m.transform(raw,slope,intercept)
            for t in m.RAW_THRESHOLDS:
                np.testing.assert_array_equal(p>=m.mapped_threshold(t,slope,intercept),raw>=t)

    def test_mae_constraint_selection(self):
        rows=[dict(strength=0.,sampled_nll=0.3,calibration_mae=0.04),
              dict(strength=0.5,sampled_nll=0.2,calibration_mae=0.042),
              dict(strength=1.,sampled_nll=0.1,calibration_mae=0.05)]
        self.assertEqual(m.pick_calibrator(rows,0.043)['strength'],0.5)
        self.assertIsNone(m.pick_calibrator(rows,0.03))

    def test_mask_and_probability_mae_use_separate_scales(self):
        raw=np.array([[0.4,0.6]],dtype=np.float32)
        gt=np.array([[False,True]])
        p=m.transform(raw,0.4,-2.)
        result=m.metrics(p,raw,gt,0.5)
        self.assertEqual(result['dice'],1.)
        self.assertAlmostEqual(result['mae'],float(np.abs(p-gt).mean()))

    def test_no_pass_blocks_before_test_reads(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            c=dict(experiment=m.EXPERIMENT,validation_status='NO_PASS')
            c['config_signature']=m.e6.signature(c)
            m.e6.write_json(root/'config.json',c)
            args=argparse.Namespace(project=str(root),config='config.json',output='final')
            with patch.object(m.e6,'validate_test_sources',side_effect=AssertionError('Test accessed')):
                with self.assertRaisesRegex(ValueError,'formal test blocked'):
                    m.command_apply_test(args)
            self.assertFalse((root/'final').exists())

    def test_synthetic_selection_and_full_image_constraint(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory); inference=root/'inference'; inference.mkdir()
            records=[]; index={}
            for i in range(4):
                # Unequal dimensions verify per-image rather than pooled weighting.
                shape=(4+4*i,5+3*i)
                gt=np.zeros(shape,dtype=np.uint8); gt[:2,:2]=255
                mask=root/f'{i}.png'; Image.fromarray(gt).save(mask)
                record=dict(sample_id=str(i),mask=mask.name,source='synthetic',
                            group_id=str(i),area_bin='medium_1to10pct',
                            split='calibration' if i<2 else 'selection')
                records.append(record)
                for model in ('e0','e4'):
                    p=np.where(gt>0,0.7 if model=='e0' else 0.9,
                               0.05+0.05*i if model=='e0' else 0.02).astype(np.float32)
                    path=inference/f'{i}_{model}.npz'; np.savez_compressed(path,probability=p)
                    index[model,str(i)]=dict(probability_path=path.name)
            (root/'manifest').mkdir(); (root/'prior').mkdir()
            m.e6.write_json(inference/'run_info.json',{})
            m.e6.write_json(root/'prior/selection_config.json',{})
            mi=dict(manifest_signature='s',manifest_sha256='h')
            ii=dict(e0_checkpoint_sha256='e0',e4_checkpoint_sha256='e4')
            args=argparse.Namespace(project=str(root),output='out',manifest_run='manifest',
                                    inference_run='inference',prior_run='prior',seed=42,
                                    max_pixels_per_image=8192,bootstrap_repetitions=1000)
            with patch.object(m.e6,'verify_development_manifest',return_value=(mi,records)), \
                 patch.object(m.e6,'verify_inference',return_value=(ii,index)), \
                 patch.object(m,'check_prior'):
                m.command_select(args)
            constraint=m.e6.read_json(root/'out/calibration_constraint.json')
            p0=[]
            for r in records[:2]:
                raw=m.e6.load_probability(inference/index['e0',r['sample_id']]['probability_path'])
                gt=m.e6.read_mask(root/r['mask'])
                p0.append(float(np.abs(raw-gt.astype(np.float32)).mean()))
            self.assertAlmostEqual(constraint['baseline_e0_mae'],float(np.mean(p0)))
            options=m.e6.read_csv(root/'out/calibrators.csv')
            for option in options:
                if option['family']=='constrained_affine':
                    self.assertLessEqual(float(option['calibration_mae']),constraint['mae_limit'])
            c=m.e6.read_json(root/'out/selection_config.json')
            sig=c.pop('config_signature'); self.assertEqual(m.e6.signature(c),sig)
            grid=m.e6.read_csv(root/'out/grid_search.csv')
            self.assertTrue(any(r['family']=='identity' and float(r['alpha'])==0 for r in grid))
            self.assertTrue(all(float(r['raw_threshold']) in m.RAW_THRESHOLDS for r in grid))

    def test_formal_application_on_synthetic_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            for folder in ('e0','e4eval'):
                (root/folder).mkdir()
            gt=np.array([[0,255],[0,255]],dtype=np.uint8)
            Image.fromarray(gt).save(root/'mask.png')
            Image.fromarray(np.zeros((2,2,3),dtype=np.uint8)).save(root/'image.png')
            p=np.array([[0.1,0.9],[0.1,0.9]],dtype=np.float32)
            for folder in ('e0','e4eval'):
                np.savez_compressed(root/folder/'p.npz',probability=p)
            record=dict(sample_id='synthetic/0',dataset='synthetic',image='image.png',mask='mask.png',
                        image_sha256=m.e6.file_hash(root/'image.png'),mask_sha256=m.e6.file_hash(root/'mask.png'))
            baseline=m.e6.probability_metrics(p,gt>0,0.5)
            table={record['sample_id']:dict(baseline,probability_path='p.npz')}
            c=dict(experiment=m.EXPERIMENT,validation_status='PASS',gates=m.e6.DEFAULT_GATES,
                   alpha_e4=0.,raw_threshold=0.5,affine_slope=1.,affine_intercept=0.,
                   e0_checkpoint_sha256='e0hash',e4_checkpoint_sha256='e4hash')
            c['config_signature']=m.e6.signature(c)
            m.e6.write_json(root/'config.json',c)
            args=argparse.Namespace(project=str(root),config='config.json',output='out',
                                    e0_run='e0',e4_run='e4',e4_eval='e4eval')
            with patch.object(m.e6,'validate_test_sources',return_value=(
                    [record],table,table,dict(checkpoint_sha256='e0hash'),dict(final_checkpoint_sha256='e4hash'))), \
                 patch.object(m.e6,'DATASETS',('synthetic',)):
                m.command_apply_test(args)
            info=m.e6.read_json(root/'out/run_info.json')
            self.assertEqual(info['status'],'COMPLETED')
            self.assertEqual(info['processed_count'],1)
            # Successful execution is distinct from meeting gain gates.
            self.assertEqual(info['decision'],'FAIL')
            np.testing.assert_array_equal(m.e6.read_mask(root/'out/masks/synthetic/0.png'),gt>0)


if __name__=='__main__':
    unittest.main(verbosity=2)
