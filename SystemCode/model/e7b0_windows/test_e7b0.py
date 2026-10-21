import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import run_e7b0 as m


class DiagnosticTests(unittest.TestCase):
    def test_groups_are_disjoint_and_deterministic(self):
        records=[dict(sample_id=f'{g}/{i}',group_id=str(g)) for g in range(10) for i in range(1+g%3)]
        first=m.assign_folds(records,5,42)
        self.assertEqual(first,m.assign_folds(records,5,42))
        assignments={}
        for r in first:
            assignments.setdefault(r['group_id'],set()).add(r['cv_fold'])
        self.assertTrue(all(len(v)==1 for v in assignments.values()))
        self.assertEqual({r['cv_fold'] for r in first},set(range(5)))

    def test_regions_partition_image_and_reconstruct_mae(self):
        gt=np.zeros((15,16),dtype=bool); gt[3:12,4:13]=True
        regions=m.regions(gt,2)
        total=sum(r.astype(np.int32) for r in regions.values())
        np.testing.assert_array_equal(total,np.ones(gt.shape))
        p=np.full(gt.shape,0.3,dtype=np.float32)
        rows=m.regional_rows('test',0,'e0',p,gt,regions)
        self.assertAlmostEqual(sum(r['image_mae_contribution'] for r in rows),
                               float(np.abs(p-gt).mean()),places=6)
        self.assertTrue(all(r['pixels']>0 for r in rows))

    def test_oracles_preserve_e0_when_fusion_is_unsafe(self):
        e0=dict(dice=0.8,mae=0.03)
        self.assertTrue(m.choose_oracle(e0,dict(dice=0.9,mae=0.04),False))
        self.assertFalse(m.choose_oracle(e0,dict(dice=0.9,mae=0.04),True))
        self.assertTrue(m.choose_oracle(e0,dict(dice=0.9,mae=0.02),True))
        self.assertFalse(m.choose_oracle(e0,dict(dice=0.7,mae=0.02),False))

    def test_features_accept_no_ground_truth(self):
        p=np.zeros((4,4),dtype=np.float32)
        features=m.predicted_features(p,p)
        self.assertEqual(features['agreement_iou'],1.)
        self.assertEqual(features['e0_predicted_area'],0.)
        self.assertTrue(all(np.isfinite(v) for v in features.values()))

    def test_end_to_end_refits_and_does_not_read_selection(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory); manifest=root/'manifest'; inference=root/'inference'
            manifest.mkdir(); inference.mkdir()
            records=[]; prob_rows=[]
            for i in range(8):
                gt=np.zeros((10+i,12),dtype=np.uint8); gt[2:7,3:8]=255
                image=root/f'{i}_image.png'; mask=root/f'{i}_mask.png'
                Image.fromarray(np.full((10+i,12,3),i*10,dtype=np.uint8)).save(image)
                Image.fromarray(gt).save(mask)
                record=dict(sample_id=str(i),group_id=str(i//2),declared_group_id=str(i//2),
                            source='synthetic',area_bin='large_ge10pct',gt_area_fraction=float((gt>0).mean()),
                            split='calibration',image=image.name,mask=mask.name,
                            image_sha256=m.e6.file_hash(image),mask_sha256=m.e6.file_hash(mask))
                records.append(record)
                for model in ('e0','e4'):
                    p=np.where(gt>0,0.75 if model=='e0' else 0.9,
                               0.05 if model=='e0' else 0.1).astype(np.float32)
                    path=inference/f'{i}_{model}.npz'
                    np.savez_compressed(path,probability=p)
                    prob_rows.append(dict(sample_id=str(i),model=model,
                                          probability_path=path.name,probability_sha256=m.e6.file_hash(path)))
            # Deliberately inaccessible selection files: successful run proves isolation.
            records.append(dict(sample_id='forbidden_selection',group_id='other',split='selection',
                                image='DO_NOT_READ_image.png',mask='DO_NOT_READ_mask.png'))
            for model in ('e0','e4'):
                prob_rows.append(dict(sample_id='forbidden_selection',model=model,
                                      probability_path='DO_NOT_READ_'+model+'.npz',probability_sha256='unused'))
            m.e6.write_json(manifest/'development_manifest.json',records)
            mi=dict(status='COMPLETED',run_kind='E6_PREPARE',
                    manifest_sha256=m.e6.file_hash(manifest/'development_manifest.json'),
                    manifest_signature=m.e6.signature(records))
            m.e6.write_json(manifest/'run_info.json',mi)
            m.e6.write_json(inference/'run_info.json',dict(status='COMPLETED',run_kind='FULL_DEVELOPMENT',
                            processed_count=len(records),manifest_signature=mi['manifest_signature'],
                            e0_checkpoint_sha256='e0',e4_checkpoint_sha256='e4'))
            m.e6.write_csv(inference/'per_image.csv',prob_rows)
            args=argparse.Namespace(project=str(root),manifest_run='manifest',inference_run='inference',
                                    output='out',folds=2,seed=42,boundary_radius=1,
                                    max_pixels_per_image=8192,bootstrap_repetitions=1000)
            original=m.fit_fold
            training_calls=[]
            def audited_fit(project,training,*args):
                training_calls.append({r['group_id'] for r in training})
                return original(project,training,*args)
            with patch.object(m,'fit_fold',side_effect=audited_fit):
                m.run(args)
            info=m.e6.read_json(root/'out/run_info.json')
            self.assertEqual(info['status'],'COMPLETED')
            self.assertEqual(info['processed_count'],8)
            self.assertEqual(info['selection_images_evaluated'],0)
            audits=m.e6.read_json(root/'out/fold_audit.json')
            self.assertEqual(len(training_calls),2)
            for call,audit in zip(training_calls,audits):
                self.assertEqual(call,set(audit['training_group_ids']))
                self.assertFalse(call & set(audit['heldout_group_ids']))
            oof=m.e6.read_csv(root/'out/oof_per_image.csv')
            self.assertEqual(len(oof),8)
            self.assertEqual(len({r['sample_id'] for r in oof}),8)
            for r in oof:
                self.assertGreaterEqual(float(r['oracle_safe_dice']),float(r['e0_dice']))
                self.assertLessEqual(float(r['oracle_safe_mae']),float(r['e0_mae']))


if __name__=='__main__':
    unittest.main(verbosity=2)
