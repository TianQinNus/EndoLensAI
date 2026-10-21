import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import run_e7b2 as m


def fixture(root,selection_available=True):
    manifest=root/'manifest'; inference=root/'inference'; prior=root/'prior'
    for directory in (manifest,inference,prior):
        directory.mkdir()
    records=[]; probabilities=[]
    for i in range(12):
        split='calibration' if i<8 else 'selection'
        prefix='c' if i<8 else 's'; sid=f'{prefix}{i}'
        gt=np.zeros((16,18),dtype=np.uint8); gt[3:11,4:13]=255
        image=root/f'{sid}_image.png'; mask=root/f'{sid}_mask.png'
        available=split=='calibration' or selection_available
        if available:
            Image.fromarray(np.full((16,18,3),i,dtype=np.uint8)).save(image)
            Image.fromarray(gt).save(mask)
        records.append(dict(sample_id=sid,group_id=str(i//2),declared_group_id=str(i//2),
            split=split,source='synthetic',area_bin='large_ge10pct',gt_area_fraction=float((gt>0).mean()),
            image=image.name,mask=mask.name,
            image_sha256=m.e6.file_hash(image) if available else 'unavailable',
            mask_sha256=m.e6.file_hash(mask) if available else 'unavailable'))
        for model in ('e0','e4'):
            p=np.where(gt>0,0.75 if model=='e0' else 0.9,
                       0.07 if model=='e0' else 0.04).astype(np.float32)
            if model=='e0' and i%2==0:
                p[3:5,4:13]=0.4
            path=inference/f'{sid}_{model}.npz'
            if available:
                np.savez_compressed(path,probability=p)
            probabilities.append(dict(sample_id=sid,model=model,probability_path=path.name,
                probability_sha256=m.e6.file_hash(path) if available else 'unavailable'))
    m.e6.write_json(manifest/'development_manifest.json',records)
    mi=dict(status='COMPLETED',run_kind='E6_PREPARE',
        manifest_sha256=m.e6.file_hash(manifest/'development_manifest.json'),
        manifest_signature=m.e6.signature(records))
    ii=dict(status='COMPLETED',run_kind='FULL_DEVELOPMENT',processed_count=len(records),
        manifest_signature=mi['manifest_signature'],e0_checkpoint_sha256='e0',e4_checkpoint_sha256='e4')
    m.e6.write_json(manifest/'run_info.json',mi)
    m.e6.write_json(inference/'run_info.json',ii)
    m.e6.write_csv(inference/'per_image.csv',probabilities)
    m.e6.write_json(prior/'run_info.json',dict(status='COMPLETED',run_kind='E7B1_CALIBRATION_GATE_CV',
        cv_status='OOF_NO_PASS',manifest_signature=mi['manifest_signature'],
        e0_checkpoint_sha256='e0',e4_checkpoint_sha256='e4'))
    m.e6.write_json(prior/'cv_decision.json',dict(primary_policy='logistic_safe',comparator_promoted=False,
        comparator_checks=dict(mean_dice_gain=True,worst_source=True,mae=True,failure_tail=True)))
    return argparse.Namespace(project=str(root),manifest_run='manifest',inference_run='inference',
        prior_run='prior',output='out',seed=42,max_pixels_per_image=8192)


def simple_config():
    return dict(rule=dict(rule_iou_max=0.5,rule_e4_area_max=0.1),
                calibrator=dict(calibrator_status='FEASIBLE',slope=0.5,intercept=-2.0))


class FrozenRuleTests(unittest.TestCase):
    def test_medians_use_only_predictions(self):
        rows=[dict(agreement_iou=x,e4_predicted_area=a,gt_area_fraction=0.99,safe_label=1)
              for x,a in ((0.,0.02),(0.2,0.04),(0.4,0.06),(1.,0.08))]
        rule=m.median_rule(rows)
        self.assertAlmostEqual(rule['rule_iou_max'],0.3)
        self.assertAlmostEqual(rule['rule_e4_area_max'],0.05)
        for r in rows:
            r.update(gt_area_fraction=0.,safe_label=0)
        self.assertEqual(rule,m.median_rule(rows))
        with self.assertRaises(ValueError):
            m.median_rule([])
        with self.assertRaises(ValueError):
            m.median_rule([dict(agreement_iou=np.nan,e4_predicted_area=0.1)])

    def test_rule_is_and_lte_and_does_not_use_gt(self):
        config=simple_config()
        x=dict(agreement_iou=0.5,e4_predicted_area=0.1)
        self.assertTrue(m.route(x,config))
        self.assertFalse(m.route(dict(x,agreement_iou=0.5001),config))
        self.assertFalse(m.route(dict(x,e4_predicted_area=0.1001),config))
        self.assertTrue(m.route(dict(x,gt_area_fraction=1.,e0_dice=0.,mae_delta=999.),config))
        config['calibrator']['calibrator_status']='INFEASIBLE'
        self.assertFalse(m.route(x,config))

    def test_mask_scale_and_e0_fallback(self):
        config=simple_config(); config['rule'].update(rule_iou_max=1.,rule_e4_area_max=1.)
        p0=np.full((8,8),0.7,dtype=np.float32); p4=np.full((8,8),0.6,dtype=np.float32)
        p,mask,_,selected=m.apply_prediction(p0,p4,config)
        self.assertTrue(selected); self.assertTrue(mask.all()); self.assertTrue((p<0.5).all())
        config['calibrator']['calibrator_status']='INFEASIBLE'
        p,mask,_,selected=m.apply_prediction(p0,p4,config)
        self.assertFalse(selected)
        np.testing.assert_array_equal(p,p0)
        np.testing.assert_array_equal(mask,p0>=0.5)

    def test_probability_validation(self):
        p=np.zeros((4,4),dtype=np.float32)
        with self.assertRaises(ValueError):
            m.apply_prediction(p,np.zeros((3,4)),simple_config())
        with self.assertRaises(ValueError):
            m.apply_prediction(p,np.full(p.shape,np.nan),simple_config())

    def test_freeze_with_inaccessible_selection_files(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory); args=fixture(root,False)
            original=m.a2.load_pair
            def calibration_only(project,record,*args):
                self.assertEqual(record['split'],'calibration')
                return original(project,record,*args)
            with patch.object(m.a2,'load_pair',side_effect=calibration_only):
                m.freeze(args)
            config=m.load_locked(root/'out')
            self.assertEqual(config['calibration_images'],8)
            self.assertEqual(config['expected_selection_images'],4)
            self.assertEqual(m.e6.read_json(root/'out/freeze_info.json')['selection_files_read'],0)
            self.assertFalse((root/'out/selection').exists())
            with self.assertRaises(FileExistsError):
                m.freeze(args)

    def test_tampered_config_is_blocked_before_selection(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory); args=fixture(root)
            m.freeze(args)
            path=root/'out/locked_config.json'
            config=m.e6.read_json(path); config['rule']['rule_iou_max']=0.123
            m.e6.write_json(path,config)
            with self.assertRaisesRegex(ValueError,'signature mismatch'):
                m.evaluate(argparse.Namespace(project=str(root),run='out',bootstrap_repetitions=1000))
            self.assertFalse((root/'out/selection').exists())

    def test_infeasible_freeze_blocks_selection(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory); args=fixture(root)
            row=dict(strength=0.,slope=1.,intercept=0.,sampled_nll=0.5,calibration_mae=0.2,mae_limit=0.1,feasible=False)
            with patch.object(m.b0,'fit_fold',return_value=(dict(row),{},[row],0.097,
                              'NO_FEASIBLE_CALIBRATOR_IDENTITY_FALLBACK')):
                m.freeze(args)
            with self.assertRaisesRegex(ValueError,'No feasible frozen calibrator'):
                m.evaluate(argparse.Namespace(project=str(root),run='out',bootstrap_repetitions=1000))
            self.assertFalse((root/'out/selection').exists())

    def test_end_to_end_freeze_evaluate_verify_and_single_attempt(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory); args=fixture(root)
            m.freeze(args)
            path=root/'out/locked_config.json'; initial_hash=m.e6.file_hash(path)
            evaluate_args=argparse.Namespace(project=str(root),run='out',bootstrap_repetitions=1000)
            original=m.evaluate_record
            def after_freeze(project,record,*args):
                self.assertEqual(m.e6.file_hash(path),initial_hash)
                self.assertEqual(record['split'],'selection')
                return original(project,record,*args)
            with patch.object(m,'evaluate_record',side_effect=after_freeze),\
                    patch.object(m.b0,'fit_fold',side_effect=AssertionError('Refit after freeze')):
                m.evaluate(evaluate_args)
            self.assertEqual(initial_hash,m.e6.file_hash(path))
            info=m.e6.read_json(root/'out/run_info.json')
            self.assertEqual(info['status'],'COMPLETED')
            self.assertEqual(info['selection_images_evaluated'],4)
            self.assertEqual(info['test_images_evaluated'],0)
            decision=m.e6.read_json(root/'out/selection/decision.json')
            self.assertEqual(decision['candidate_count'],1)
            self.assertFalse(decision['formal_test_allowed'])
            self.assertEqual(m.load_locked(root/'out')['validation_status'],'NOT_EVALUATED')
            m.verify(argparse.Namespace(project=str(root),run='out'))
            with self.assertRaisesRegex(ValueError,'already attempted'):
                m.evaluate(evaluate_args)

    def test_failed_selection_cannot_be_retried(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory); args=fixture(root,False)
            m.freeze(args)
            evaluate_args=argparse.Namespace(project=str(root),run='out',bootstrap_repetitions=1000)
            with self.assertRaises(FileNotFoundError):
                m.evaluate(evaluate_args)
            self.assertEqual(m.e6.read_json(root/'out/run_info.json')['status'],'FAILED')
            with self.assertRaisesRegex(ValueError,'already attempted'):
                m.evaluate(evaluate_args)


if __name__=='__main__':
    unittest.main(verbosity=2)
