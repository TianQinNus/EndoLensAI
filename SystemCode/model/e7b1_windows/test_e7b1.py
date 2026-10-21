import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import run_e7b1 as m


def training_rows(n=40):
    rows=[]
    for i in range(n):
        value=i/(n-1)
        row=dict(sample_id=str(i),group_id=str(i//2),safe_label=int(value>0.65))
        row.update({feature:value for feature in m.FEATURES})
        rows.append(row)
    return rows


class GateTests(unittest.TestCase):
    def test_logistic_fits_signal_and_finite_collinear_features(self):
        rows=training_rows()
        gate=m.fit_gate(rows)
        self.assertEqual(gate['status'],'FITTED')
        low=m.predict_routes(rows[0],gate,'FEASIBLE')
        high=m.predict_routes(rows[-1],gate,'FEASIBLE')
        self.assertLess(low['logistic_score'],high['logistic_score'])
        self.assertFalse(low['logistic_safe'])
        self.assertTrue(high['logistic_safe'])
        self.assertTrue(np.isfinite(gate['coefficients']).all())

    def test_patient_weights_and_training_only_scaler(self):
        rows=training_rows()
        rows[0]['group_id']='one_frame_patient'
        weights=m.patient_weights(rows)
        group_sums={}
        for r,w in zip(rows,weights):
            group_sums[r['group_id']]=group_sums.get(r['group_id'],0)+w
        self.assertAlmostEqual(min(group_sums.values()),max(group_sums.values()))
        gate=m.fit_gate(rows)
        expected=np.sum(weights[:,None]*m.feature_matrix(rows),axis=0)
        np.testing.assert_allclose(gate['mean'],expected)
        snapshot=list(gate['mean'])
        extreme={k:100.0 for k in m.FEATURES}
        m.predict_routes(extreme,gate,'FEASIBLE')
        self.assertEqual(snapshot,gate['mean'])

    def test_single_class_and_infeasible_calibrator_fallback(self):
        rows=training_rows()
        for r in rows:
            r['safe_label']=0
        gate=m.fit_gate(rows)
        self.assertEqual(gate['status'],'SINGLE_CLASS_FALLBACK_E0')
        self.assertFalse(m.predict_routes(rows[-1],gate,'FEASIBLE')['logistic_safe'])
        gate=m.fit_gate(training_rows())
        decision=m.predict_routes(training_rows()[-1],gate,'NO_FEASIBLE_CALIBRATOR_IDENTITY_FALLBACK')
        self.assertFalse(decision['logistic_safe'])
        self.assertFalse(decision['quantile_rule'])

    def test_route_ignores_ground_truth_metadata(self):
        rows=training_rows()
        gate=m.fit_gate(rows)
        features={k:rows[-1][k] for k in m.FEATURES}
        before=m.predict_routes(features,gate,'FEASIBLE')
        features.update(gt_area_fraction=1.,safe_label=0,e0_dice=0.,fusion_mae_delta=999.)
        self.assertEqual(before,m.predict_routes(features,gate,'FEASIBLE'))

    def test_prediction_application_preserves_raw_mask_scale(self):
        gate=m.fit_gate(training_rows())
        gate.update(status='FITTED',coefficients=[20.]+[0.]*len(m.FEATURES))
        p0=np.full((8,8),0.7,dtype=np.float32)
        p4=np.full((8,8),0.6,dtype=np.float32)
        calibrator=dict(calibrator_status='FEASIBLE',slope=0.5,intercept=-2.)
        probability,mask,decision=m.apply_prediction(p0,p4,gate,calibrator)
        self.assertTrue(decision['logistic_safe'])
        self.assertTrue(mask.all())
        self.assertTrue((probability<0.5).all())
        probability,mask,decision=m.apply_prediction(p0,p4,gate,
            dict(calibrator,calibrator_status='INFEASIBLE'))
        np.testing.assert_array_equal(probability,p0)
        np.testing.assert_array_equal(mask,p0>=0.5)

    def test_end_to_end_crossfit_isolation_and_no_selection_read(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory); manifest=root/'manifest'; inference=root/'inference'
            manifest.mkdir(); inference.mkdir()
            records=[]; probability_rows=[]
            for i in range(18):
                gt=np.zeros((16,18),dtype=np.uint8); gt[3:11,4:13]=255
                image=root/f'{i}_image.png'; mask=root/f'{i}_mask.png'
                Image.fromarray(np.full((16,18,3),i,dtype=np.uint8)).save(image)
                Image.fromarray(gt).save(mask)
                record=dict(sample_id=str(i),group_id=str(i//2),declared_group_id=str(i//2),
                    source='synthetic',split='calibration',area_bin='large_ge10pct',
                    gt_area_fraction=float((gt>0).mean()),image=image.name,mask=mask.name,
                    image_sha256=m.e6.file_hash(image),mask_sha256=m.e6.file_hash(mask))
                records.append(record)
                for model in ('e0','e4'):
                    p=np.where(gt>0,0.85 if model=='e0' else 0.95,
                               0.1 if model=='e0' else 0.05).astype(np.float32)
                    if model=='e0' and i%3==0:
                        p[3:5,4:13]=0.3
                    path=inference/f'{i}_{model}.npz'
                    np.savez_compressed(path,probability=p)
                    probability_rows.append(dict(sample_id=str(i),model=model,probability_path=path.name,
                                                  probability_sha256=m.e6.file_hash(path)))
            records.append(dict(sample_id='forbidden_selection',group_id='forbidden',split='selection',
                                image='DO_NOT_READ_image.png',mask='DO_NOT_READ_mask.png'))
            for model in ('e0','e4'):
                probability_rows.append(dict(sample_id='forbidden_selection',model=model,
                    probability_path='DO_NOT_READ_'+model+'.npz',probability_sha256='unused'))
            m.e6.write_json(manifest/'development_manifest.json',records)
            mi=dict(status='COMPLETED',run_kind='E6_PREPARE',
                manifest_sha256=m.e6.file_hash(manifest/'development_manifest.json'),
                manifest_signature=m.e6.signature(records))
            m.e6.write_json(manifest/'run_info.json',mi)
            m.e6.write_json(inference/'run_info.json',dict(status='COMPLETED',run_kind='FULL_DEVELOPMENT',
                processed_count=len(records),manifest_signature=mi['manifest_signature'],
                e0_checkpoint_sha256='e0',e4_checkpoint_sha256='e4'))
            m.e6.write_csv(inference/'per_image.csv',probability_rows)
            args=argparse.Namespace(project=str(root),manifest_run='manifest',inference_run='inference',
                output='out',outer_folds=3,inner_folds=2,seed=42,max_pixels_per_image=8192,
                bootstrap_repetitions=1000)
            original_pipeline=m.fit_pipeline
            original_load=m.a2.load_pair
            calls=[]
            def audited_pipeline(project,training,*args):
                allowed={r['sample_id'] for r in training}
                def isolated_load(project,record,*args):
                    self.assertIn(record['sample_id'],allowed)
                    return original_load(project,record,*args)
                calls.append(allowed)
                with patch.object(m.a2,'load_pair',side_effect=isolated_load):
                    return original_pipeline(project,training,*args)
            with patch.object(m,'fit_pipeline',side_effect=audited_pipeline):
                m.run(args)
            info=m.e6.read_json(root/'out/run_info.json')
            self.assertEqual(info['status'],'COMPLETED')
            self.assertEqual(info['processed_count'],18)
            self.assertEqual(info['selection_images_evaluated'],0)
            self.assertEqual(info['test_images_evaluated'],0)
            self.assertEqual(info['calibrator_fit_count'],9)
            oof=m.e6.read_csv(root/'out/oof_per_image.csv')
            self.assertEqual(len(oof),18)
            self.assertEqual(len({r['sample_id'] for r in oof}),18)
            audits=m.e6.read_json(root/'out/fit_audit.json')
            self.assertEqual(len(calls),3)
            for allowed,audit in zip(calls,audits):
                outer_held=set(audit['outer_heldout_group_ids'])
                for fit in audit['fits']:
                    self.assertFalse(set(fit['training_group_ids']) & outer_held)
                    self.assertFalse(set(fit['heldout_group_ids']) & outer_held)
                    self.assertFalse(set(fit['training_group_ids']) & set(fit['heldout_group_ids']))
                    self.assertTrue(set(fit['training_sample_ids']) <= allowed)
            decision=m.e6.read_json(root/'out/cv_decision.json')
            self.assertFalse(decision['formal_test_allowed'])
            self.assertFalse(decision['comparator_promoted'])


if __name__=='__main__':
    unittest.main(verbosity=2)
