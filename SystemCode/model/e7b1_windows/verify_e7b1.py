"""Read-only verification of actual E7-B1 CV outputs and mask-free inference."""
import argparse
from pathlib import Path
import numpy as np
import run_e7b1 as m


def verify(project, output):
    project, output=Path(project).resolve(), Path(output).resolve()
    info=m.e6.read_json(output/'run_info.json')
    protocol=m.e6.read_json(output/'protocol.json')
    if info['status']!='COMPLETED' or info['selection_images_evaluated']!=0 or info['test_images_evaluated']!=0:
        raise ValueError('Not a completed calibration-only run')
    for name,relative in (('run_e7b1.py','e7b1_windows/run_e7b1.py'),
                          ('run_e7b0.py','e7b0_windows/run_e7b0.py'),
                          ('run_e7a2.py','e7a2_windows/run_e7a2.py'),
                          ('run_e7a.py','e7a_windows/run_e7a.py'),
                          ('run_e6.py','e6_windows/run_e6.py')):
        if m.e6.file_hash(project/relative)!=protocol['shared_source_hashes'][name]:
            raise ValueError('Shared source changed: '+name)
    if m.e6.file_hash(output/'executed_source_snapshot.py')!=protocol['shared_source_hashes']['run_e7b1.py']:
        raise ValueError('Executed snapshot differs from protocol')
    audits=m.e6.read_json(output/'fit_audit.json')
    labels=m.e6.read_csv(output/'crossfit_training_labels.csv')
    fit_count=0
    for audit in audits:
        training=set(audit['outer_training_group_ids'])
        heldout=set(audit['outer_heldout_group_ids'])
        if training & heldout:
            raise ValueError('Outer patient leakage')
        fold_labels=[r for r in labels if int(r['outer_fold'])==audit['outer_fold']]
        if len({r['sample_id'] for r in fold_labels})!=len(fold_labels):
            raise ValueError('Duplicate label in an outer training set')
        for fit in audit['fits']:
            fit_count+=1
            tg=set(fit['training_group_ids']); hg=set(fit['heldout_group_ids'])
            if tg & hg or (tg|hg)&heldout or not (tg|hg)<=training:
                raise ValueError('Cross-fitted patient leakage')
            if fit['inner_fold']>=0:
                matched=[r for r in fold_labels if int(r['label_fold'])==fit['inner_fold']]
                if {r['sample_id'] for r in matched}!=set(fit['heldout_sample_ids']):
                    raise ValueError('Label-fold coverage mismatch')
                for r in matched:
                    expected=int(fit['calibrator']['calibrator_status']=='FEASIBLE'
                                 and float(r['inner_fusion_dice_delta'])>0
                                 and float(r['inner_fusion_mae_delta'])<=0)
                    if int(r['safe_label'])!=expected:
                        raise ValueError('Incorrect safe label')
            else:
                if {r['sample_id'] for r in fold_labels}!=set(fit['training_sample_ids']):
                    raise ValueError('Incomplete final gate training labels')
    if fit_count!=info['calibrator_fit_count']:
        raise ValueError('Unexpected fit count')
    inference=project/'runs/e6_development_inference_seed42'
    manifest_info,inference_info,records,index=m.b0.verify_inputs(
        project,project/'runs/e6_development_seed42',inference)
    if manifest_info['manifest_signature']!=info['manifest_signature']:
        raise ValueError('Output manifest signature mismatch')
    for model in ('e0','e4'):
        key=model+'_checkpoint_sha256'
        if inference_info[key]!=info[key]:
            raise ValueError('Output checkpoint mismatch')
    records={r['sample_id']:r for r in records}
    rows=m.e6.read_csv(output/'oof_per_image.csv')
    if len(rows)!=len(records) or {r['sample_id'] for r in rows}!=set(records):
        raise ValueError('Incomplete/duplicated OOF coverage')
    features=m.e6.read_csv(output/'prediction_features.csv')
    if set(features[0])!=set(('sample_id','group_id','cv_fold')+m.FEATURES):
        raise ValueError('Unexpected runtime feature column, possible GT metadata')
    models={x['outer_fold']:x for x in m.e6.read_json(output/'fold_models.json')}
    expected_folds=set(range(info['outer_folds']))
    if set(models)!=expected_folds or {r['outer_fold'] for r in audits}!=expected_folds:
        raise ValueError('Outer fold coverage mismatch')
    assignments=m.e6.read_csv(output/'fold_assignment.csv')
    assigned_folds={r['sample_id']:int(r['cv_fold']) for r in assignments}
    if len(assigned_folds)!=len(rows):
        raise ValueError('Assignment coverage mismatch')
    for row in rows:
        if assigned_folds[row['sample_id']]!=int(row['cv_fold']):
            raise ValueError('OOF assignment mismatch')
    max_error=0.0
    for i,row in enumerate(rows,1):
        record=records[row['sample_id']]
        model=models[int(row['cv_fold'])]
        p0,p4,gt=m.a2.load_pair(project,record,inference,index)
        for policy in ('logistic_safe','quantile_rule'):
            probability,mask,decision=m.apply_prediction(p0,p4,model['gate'],model['calibrator'],policy)
            if decision[policy]!=(row[policy+'_uses_fusion']=='True'):
                raise ValueError('Prediction-only inference route mismatch')
            metrics=m.e6.probability_metrics(mask.astype(np.float32),gt,0.5)
            metrics['mae']=float(np.abs(probability-gt.astype(np.float32)).mean())
            for metric in ('dice','iou','mae'):
                error=abs(metrics[metric]-float(row[policy+'_'+metric]))
                max_error=max(max_error,error)
                if error>1e-7:
                    raise ValueError('Inference score mismatch: '+policy+' '+row['sample_id'])
        if i%40==0:
            print(f'[verified predictions {i}/{len(rows)}]',flush=True)
    result=dict(status='VERIFIED',images=len(rows),patients=len({r['group_id'] for r in rows}),
                calibrator_fits=fit_count,max_inference_metric_error=max_error,
                selection_images_evaluated=0,test_images_evaluated=0,
                source_hashes_match=True,crossfit_audits_pass=True)
    print(result,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default=str(m.ROOT))
    p.add_argument('--output',default='runs/e7b1_gate_cv_seed42')
    args=p.parse_args()
    verify(args.project,m.e6.resolve(Path(args.project).resolve(),args.output))
