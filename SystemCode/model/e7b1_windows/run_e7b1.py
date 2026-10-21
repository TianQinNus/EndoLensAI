"""E7-B1: fixed prediction-only gates with patient-grouped cross-fitted labels.

Calibration-only adaptive development. No model/threshold search, selection
evaluation, formal-test configuration or automatic promotion of a comparator.
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'e7b0_windows'))
import run_e7b0 as b0
a2, a1, e6 = b0.a2, b0.a1, b0.e6
FEATURES = b0.FEATURES
MODELS = ('e0', 'calibrated_fusion', 'logistic_safe', 'quantile_rule', 'oracle_safe')
RIDGE = 0.1
ROUTE_THRESHOLD = 0.5


def feature_matrix(rows):
    result = np.asarray([[r[k] for k in FEATURES] for r in rows], dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(FEATURES) or not np.isfinite(result).all():
        raise ValueError('Empty or nonfinite prediction features')
    return result


def patient_weights(rows):
    counts = Counter(r['group_id'] for r in rows)
    weights = np.array([1 / counts[r['group_id']] for r in rows], dtype=np.float64)
    return weights / weights.sum()


def fit_gate(rows):
    x = feature_matrix(rows)
    y = np.array([r['safe_label'] for r in rows], dtype=np.float64)
    if not np.isin(y, (0, 1)).all():
        raise ValueError('Gate target must be binary')
    weights = patient_weights(rows)
    mean = np.sum(weights[:, None] * x, axis=0)
    scale = np.sqrt(np.sum(weights[:, None] * (x - mean)**2, axis=0))
    scale = np.where(scale < 1e-8, 1.0, scale)
    result = dict(features=list(FEATURES), mean=mean.tolist(), scale=scale.tolist(),
                  ridge=RIDGE, route_threshold=ROUTE_THRESHOLD,
                  training_count=len(rows), positive_labels=int(y.sum()),
                  training_patients=len({r['group_id'] for r in rows}),
                  patient_weighted_prevalence=float(np.sum(weights*y)),
                  # Quantiles use only training predictions, never label outcomes.
                  rule_iou_max=float(np.median(x[:, FEATURES.index('agreement_iou')])),
                  rule_e4_area_max=float(np.median(x[:, FEATURES.index('e4_predicted_area')])) )
    if np.unique(y).size < 2:
        result.update(status='SINGLE_CLASS_FALLBACK_E0', coefficients=[0.]*(len(FEATURES)+1),
                      converged=True, iterations=0)
        return result
    design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    beta = np.zeros(design.shape[1])
    prevalence = float(np.sum(weights*y))
    beta[0] = np.log(prevalence / (1-prevalence))
    penalty = np.eye(design.shape[1])*RIDGE
    penalty[0, 0] = 0.0

    def loss(coef):
        logits = design @ coef
        return float(np.sum(weights*(np.logaddexp(0, logits)-y*logits))
                     + 0.5*coef @ penalty @ coef)

    converged = False
    for iteration in range(1, 101):
        probability = a1.sigmoid(design @ beta)
        gradient = design.T @ (weights*(probability-y)) + penalty @ beta
        hessian = design.T @ ((weights*probability*(1-probability))[:, None]*design) + penalty
        step = np.linalg.solve(hessian + np.eye(len(beta))*1e-10, gradient)
        scale_step = 1.0
        current = loss(beta)
        accepted = False
        for _ in range(30):
            candidate = beta-scale_step*step
            if loss(candidate) <= current+1e-12:
                accepted = True
                break
            scale_step *= 0.5
        if not accepted:
            break
        change = float(np.max(np.abs(candidate-beta)))
        beta = candidate
        if change < 1e-8 or np.max(np.abs(gradient)) < 1e-9:
            converged = True
            break
    result.update(status='FITTED' if converged else 'NONCONVERGED_FALLBACK_E0',
                  coefficients=beta.tolist(), converged=converged, iterations=iteration,
                  weighted_penalized_nll=loss(beta))
    return result


def predict_routes(features, gate, calibrator_status):
    """Only prediction features and fitted parameters enter route decisions."""
    x = feature_matrix([features])[0]
    z = (x-np.asarray(gate['mean']))/np.asarray(gate['scale'])
    beta = np.asarray(gate['coefficients'])
    score = float(a1.sigmoid(beta[0]+z @ beta[1:])) if gate['status']=='FITTED' else 0.0
    feasible = calibrator_status == 'FEASIBLE'
    return dict(logistic_score=score,
                logistic_safe=bool(feasible and score >= gate['route_threshold']),
                quantile_rule=bool(feasible and features['agreement_iou'] <= gate['rule_iou_max']
                                   and features['e4_predicted_area'] <= gate['rule_e4_area_max']))


def apply_prediction(p0, p4, gate, calibrator, policy='logistic_safe'):
    """Inference without a mask: return probability, binary mask and decision."""
    if policy not in ('logistic_safe', 'quantile_rule'):
        raise ValueError('Unknown prediction-only policy')
    features = b0.predicted_features(p0, p4)
    decision = predict_routes(features, gate, calibrator['calibrator_status'])
    if decision[policy]:
        raw = (0.5*p0+0.5*p4).astype(np.float32)
        probability = a2.transform(raw, calibrator['slope'], calibrator['intercept'])
        mask = raw >= 0.5
    else:
        probability, mask = p0.copy(), p0 >= 0.5
    return probability, mask, decision


def evaluate_pair(project, record, inference, index, calibrator):
    p0, p4, gt = a2.load_pair(project, record, inference, index)
    raw = (0.5*p0+0.5*p4).astype(np.float32)
    p = a2.transform(raw, calibrator['slope'], calibrator['intercept'])
    return (b0.predicted_features(p0, p4), e6.probability_metrics(p0, gt, 0.5),
            a2.metrics(p, raw, gt, 0.5))


def calibrate(project, training, inference, index, args, outer_fold, inner_fold, heldout):
    train_groups = {r['group_id'] for r in training}
    heldout_groups = {r['group_id'] for r in heldout}
    if train_groups & heldout_groups:
        raise ValueError('Patient leakage in calibration fit')
    selected, fit, path, baseline, status = b0.fit_fold(
        project, training, inference, index, args.seed, args.max_pixels_per_image, 0.5)
    calibrator = dict(selected, calibrator_status=status,
                      training_e0_mae=baseline, unconstrained_fit=fit)
    audit = dict(outer_fold=outer_fold, inner_fold=inner_fold,
                 training_sample_ids=[r['sample_id'] for r in training],
                 heldout_sample_ids=[r['sample_id'] for r in heldout],
                 training_group_ids=sorted(train_groups), heldout_group_ids=sorted(heldout_groups),
                 shared_patients=0, calibrator=calibrator)
    paths = [dict(outer_fold=outer_fold, inner_fold=inner_fold, **r) for r in path]
    return calibrator, audit, paths


def fit_pipeline(project, training, inference, index, args, outer_fold):
    """Outer heldout records are not accepted by the training pipeline."""
    crossfit = b0.assign_folds(training, args.inner_folds, args.seed)
    labels, audits, paths = [], [], []
    for inner in range(args.inner_folds):
        fit_records = [r for r in crossfit if r['cv_fold'] != inner]
        label_records = [r for r in crossfit if r['cv_fold'] == inner]
        print(f'  [outer {outer_fold+1}, label fold {inner+1}/{args.inner_folds}] '
              f'fit={len(fit_records)} labels={len(label_records)}', flush=True)
        calibrator, audit, path = calibrate(project, fit_records, inference, index,
                                            args, outer_fold, inner, label_records)
        audits.append(audit); paths.extend(path)
        for r in label_records:
            features, baseline, fused = evaluate_pair(project, r, inference, index, calibrator)
            safe = calibrator['calibrator_status']=='FEASIBLE' and b0.choose_oracle(baseline, fused, True)
            labels.append(dict(sample_id=r['sample_id'], group_id=r['group_id'],
                               outer_fold=outer_fold, label_fold=inner, safe_label=int(safe),
                               inner_fusion_dice_delta=fused['dice']-baseline['dice'],
                               inner_fusion_mae_delta=fused['mae']-baseline['mae'], **features))
    if {r['sample_id'] for r in labels} != {r['sample_id'] for r in training} or len(labels)!=len(training):
        raise ValueError('Cross-fitted label coverage mismatch')
    gate = fit_gate(labels)
    print(f'  [outer {outer_fold+1} final calibrator] train={len(training)}', flush=True)
    calibrator, audit, path = calibrate(project, training, inference, index,
                                        args, outer_fold, -1, [])
    audits.append(audit); paths.extend(path)
    return gate, calibrator, labels, audits, paths


def run(args):
    project = Path(args.project).resolve()
    output = e6.resolve(project, args.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite: '+str(output))
    output.mkdir(parents=True)
    info = dict(status='RUNNING', run_kind='E7B1_CALIBRATION_GATE_CV', started_at=e6.utc_now())
    e6.write_json(output/'run_info.json', info)
    protocol = dict(seed=args.seed, outer_folds=args.outer_folds, label_crossfit_folds=args.inner_folds,
                    primary_policy='logistic_safe', comparator='quantile_rule',
                    ridge=RIDGE, logistic_route_threshold=ROUTE_THRESHOLD,
                    logistic_features=list(FEATURES), fitting='patient-equal weighted NLL; unpenalized intercept',
                    rule='IoU <= training image median AND E4 predicted area <= training image median',
                    safe_label='inner heldout Dice gain > 0 AND calibrated MAE <= E0; feasible calibrator required',
                    fallback='E0 on infeasible calibration; logistic also falls back on single class/nonconvergence',
                    alpha_e4=0.5, raw_threshold=0.5, max_pixels_per_image=args.max_pixels_per_image,
                    gates=dict(e6.DEFAULT_GATES), calibration_strengths=list(a2.STRENGTHS),
                    hyperparameter_search=False, comparator_promotion=False,
                    adaptation='Hypotheses follow E7-B0 on this same calibration set; exploratory, not fresh validation',
                    data_use='Calibration files only; no selection/test evaluation',
                    bootstrap='Patient-cluster conditional OOF; no refitting',
                    shared_source_hashes={p.name:e6.file_hash(p) for p in (
                        Path(__file__), ROOT/'e7b0_windows/run_e7b0.py', ROOT/'e7a2_windows/run_e7a2.py',
                        ROOT/'e7a_windows/run_e7a.py', ROOT/'e6_windows/run_e6.py')})
    e6.write_json(output/'protocol.json', protocol)
    shutil.copy2(Path(__file__), output/'executed_source_snapshot.py')
    try:
        inference = e6.resolve(project, args.inference_run)
        mi, ii, records, index = b0.verify_inputs(project, e6.resolve(project,args.manifest_run), inference)
        records = b0.assign_folds(records, args.outer_folds, args.seed)
        e6.write_csv(output/'fold_assignment.csv', [{k:r[k] for k in
            ('sample_id','group_id','declared_group_id','cv_fold')} for r in records])
        rows, labels, audits, paths, models, fold_summary, fold_routing = [], [], [], [], [], [], []
        for outer in range(args.outer_folds):
            training = [r for r in records if r['cv_fold']!=outer]
            heldout = [r for r in records if r['cv_fold']==outer]
            train_groups, heldout_groups = ({r['group_id'] for r in items} for items in (training, heldout))
            if train_groups & heldout_groups:
                raise ValueError('Outer patient leakage')
            print(f'[outer fold {outer+1}/{args.outer_folds}] training={len(training)} heldout={len(heldout)}', flush=True)
            gate, calibrator, fold_labels, fold_audits, fold_paths = fit_pipeline(
                project, training, inference, index, args, outer)
            for audit in fold_audits:
                if set(audit['training_group_ids']) & heldout_groups or set(audit['heldout_group_ids']) & heldout_groups:
                    raise ValueError('Outer heldout patient used to fit labels/calibrator')
            audits.append(dict(outer_fold=outer, outer_training_group_ids=sorted(train_groups),
                               outer_heldout_group_ids=sorted(heldout_groups), fits=fold_audits))
            models.append(dict(outer_fold=outer, gate=gate, calibrator=calibrator))
            labels.extend(fold_labels); paths.extend(fold_paths)
            fold_rows = []
            for r in heldout:
                features, baseline, fused = evaluate_pair(project, r, inference, index, calibrator)
                # Route selection receives no GT/metrics; oracle is a separate diagnostic.
                decision = predict_routes(features, gate, calibrator['calibrator_status'])
                oracle = calibrator['calibrator_status']=='FEASIBLE' and b0.choose_oracle(baseline,fused,True)
                mm = dict(e0=baseline, calibrated_fusion=fused,
                          logistic_safe=fused if decision['logistic_safe'] else baseline,
                          quantile_rule=fused if decision['quantile_rule'] else baseline,
                          oracle_safe=fused if oracle else baseline)
                row = dict(sample_id=r['sample_id'],group_id=r['group_id'],source=r['source'],cv_fold=outer,
                           area_bin=r['area_bin'],gt_area_fraction=r['gt_area_fraction'], **features,
                           logistic_score=decision['logistic_score'],
                           logistic_safe_uses_fusion=decision['logistic_safe'],
                           quantile_rule_uses_fusion=decision['quantile_rule'], oracle_safe_uses_fusion=oracle)
                for model, metrics in mm.items():
                    row.update({model+'_'+k:v for k,v in metrics.items()})
                rows.append(row); fold_rows.append(row)
            fold_summary.extend(dict(cv_fold=outer,**r) for r in b0.flattened(b0.summarize(fold_rows,MODELS)))
            for model in ('logistic_safe','quantile_rule','oracle_safe'):
                selected = [r for r in fold_rows if r[model+'_uses_fusion']]
                fold_routing.append(dict(cv_fold=outer, model=model,images=len(fold_rows),
                    routed_images=len(selected),routed_patients=len({r['group_id'] for r in selected}),
                    true_safe_images=sum(r['oracle_safe_uses_fusion'] for r in selected),
                    routed_dice_harm=sum(r[model+'_dice']<r['e0_dice'] for r in selected),
                    routed_mae_harm=sum(r[model+'_mae']>r['e0_mae'] for r in selected)))
            e6.write_csv(output/'oof_per_image.csv',rows)
            e6.write_json(output/'fold_models.json',models)
            e6.write_json(output/'fit_audit.json',audits)
            e6.write_csv(output/'crossfit_training_labels.csv',labels)
            e6.write_csv(output/'calibration_paths.csv',paths)
            print(f'[outer fold {outer+1} complete] labels={gate["positive_labels"]}/{len(training)} '
                  f'logistic_routes={sum(r["logistic_safe_uses_fusion"] for r in fold_rows)}',flush=True)
        if len(rows)!=len(records) or len({r['sample_id'] for r in rows})!=len(records):
            raise ValueError('OOF coverage mismatch')
        summary=b0.summarize(rows,MODELS)
        e6.write_csv(output/'oof_summary.csv',b0.flattened(summary))
        e6.write_csv(output/'fold_summary.csv',fold_summary)
        e6.write_csv(output/'fold_routing.csv',fold_routing)
        e6.write_csv(output/'prediction_features.csv', [{k:r[k] for k in
            ('sample_id','group_id','cv_fold')+FEATURES} for r in rows])
        e6.write_csv(output/'oof_by_area.csv', [dict(area_bin=area,**s) for area in
            sorted({r['area_bin'] for r in rows}) for s in b0.flattened(b0.summarize(
                [r for r in rows if r['area_bin']==area],MODELS))])
        bootstrap={m:a1.cluster_bootstrap([dict(group_id=r['group_id'],delta_dice_vs_e0=
            r[m+'_dice']-r['e0_dice']) for r in rows],args.seed,args.bootstrap_repetitions) for m in MODELS[1:]}
        e6.write_json(output/'patient_bootstrap.json',bootstrap)
        primary=next(s for s in summary if s['model']=='logistic_safe')
        comparator=next(s for s in summary if s['model']=='quantile_rule')
        status='OOF_PASS_DEVELOPMENT_ONLY' if primary['passes'] else 'OOF_NO_PASS'
        decision=dict(status=status,kind='ADAPTIVE_CALIBRATION_CV_NOT_FORMAL_SELECTION',
            primary_policy='logistic_safe',primary_checks=primary['checks'],
            comparator_checks=comparator['checks'],comparator_promoted=False,
            primary_passing_folds=sum(r['model']=='logistic_safe' and r['passes'] for r in fold_summary),
            primary_routed_images=sum(r['logistic_safe_uses_fusion'] for r in rows),
            all_outer_calibrators_feasible=all(m['calibrator']['calibrator_status']=='FEASIBLE' for m in models),
            formal_test_allowed=False,selection_evaluated=False,
            next_step='Review fold robustness before any full-calibration refit/selection' if primary['passes'] else
                      'Record negative result; do not relax gates or auto-promote comparator',
            limitations=['Same calibration data informed E7-B0 hypotheses; CV is adaptive development.',
                        'Only one source and polyp-positive images.',
                        'Bootstrap does not refit calibrators or gate models.',
                        'Cross-fitted label calibrators use fewer training patients than final outer calibrators.',
                        'Single-class/nonconverged logistic fits conservatively route all images to E0.'])
        e6.write_json(output/'cv_decision.json',decision)
        info.update(status='COMPLETED',finished_at=e6.utc_now(),processed_count=len(rows),
                    patient_count=len({r['group_id'] for r in rows}),outer_folds=args.outer_folds,
                    inner_folds=args.inner_folds,calibrator_fit_count=len(models)*(args.inner_folds+1),
                    selection_images_evaluated=0,test_images_evaluated=0,cv_status=status,
                    manifest_signature=mi['manifest_signature'],
                    inference_info_sha256=e6.file_hash(inference/'run_info.json'),
                    e0_checkpoint_sha256=ii['e0_checkpoint_sha256'],e4_checkpoint_sha256=ii['e4_checkpoint_sha256'])
        e6.write_json(output/'run_info.json',info)
        print('E7-B1 complete:',status,flush=True)
        print('Output:',output,flush=True)
    except BaseException as error:
        info.update(status='FAILED',finished_at=e6.utc_now(),error=f'{type(error).__name__}: {error}')
        e6.write_json(output/'run_info.json',info)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default=str(ROOT))
    p.add_argument('--manifest-run',default='runs/e6_development_seed42')
    p.add_argument('--inference-run',default='runs/e6_development_inference_seed42')
    p.add_argument('--output',default='runs/e7b1_gate_cv_seed42')
    p.add_argument('--outer-folds',type=int,default=5)
    p.add_argument('--inner-folds',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-pixels-per-image',type=int,default=8192)
    p.add_argument('--bootstrap-repetitions',type=int,default=20000)
    return p


if __name__=='__main__':
    p=parser(); args=p.parse_args()
    if min(args.outer_folds,args.inner_folds)<2 or args.seed<0 or args.max_pixels_per_image<256 or args.bootstrap_repetitions<1000:
        p.error('Invalid folds/seed/sample/bootstrap settings')
    run(args)
