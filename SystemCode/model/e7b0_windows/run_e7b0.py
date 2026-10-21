"""Calibration-only E7-B0 regional diagnostics and patient-grouped cross-validation.

No selection/test probability or ground-truth files are loaded. No deployable gate
is fitted: oracle routes use ground truth solely to diagnose available headroom.
"""
import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'e7a2_windows'))
import run_e7a2 as a2
a1, e6 = a2.a1, a2.e6
MODELS = ('e0', 'e4', 'raw_fusion', 'calibrated_fusion', 'oracle_dice', 'oracle_safe')
REGIONS = ('foreground_core', 'foreground_boundary', 'background_boundary', 'background_far')
FEATURES = ('e0_predicted_area', 'e4_predicted_area', 'agreement_iou',
            'probability_disagreement', 'union_entropy', 'binary_disagreement')


def verify_inputs(project, manifest_run, inference):
    mi = e6.read_json(manifest_run / 'run_info.json')
    path = manifest_run / 'development_manifest.json'
    if mi.get('status') != 'COMPLETED' or mi.get('run_kind') != 'E6_PREPARE':
        raise ValueError('Completed frozen E6 manifest required')
    if e6.file_hash(path) != mi['manifest_sha256']:
        raise ValueError('Manifest hash changed')
    records = e6.read_json(path)
    if e6.signature(records) != mi['manifest_signature']:
        raise ValueError('Manifest signature changed')
    cal = [r for r in records if r['split'] == 'calibration']
    other = [r for r in records if r['split'] != 'calibration']
    if not cal or {r['group_id'] for r in cal} & {r['group_id'] for r in other}:
        raise ValueError('Missing calibration partition or cross-partition group leakage')
    for r in cal:
        for kind in ('image', 'mask'):
            if e6.file_hash(e6.safe_under(project, r[kind])) != r[kind + '_sha256']:
                raise ValueError('Calibration file changed: ' + r['sample_id'])
    ii = e6.read_json(inference / 'run_info.json')
    if (ii.get('status') != 'COMPLETED' or ii.get('run_kind') != 'FULL_DEVELOPMENT'
            or ii.get('processed_count') != len(records)
            or ii.get('manifest_signature') != mi['manifest_signature']):
        raise ValueError('Inference metadata mismatch')
    rows = e6.read_csv(inference / 'per_image.csv')
    index = {(r['model'], r['sample_id']): r for r in rows}
    expected = {(m, r['sample_id']) for r in records for m in ('e0', 'e4')}
    if len(index) != len(rows) or set(index) != expected:
        raise ValueError('Probability index incomplete or duplicated')
    for r in cal:
        for model in ('e0', 'e4'):
            row = index[model, r['sample_id']]
            if e6.file_hash(e6.safe_under(inference, row['probability_path'])) != row['probability_sha256']:
                raise ValueError('Calibration probability changed: ' + r['sample_id'])
    return mi, ii, cal, index


def assign_folds(records, nfolds, seed):
    groups = defaultdict(list)
    for r in records:
        groups[r['group_id']].append(r)
    if len(groups) < nfolds or nfolds < 2:
        raise ValueError('Too few patient groups for requested folds')
    # Balance image counts and group counts; fold construction uses no outcomes.
    rng = np.random.default_rng(seed)
    names = sorted(groups)
    rng.shuffle(names)
    names.sort(key=lambda g: -len(groups[g]))
    loads = [0] * nfolds
    patients = [0] * nfolds
    assignments = {}
    for name in names:
        best = min((loads[i], patients[i]) for i in range(nfolds))
        ties = [i for i in range(nfolds) if (loads[i], patients[i]) == best]
        fold = int(rng.choice(ties))
        assignments[name] = fold
        loads[fold] += len(groups[name])
        patients[fold] += 1
    return [dict(r, cv_fold=assignments[r['group_id']]) for r in records]


def morphology(gt, radius, dilate):
    result = gt.astype(bool).copy()
    for _ in range(radius):
        padded = np.pad(result, 1, mode='constant', constant_values=False)
        h, w = result.shape
        windows = [padded[dy:dy+h, dx:dx+w] for dy in range(3) for dx in range(3)]
        result = np.logical_or.reduce(windows) if dilate else np.logical_and.reduce(windows)
    return result


def regions(gt, radius):
    core = morphology(gt, radius, False)
    expanded = morphology(gt, radius, True)
    return dict(foreground_core=core, foreground_boundary=gt & ~core,
                background_boundary=~gt & expanded, background_far=~expanded)


def predicted_features(p0, p4):
    m0, m4 = p0 >= 0.5, p4 >= 0.5
    union = m0 | m4
    union_count = int(union.sum())
    entropy = []
    for p in (p0, p4):
        q = np.clip(p.astype(np.float64), 1e-6, 1-1e-6)
        h = -(q*np.log(q) + (1-q)*np.log1p(-q))
        entropy.append(float(h[union].mean()) if union_count else float(h.mean()))
    return dict(e0_predicted_area=float(m0.mean()), e4_predicted_area=float(m4.mean()),
                agreement_iou=float((m0 & m4).sum()/union_count) if union_count else 1.0,
                probability_disagreement=float(np.abs(p0-p4).mean()),
                union_entropy=float(np.mean(entropy)), binary_disagreement=float((m0 != m4).mean()))


def choose_oracle(e0metrics, fusedmetrics, safe):
    if fusedmetrics['dice'] <= e0metrics['dice']:
        return False
    if safe and fusedmetrics['mae'] > e0metrics['mae']:
        return False
    return True


def fit_fold(project, training, inference, index, seed, maximum, alpha):
    s0, s4, target = a1.calibration_samples(project, training, inference, index, seed, maximum)
    raw_sample = ((1-alpha)*s0 + alpha*s4).astype(np.float32)
    fit = a1.fit_affine(raw_sample, target)
    logits_sample = e6.probability_to_logit(raw_sample)
    path = []
    for strength in a2.STRENGTHS:
        slope, intercept = a2.path_parameters(fit, strength)
        path.append(dict(strength=strength, slope=slope, intercept=intercept,
                         sampled_nll=a1.nll_from_logits(slope*logits_sample+intercept, target),
                         calibration_mae=0.0))
    baseline_maes = []
    for pos, r in enumerate(training, 1):
        p0, p4, gt = a2.load_pair(project, r, inference, index)
        target = gt.astype(np.float32)
        baseline_maes.append(float(np.abs(p0-target).mean()))
        raw = ((1-alpha)*p0 + alpha*p4).astype(np.float32)
        logits = e6.probability_to_logit(raw)
        for row in path:
            p = raw if row['strength'] == 0 else a1.sigmoid(
                row['slope']*logits+row['intercept']).astype(np.float32)
            row['calibration_mae'] += float(np.abs(p-target).mean()) / len(training)
        if pos % 40 == 0 or pos == len(training):
            print(f'  [training full-image MAE {pos}/{len(training)}]', flush=True)
    baseline = float(np.mean(baseline_maes))
    limit = baseline + e6.DEFAULT_GATES['max_mae_increase']
    for row in path:
        row.update(mae_limit=limit, feasible=row['calibration_mae'] <= limit)
    selected = a2.pick_calibrator(path, limit)
    if selected is None:
        selected = dict(path[0])
        status = 'NO_FEASIBLE_CALIBRATOR_IDENTITY_FALLBACK'
    else:
        status = 'FEASIBLE'
    return dict(selected), fit, path, baseline, status


def regional_rows(sample_id, fold, model, probability, gt, masks):
    error = np.abs(probability-gt.astype(np.float32))
    rows = []
    for name, region in masks.items():
        count = int(region.sum())
        rows.append(dict(sample_id=sample_id, cv_fold=fold, model=model, region=name,
                         pixels=count, area_fraction=float(region.mean()),
                         conditional_mae=float(error[region].mean()) if count else None,
                         image_mae_contribution=float(error[region].sum(dtype=np.float64)/gt.size),
                         mean_probability=float(probability[region].mean()) if count else None))
    return rows


def summarize(rows, models=MODELS):
    result = []
    for model in models:
        scores = [r[model+'_dice'] for r in rows]
        base_scores = [r['e0_dice'] for r in rows]
        item = dict(model=model, count=len(rows), dice=float(np.mean(scores)),
                    iou=float(np.mean([r[model+'_iou'] for r in rows])),
                    mae=float(np.mean([r[model+'_mae'] for r in rows])),
                    dice_lt05=sum(s < 0.5 for s in scores), zero_dice=sum(s == 0 for s in scores),
                    source_dice={s:float(np.mean([r[model+'_dice'] for r in rows if r['source']==s]))
                                 for s in sorted({r['source'] for r in rows})})
        baseline = dict(dice=float(np.mean(base_scores)),
                        mae=float(np.mean([r['e0_mae'] for r in rows])),
                        dice_lt05=sum(s < 0.5 for s in base_scores),
                        source_dice={s:float(np.mean([r['e0_dice'] for r in rows if r['source']==s]))
                                     for s in sorted({r['source'] for r in rows})})
        e6.gate_candidate(item, baseline, e6.DEFAULT_GATES)
        result.append(item)
    return result


def flattened(summary):
    return [{k:v for k,v in row.items() if k not in ('source_dice', 'checks')} for row in summary]


def feature_profiles(rows):
    output = []
    for feature in FEATURES:
        values = np.array([r[feature] for r in rows])
        # Equal values stay in the same bin; fewer than four bins are possible.
        cutpoints = np.unique(np.quantile(values, [0.25,0.5,0.75]))
        ids = np.searchsorted(cutpoints, values, side='right')
        for q in sorted(set(ids.tolist())):
            group = [r for r,i in zip(rows,ids) if i==q]
            output.append(dict(feature=feature, bin=int(q), count=len(group),
                               value_min=min(r[feature] for r in group),
                               value_max=max(r[feature] for r in group),
                               safe_opportunity_count=sum(r['safe_opportunity'] for r in group),
                               mean_fusion_dice_delta=float(np.mean([r['fusion_dice_delta'] for r in group])),
                               mean_fusion_mae_delta=float(np.mean([r['fusion_mae_delta'] for r in group]))))
    return output


def run(args):
    project = Path(args.project).resolve()
    output = e6.resolve(project, args.output)
    if output.exists():
        raise FileExistsError('Output already exists: '+str(output))
    output.mkdir(parents=True)
    info = dict(status='RUNNING',run_kind='E7B0_CALIBRATION_DIAGNOSTICS',started_at=e6.utc_now())
    e6.write_json(output/'run_info.json',info)
    protocol = dict(folds=args.folds,seed=args.seed,alpha_e4=0.5,raw_threshold=0.5,
                    boundary_radius_original_pixels=args.boundary_radius,
                    calibration_strengths=list(a2.STRENGTHS),gates=dict(e6.DEFAULT_GATES),
                    max_pixels_per_image=args.max_pixels_per_image,
                    data_use='Only calibration probability/mask files; selection/test files never loaded',
                    hypothesis='Fixed equal fusion chosen following earlier adaptive development; CV is diagnostic',
                    oracle_dice='Choose fusion only if its OOF Dice exceeds E0; otherwise E0',
                    oracle_safe='Choose fusion only if OOF Dice exceeds E0 AND OOF MAE <= E0',
                    oracle_limit='Safe route is a sufficient opportunity estimate, not the optimal budget-constrained upper bound',
                    no_deployable_gate=True,
                    shared_source_hashes={p.name:e6.file_hash(p) for p in (
                        Path(__file__),ROOT/'e7a2_windows/run_e7a2.py',
                        ROOT/'e7a_windows/run_e7a.py',ROOT/'e6_windows/run_e6.py')})
    e6.write_json(output/'protocol.json',protocol)
    shutil.copy2(Path(__file__),output/'executed_source_snapshot.py')
    try:
        manifest = e6.resolve(project,args.manifest_run)
        inference = e6.resolve(project,args.inference_run)
        mi,ii,calibration,index = verify_inputs(project,manifest,inference)
        records = assign_folds(calibration,args.folds,args.seed)
        e6.write_csv(output/'fold_assignment.csv',[
            {k:r[k] for k in ('sample_id','group_id','declared_group_id','cv_fold','area_bin')}
            for r in records])
        rows=[]; region_rows=[]; fits=[]; paths=[]; fold_summaries=[]; audits=[]
        for fold in range(args.folds):
            training=[r for r in records if r['cv_fold']!=fold]
            heldout=[r for r in records if r['cv_fold']==fold]
            train_groups={r['group_id'] for r in training}
            validation_groups={r['group_id'] for r in heldout}
            if train_groups & validation_groups:
                raise ValueError('Cross-fold patient leakage')
            print(f'[fold {fold+1}/{args.folds}] training={len(training)} heldout={len(heldout)}',flush=True)
            selected,fit,path,baseline_mae,status = fit_fold(
                project,training,inference,index,args.seed,args.max_pixels_per_image,0.5)
            fits.append(dict(cv_fold=fold,training_images=len(training),heldout_images=len(heldout),
                             training_patients=len(train_groups),heldout_patients=len(validation_groups),
                             calibrator_status=status,training_e0_mae=baseline_mae,
                             **{k:selected[k] for k in ('strength','slope','intercept','sampled_nll','calibration_mae','mae_limit')},
                             unconstrained_slope=fit['slope'],unconstrained_intercept=fit['intercept']))
            paths.extend(dict(cv_fold=fold,**r) for r in path)
            audits.append(dict(cv_fold=fold,training_group_ids=sorted(train_groups),
                               heldout_group_ids=sorted(validation_groups),shared_patients=0))
            fold_rows=[]
            for record in heldout:
                p0,p4,gt=a2.load_pair(project,record,inference,index)
                raw=(0.5*p0+0.5*p4).astype(np.float32)
                p=a2.transform(raw,selected['slope'],selected['intercept'])
                mm=dict(e0=e6.probability_metrics(p0,gt,0.5),
                        e4=e6.probability_metrics(p4,gt,0.5),
                        raw_fusion=e6.probability_metrics(raw,gt,0.5),
                        calibrated_fusion=a2.metrics(p,raw,gt,0.5))
                route_dice=choose_oracle(mm['e0'],mm['calibrated_fusion'],False)
                route_safe=choose_oracle(mm['e0'],mm['calibrated_fusion'],True)
                mm['oracle_dice']=mm['calibrated_fusion'] if route_dice else mm['e0']
                mm['oracle_safe']=mm['calibrated_fusion'] if route_safe else mm['e0']
                row=dict(sample_id=record['sample_id'],group_id=record['group_id'],source=record['source'],
                         cv_fold=fold,area_bin=record['area_bin'],gt_area_fraction=record['gt_area_fraction'],
                         oracle_dice_uses_fusion=route_dice,oracle_safe_uses_fusion=route_safe,
                         safe_opportunity=route_safe,
                         fusion_dice_delta=mm['calibrated_fusion']['dice']-mm['e0']['dice'],
                         fusion_mae_delta=mm['calibrated_fusion']['mae']-mm['e0']['mae'],
                         **predicted_features(p0,p4))
                for model,values in mm.items():
                    row.update({model+'_'+k:v for k,v in values.items()})
                masks=regions(gt,args.boundary_radius)
                for model,prob in (('e0',p0),('e4',p4),('raw_fusion',raw),('calibrated_fusion',p)):
                    region_rows.extend(regional_rows(record['sample_id'],fold,model,prob,gt,masks))
                rows.append(row); fold_rows.append(row)
            fold_summaries.extend(dict(cv_fold=fold,**r) for r in flattened(summarize(fold_rows)))
            print(f'[fold {fold+1} complete] strength={selected["strength"]:.1f} status={status}',flush=True)
            # Save progress so diagnostics remain available if a later fold fails.
            e6.write_csv(output/'oof_per_image.csv',rows)
            e6.write_csv(output/'fold_calibrators.csv',fits)
            e6.write_csv(output/'fold_calibration_paths.csv',paths)
        if len(rows)!=len(records) or len({r['sample_id'] for r in rows})!=len(records):
            raise ValueError('OOF coverage is incomplete or duplicated')
        summary=summarize(rows)
        e6.write_csv(output/'oof_summary.csv',flattened(summary))
        e6.write_csv(output/'fold_summary.csv',fold_summaries)
        e6.write_json(output/'fold_audit.json',audits)
        e6.write_csv(output/'region_per_image.csv',region_rows)
        region_summary=[]
        for model in ('e0','e4','raw_fusion','calibrated_fusion'):
            for region in REGIONS:
                group=[r for r in region_rows if r['model']==model and r['region']==region]
                valid=[r for r in group if r['pixels']>0]
                region_summary.append(dict(model=model,region=region,images=len(group),nonempty_images=len(valid),
                    mean_conditional_mae=float(np.mean([r['conditional_mae'] for r in valid])) if valid else None,
                    mean_image_mae_contribution=float(np.mean([r['image_mae_contribution'] for r in group])),
                    mean_area_fraction=float(np.mean([r['area_fraction'] for r in group]))))
        e6.write_csv(output/'region_summary.csv',region_summary)
        area_summary=[]
        for area in ('small_lt1pct','medium_1to10pct','large_ge10pct'):
            group=[r for r in rows if r['area_bin']==area]
            if group:
                area_summary.extend(dict(area_bin=area,**r) for r in flattened(summarize(group)))
        e6.write_csv(output/'oof_by_area.csv',area_summary)
        feature_rows=[{k:r[k] for k in ('sample_id','group_id','cv_fold')+FEATURES} for r in rows]
        e6.write_csv(output/'prediction_features.csv',feature_rows)
        e6.write_csv(output/'feature_opportunity_profiles.csv',feature_profiles(rows))
        bootstrap={}
        for model in MODELS[1:]:
            delta=[dict(group_id=r['group_id'],delta_dice_vs_e0=r[model+'_dice']-r['e0_dice']) for r in rows]
            bootstrap[model]=a1.cluster_bootstrap(delta,seed=args.seed,repetitions=args.bootstrap_repetitions)
        e6.write_json(output/'patient_bootstrap.json',bootstrap)
        safe=next(r for r in summary if r['model']=='oracle_safe')
        dice=next(r for r in summary if r['model']=='oracle_dice')
        fixed=next(r for r in summary if r['model']=='calibrated_fusion')
        if dice['delta_dice_vs_e0']<e6.DEFAULT_GATES['min_mean_dice_gain']:
            recommendation='INSUFFICIENT_DICE_HEADROOM_FOR_THIS_TWO_EXPERT_SET'
        elif safe['passes']:
            recommendation='HEADROOM_PRESENT_TEST_SIMPLE_PREDICTION_ONLY_GATE_IN_E7B1'
        else:
            recommendation='SAFE_ORACLE_INCONCLUSIVE_ANALYZE_DICE_MAE_TRADEOFF_BEFORE_GATE'
        decision=dict(kind='DIAGNOSTIC_ONLY_NOT_FORMAL_VALIDATION',recommendation=recommendation,
                      fixed_fusion_checks=fixed['checks'],oracle_safe_checks=safe['checks'],
                      oracle_safe_uses_fusion=sum(r['oracle_safe_uses_fusion'] for r in rows),
                      oracle_dice_uses_fusion=sum(r['oracle_dice_uses_fusion'] for r in rows),
                      all_folds_feasible=all(r['calibrator_status']=='FEASIBLE' for r in fits),
                      limitations=[
                          'Oracles use heldout ground truth; they are not deployable gates.',
                          'Safe oracle is conservative; failure does not rule out a budget-constrained gate.',
                          'Earlier calibration data and selection outcomes influenced this fixed hypothesis.',
                          'Cluster bootstrap describes OOF deltas conditional on fitted folds; it does not refit models.',
                          'Boundary width is fixed original-image pixels, so spatial scale varies with resolution.',
                          'Only polyp-positive images and one external source are present.'])
        e6.write_json(output/'diagnostic_decision.json',decision)
        info.update(status='COMPLETED',finished_at=e6.utc_now(),processed_count=len(rows),
                    patient_count=len({r['group_id'] for r in rows}),folds=args.folds,
                    selection_images_evaluated=0,test_images_evaluated=0,
                    manifest_signature=mi['manifest_signature'],
                    inference_info_sha256=e6.file_hash(inference/'run_info.json'),
                    e0_checkpoint_sha256=ii['e0_checkpoint_sha256'],
                    e4_checkpoint_sha256=ii['e4_checkpoint_sha256'],
                    recommendation=recommendation)
        e6.write_json(output/'run_info.json',info)
        print('E7-B0 complete:',recommendation,flush=True)
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
    p.add_argument('--output',default='runs/e7b0_diagnostics_seed42')
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--boundary-radius',type=int,default=3)
    p.add_argument('--max-pixels-per-image',type=int,default=8192)
    p.add_argument('--bootstrap-repetitions',type=int,default=20000)
    return p


if __name__=='__main__':
    p=parser(); args=p.parse_args()
    if args.folds<2 or args.seed<0 or args.boundary_radius<1 or args.max_pixels_per_image<256 or args.bootstrap_repetitions<1000:
        p.error('Invalid fold, seed, radius, sampling or bootstrap arguments')
    run(args)
