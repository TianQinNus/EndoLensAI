"""E7-A2: calibration-only MAE-constrained affine shrinkage and raw-scale thresholds."""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'e7a_windows'))
import run_e7a as a1
e6 = a1.e6

ALPHAS = a1.ALPHAS
RAW_THRESHOLDS = e6.THRESHOLDS
STRENGTHS = tuple(i / 10 for i in range(11))
EXPERIMENT = 'E7A2_MAE_CONSTRAINED_POST_FUSION'


def transform(p, slope, intercept):
    # Preserve the identity exactly, including stored 0/1 endpoints.
    if slope == 1.0 and intercept == 0.0:
        return p.copy()
    return a1.affine_calibrate(p, slope, intercept)


def mapped_threshold(raw_threshold, slope, intercept):
    return float(a1.sigmoid(slope * e6.probability_to_logit(np.array(raw_threshold)) + intercept))


def metrics(p, raw, gt, raw_threshold):
    # Classify on the raw scale to avoid float32 rounding at the mapped threshold.
    result = e6.probability_metrics(raw, gt, raw_threshold)
    result['mae'] = float(np.abs(p - gt.astype(np.float32)).mean())
    return result


def load_pair(project, record, inference, index):
    gt = e6.read_mask(e6.safe_under(project, record['mask']))
    pair = [e6.load_probability(e6.probability_path(inference, index, m, record['sample_id']), gt.shape)
            for m in ('e0', 'e4')]
    return pair[0], pair[1], gt


def path_parameters(fit, strength):
    return 1 + strength * (fit['slope'] - 1), strength * fit['intercept']


def pick_calibrator(rows, limit):
    feasible = [r for r in rows if r['calibration_mae'] <= limit]
    return min(feasible, key=lambda r: (r['sampled_nll'], r['strength'])) if feasible else None


def check_prior(path, manifest_info, inference_info):
    info = e6.read_json(path / 'run_info.json')
    config = e6.read_json(path / 'selection_config.json')
    sig = config.pop('config_signature')
    if e6.signature(config) != sig:
        raise ValueError('E7-A config signature changed')
    if info.get('status') != 'COMPLETED' or config.get('validation_status') != 'NO_PASS':
        raise ValueError('Expected completed E7-A NO_PASS')
    if config['development_manifest_signature'] != manifest_info['manifest_signature']:
        raise ValueError('Prior development manifest differs')
    for model in ('e0', 'e4'):
        if config[model + '_checkpoint_sha256'] != inference_info[model + '_checkpoint_sha256']:
            raise ValueError('Prior checkpoint differs')


def command_select(args):
    project = Path(args.project).resolve()
    output = e6.resolve(project, args.output)
    if output.exists():
        raise FileExistsError('Output already exists: ' + str(output))
    manifest_run = e6.resolve(project, args.manifest_run)
    inference = e6.resolve(project, args.inference_run)
    prior = e6.resolve(project, args.prior_run)
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / 'executed_source_snapshot.py')
    info = dict(status='RUNNING', run_kind='E7A2_SELECT', started_at=e6.utc_now())
    e6.write_json(output / 'run_info.json', info)
    protocol = dict(
        experiment=EXPERIMENT, alpha_e4=list(ALPHAS), raw_thresholds=list(RAW_THRESHOLDS),
        strengths=list(STRENGTHS), gates=dict(e6.DEFAULT_GATES),
        constraint='Full-resolution per-image mean calibration MAE <= raw E0 MAE + 0.003',
        fitting='Sampled NLL affine fit; choose lowest sampled NLL feasible point on identity-to-fit path',
        families=['identity', 'constrained_affine'],
        selection_reuse='Adaptive reuse after E6/E7-A; exploratory development, not fresh validation',
        bootstrap='Patient cluster percentile CI after selection; descriptive, not selection-adjusted',
        source_code_sha256=e6.file_hash(Path(__file__)),
        seed=args.seed, max_pixels_per_image=args.max_pixels_per_image,
    )
    # Persist the revision before evaluating any new candidate.
    e6.write_json(output / 'protocol.json', protocol)
    try:
        mi, records = e6.verify_development_manifest(project, manifest_run)
        ii, index = e6.verify_inference(project, mi, records, inference)
        check_prior(prior, mi, ii)
        calibration = [r for r in records if r['split'] == 'calibration']
        selection = [r for r in records if r['split'] == 'selection']
        if not calibration or not selection:
            raise ValueError('Both development partitions are required')
        if {r['group_id'] for r in calibration} & {r['group_id'] for r in selection}:
            raise ValueError('Patient group leakage')
        s0, s4, y = a1.calibration_samples(project, calibration, inference, index,
                                         args.seed, args.max_pixels_per_image)
        fits = a1.fit_alpha_calibrators(s0, s4, y)
        path_rows = []
        for alpha in ALPHAS:
            raw_sample = ((1-alpha)*s0 + alpha*s4).astype(np.float32)
            logits = e6.probability_to_logit(raw_sample)
            for strength in STRENGTHS:
                slope, intercept = path_parameters(fits[alpha], strength)
                path_rows.append(dict(alpha=alpha, strength=strength, slope=slope, intercept=intercept,
                                      sampled_nll=a1.nll_from_logits(slope*logits + intercept, y),
                                      calibration_mae=0.0))
        e0_mae = []
        # Use every calibration pixel; average image MAEs, not pooled pixel error.
        for pos, record in enumerate(calibration, 1):
            p0, p4, gt = load_pair(project, record, inference, index)
            target = gt.astype(np.float32)
            e0_mae.append(float(np.abs(p0-target).mean()))
            for alpha in ALPHAS:
                raw = ((1-alpha)*p0 + alpha*p4).astype(np.float32)
                logits = e6.probability_to_logit(raw)
                for row in path_rows:
                    if row['alpha'] != alpha:
                        continue
                    p = raw if row['strength'] == 0 else a1.sigmoid(
                        row['slope']*logits + row['intercept']).astype(np.float32)
                    row['calibration_mae'] += float(np.abs(p-target).mean()) / len(calibration)
            if pos % 20 == 0 or pos == len(calibration):
                print(f'[full calibration MAE {pos}/{len(calibration)}]', flush=True)
        baseline_cal_mae = float(np.mean(e0_mae))
        limit = baseline_cal_mae + e6.DEFAULT_GATES['max_mae_increase']
        options = []
        for row in path_rows:
            row['mae_limit'] = limit
            row['feasible'] = row['calibration_mae'] <= limit
        for alpha in ALPHAS:
            rows = [r for r in path_rows if r['alpha'] == alpha]
            identity = dict(rows[0], family='identity')
            options.append(identity)
            picked = pick_calibrator(rows, limit)
            if picked is not None:
                options.append(dict(picked, family='constrained_affine'))
        e6.write_csv(output / 'calibration_path.csv', path_rows)
        e6.write_csv(output / 'calibrators.csv', options)
        e6.write_csv(output / 'unconstrained_fits.csv', [fits[a] for a in ALPHAS])
        e6.write_json(output / 'calibration_constraint.json', dict(
            baseline_e0_mae=baseline_cal_mae, mae_limit=limit,
            full_image_count=len(calibration), selected_on='calibration_only'))
        acc = {(i,t): e6.empty_accumulator() for i in range(len(options)) for t in RAW_THRESHOLDS}
        base = e6.empty_accumulator()
        for pos, record in enumerate(selection, 1):
            p0, p4, gt = load_pair(project, record, inference, index)
            e6.add_metrics(base, record['source'], e6.probability_metrics(p0,gt,0.5))
            for i, option in enumerate(options):
                alpha = option['alpha']
                raw = ((1-alpha)*p0 + alpha*p4).astype(np.float32)
                p = transform(raw, option['slope'], option['intercept'])
                for t in RAW_THRESHOLDS:
                    e6.add_metrics(acc[i,t], record['source'], metrics(p,raw,gt,t))
            if pos % 25 == 0 or pos == len(selection):
                print(f'[selection {pos}/{len(selection)}]', flush=True)
        baseline = e6.finish_accumulator(base)
        candidates = []
        for i, option in enumerate(options):
            for t in RAW_THRESHOLDS:
                candidate = e6.finish_accumulator(acc[i,t])
                candidate.update(alpha=option['alpha'], threshold=t, raw_threshold=t,
                                 calibrated_threshold=mapped_threshold(t,option['slope'],option['intercept']),
                                 slope=option['slope'], intercept=option['intercept'],
                                 strength=option['strength'], family=option['family'])
                e6.gate_candidate(candidate,baseline,e6.DEFAULT_GATES)
                candidates.append(candidate)
        chosen, status = e6.choose_candidate(candidates)
        e6.write_csv(output / 'grid_search.csv', [
            {k:v for k,v in c.items() if k not in ('source_dice','checks')} for c in candidates])
        per_image = []
        for record in selection:
            p0,p4,gt = load_pair(project,record,inference,index)
            raw = ((1-chosen['alpha'])*p0+chosen['alpha']*p4).astype(np.float32)
            p = transform(raw,chosen['slope'],chosen['intercept'])
            m0=e6.probability_metrics(p0,gt,0.5)
            mc=metrics(p,raw,gt,chosen['raw_threshold'])
            row=dict(sample_id=record['sample_id'], group_id=record['group_id'],
                     source=record['source'],area_bin=record['area_bin'],
                     delta_dice_vs_e0=mc['dice']-m0['dice'])
            for key in ('dice','iou','mae'):
                row['e0_'+key]=m0[key]
                row['e7a2_'+key]=mc[key]
            per_image.append(row)
        bootstrap=a1.cluster_bootstrap(per_image,seed=args.seed,repetitions=args.bootstrap_repetitions)
        area_input=[dict(r,**{'e7a_'+k:r['e7a2_'+k] for k in ('dice','iou','mae')}) for r in per_image]
        areas=[{k.replace('e7a_', 'e7a2_'):v for k,v in r.items()} for r in a1.area_summary(area_input)]
        e6.write_csv(output/'selected_per_image.csv',per_image)
        e6.write_csv(output/'selected_by_area.csv',areas)
        e6.write_json(output/'patient_bootstrap.json',bootstrap)
        config=dict(experiment=EXPERIMENT,format_version=1,validation_status=status,
                    alpha_e4=chosen['alpha'],raw_threshold=chosen['raw_threshold'],
                    calibrated_threshold=chosen['calibrated_threshold'],
                    affine_slope=chosen['slope'],affine_intercept=chosen['intercept'],
                    family=chosen['family'],strength=chosen['strength'],gates=dict(e6.DEFAULT_GATES),
                    selection_baseline_e0=baseline,selected_validation_metrics=chosen,
                    patient_bootstrap=bootstrap,protocol=protocol,
                    development_manifest_signature=mi['manifest_signature'],
                    development_manifest_sha256=mi['manifest_sha256'],
                    inference_info_sha256=e6.file_hash(inference/'run_info.json'),
                    prior_config_sha256=e6.file_hash(prior/'selection_config.json'),
                    e0_checkpoint_sha256=ii['e0_checkpoint_sha256'],
                    e4_checkpoint_sha256=ii['e4_checkpoint_sha256'],created_at=e6.utc_now())
        config['config_signature']=e6.signature(config)
        e6.write_json(output/'selection_config.json',config)
        info.update(status='COMPLETED',finished_at=e6.utc_now(),selection_status=status,
                    candidate_count=len(candidates),passing_count=sum(c['passes'] for c in candidates),
                    selected={k:chosen[k] for k in ('alpha','raw_threshold','family','strength','slope','intercept')},
                    config_signature=config['config_signature'])
        e6.write_json(output/'run_info.json',info)
        print('E7-A2 selection status:',status,flush=True)
        print('Output:',output,flush=True)
    except BaseException as error:
        info.update(status='FAILED',finished_at=e6.utc_now(),error=f'{type(error).__name__}: {error}')
        e6.write_json(output/'run_info.json',info)
        raise


def load_locked_config(path):
    c=e6.read_json(path)
    sig=c.pop('config_signature')
    if e6.signature(c)!=sig:
        raise ValueError('Locked configuration changed')
    c['config_signature']=sig
    if c.get('experiment')!=EXPERIMENT or c.get('validation_status')!='PASS':
        raise ValueError('E7-A2 PASS required; formal test blocked')
    if c['gates']!=e6.DEFAULT_GATES:
        raise ValueError('Gates changed')
    return c


def command_apply_test(args):
    project=Path(args.project).resolve()
    config=load_locked_config(e6.resolve(project,args.config))
    output=e6.resolve(project,args.output)
    if output.exists():
        raise FileExistsError('Output already exists: '+str(output))
    e0run=e6.resolve(project,args.e0_run); e4run=e6.resolve(project,args.e4_run)
    e4eval=e6.resolve(project,args.e4_eval)
    manifest,r0,r4,i0,i4=e6.validate_test_sources(project,e0run,e4run,e4eval)
    if config['e0_checkpoint_sha256']!=i0['checkpoint_sha256'] or config['e4_checkpoint_sha256']!=i4['final_checkpoint_sha256']:
        raise ValueError('Checkpoint mismatch')
    output.mkdir(parents=True)
    info=dict(status='RUNNING',run_kind='FULL_E7A2',started_at=e6.utc_now(),
              config_signature=config['config_signature'])
    e6.write_json(output/'run_info.json',info)
    try:
        rows=[]; fingerprints=[]
        for pos,record in enumerate(manifest,1):
            sid=record['sample_id']
            for key in ('image','mask'):
                if e6.file_hash(e6.safe_under(project,record[key]))!=record[key+'_sha256']:
                    raise ValueError('Frozen test changed: '+sid)
            gt=e6.read_mask(e6.safe_under(project,record['mask']))
            paths=[e6.safe_under(root,table[sid]['probability_path']) for root,table in ((e0run,r0),(e4eval,r4))]
            for path,table in zip(paths,(r0,r4)):
                expected=table[sid].get('probability_sha256')
                if expected is not None and e6.file_hash(path)!=expected:
                    raise ValueError('Frozen probability changed: '+sid)
            p0,p4=[e6.load_probability(path,gt.shape) for path in paths]
            alpha=config['alpha_e4']
            raw=((1-alpha)*p0+alpha*p4).astype(np.float32)
            p=transform(raw,config['affine_slope'],config['affine_intercept'])
            result={
                'e0':e6.probability_metrics(p0,gt,0.5),
                'e4':e6.probability_metrics(p4,gt,0.5),
                'e5':e6.probability_metrics((0.5*p0+0.5*p4).astype(np.float32),gt,0.5),
                'e7a2':metrics(p,raw,gt,config['raw_threshold'])}
            # Older E0/E4 CSVs lack probability hashes: verify recorded metrics
            # and record new fingerprints without claiming historical hash verification.
            for model,table in (('e0',r0),('e4',r4)):
                for key in ('dice','iou','mae'):
                    if abs(result[model][key]-float(table[sid][key]))>1e-6:
                        raise ValueError('Recorded baseline metrics differ: '+sid+' '+model+' '+key)
            prel=Path('probabilities')/(sid+'.npz'); mrel=Path('masks')/(sid+'.png')
            for rel in (prel,mrel):
                (output/rel.parent).mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output/prel,probability=p)
            Image.fromarray((raw>=config['raw_threshold']).astype(np.uint8)*255).save(output/mrel)
            row=dict(sample_id=sid,dataset=record['dataset'],probability_path=prel.as_posix(),mask_path=mrel.as_posix())
            for model,mm in result.items():
                row.update({model+'_'+k:v for k,v in mm.items()})
            rows.append(row)
            fingerprints.append(dict(sample_id=sid,e0_probability_sha256=e6.file_hash(paths[0]),
                e4_probability_sha256=e6.file_hash(paths[1]),e7a2_probability_sha256=e6.file_hash(output/prel)))
            if pos%50==0 or pos==len(manifest):
                print(f'[test {pos}/{len(manifest)}]',flush=True)
        summary=[]
        for source in ('ALL',)+e6.DATASETS:
            group=rows if source=='ALL' else [r for r in rows if r['dataset']==source]
            item=dict(dataset=source,count=len(group))
            for model in ('e0','e4','e5','e7a2'):
                for metric in ('dice','iou','mae'):
                    item[model+'_'+metric]=float(np.mean([r[model+'_'+metric] for r in group]))
                item[model+'_dice_lt05']=sum(r[model+'_dice']<0.5 for r in group)
                item[model+'_zero_dice']=sum(r[model+'_dice']==0 for r in group)
            for ref in ('e0','e5'):
                for metric in ('dice','iou','mae'):
                    item['delta_'+metric+'_vs_'+ref]=item['e7a2_'+metric]-item[ref+'_'+metric]
            summary.append(item)
        allrow=summary[0]; gates=config['gates']
        checks=dict(mean_dice_gain=allrow['delta_dice_vs_e0']>=gates['min_mean_dice_gain'],
                    all_datasets=min(r['delta_dice_vs_e0'] for r in summary[1:])>=gates['min_source_dice_delta'],
                    mae=allrow['delta_mae_vs_e0']<=gates['max_mae_increase'],
                    failure_tail=allrow['e7a2_dice_lt05']-allrow['e0_dice_lt05']<=gates['max_dice_lt05_increase'])
        decision=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,gates=gates,
                      paired_image_bootstrap=e6.paired_bootstrap([r['e7a2_dice']-r['e0_dice'] for r in rows]),
                      limitation='Image bootstrap; test patient IDs unavailable. Old probability CSVs lack historical hashes; baseline metrics audited and current fingerprints recorded.')
        e6.write_csv(output/'per_image.csv',rows); e6.write_csv(output/'summary.csv',summary)
        e6.write_csv(output/'input_fingerprints.csv',fingerprints)
        e6.write_json(output/'locked_config.json',config); e6.write_json(output/'decision.json',decision)
        e6.write_json(output/'manifest.json',manifest)
        info.update(status='COMPLETED',finished_at=e6.utc_now(),processed_count=len(rows),decision=decision['status'])
        e6.write_json(output/'run_info.json',info)
    except BaseException as error:
        info.update(status='FAILED',finished_at=e6.utc_now(),error=f'{type(error).__name__}: {error}')
        e6.write_json(output/'run_info.json',info)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default=str(ROOT))
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('select')
    s.add_argument('--manifest-run',default='runs/e6_development_seed42')
    s.add_argument('--inference-run',default='runs/e6_development_inference_seed42')
    s.add_argument('--prior-run',default='runs/e7a_selection_seed42')
    s.add_argument('--output',default='runs/e7a2_selection_seed42')
    s.add_argument('--seed',type=int,default=42)
    s.add_argument('--max-pixels-per-image',type=int,default=8192)
    s.add_argument('--bootstrap-repetitions',type=int,default=20000)
    t=sub.add_parser('apply-test')
    t.add_argument('--config',default='runs/e7a2_selection_seed42/selection_config.json')
    t.add_argument('--output',default='runs/e7a2_final_seed42')
    t.add_argument('--e0-run',default='runs/e0_20261004_221345_872444')
    t.add_argument('--e4-run',default='runs/e4_final_seed42')
    t.add_argument('--e4-eval',default='runs/e4_final_seed42/evaluation_cuda_20261006_235221_022843')
    return p


if __name__=='__main__':
    p=parser(); args=p.parse_args()
    if args.command=='select' and (args.seed<0 or args.max_pixels_per_image<256 or args.bootstrap_repetitions<1000):
        p.error('Invalid sampling/bootstrap arguments')
    {'select':command_select,'apply-test':command_apply_test}[args.command](args)
