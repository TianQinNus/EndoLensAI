"""E7-B2: freeze calibration-median hard routing, then evaluate selection once.

An adaptive development experiment, not fresh external validation. No formal-test
entry point, grid search, logistic fit or post-selection configuration mutation.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'e7b0_windows'))
import run_e7b0 as b0
a2, a1, e6 = b0.a2, b0.a1, b0.e6
EXPERIMENT = 'E7B2_FROZEN_CALIBRATION_MEDIAN_RULE'
MODELS = ('e0', 'calibrated_fusion', 'e7b2')
SOURCES = (Path(__file__), ROOT/'e7b0_windows/run_e7b0.py',
           ROOT/'e7a2_windows/run_e7a2.py', ROOT/'e7a_windows/run_e7a.py', ROOT/'e6_windows/run_e6.py')


def source_hashes():
    return {p.name:e6.file_hash(p) for p in SOURCES}


def median_rule(features):
    if not features:
        raise ValueError('No calibration predictions')
    x=np.array([[r['agreement_iou'],r['e4_predicted_area']] for r in features],dtype=np.float64)
    if not np.isfinite(x).all() or x.min()<0 or x.max()>1:
        raise ValueError('Invalid rule features')
    return dict(rule_iou_max=float(np.median(x[:,0])),rule_e4_area_max=float(np.median(x[:,1])),
                quantile=0.5,definition='unweighted_image_numpy_median',operator='and_lte')


def route(features, config):
    rule=config['rule']
    return bool(config['calibrator']['calibrator_status']=='FEASIBLE'
                and features['agreement_iou']<=rule['rule_iou_max']
                and features['e4_predicted_area']<=rule['rule_e4_area_max'])


def apply_prediction(p0,p4,config):
    """Mask-free inference. Probabilities and binary masks use different scales."""
    for p in (p0,p4):
        if p.ndim!=2 or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
            raise ValueError('Invalid expert probability')
    if p0.shape!=p4.shape:
        raise ValueError('Expert shape mismatch')
    features=b0.predicted_features(p0,p4)
    selected=route(features,config)
    if selected:
        raw=(0.5*p0+0.5*p4).astype(np.float32)
        cal=config['calibrator']
        probability=a2.transform(raw,cal['slope'],cal['intercept'])
        mask=raw>=0.5
    else:
        probability,mask=p0.copy(),p0>=0.5
    return probability,mask,features,selected


def mask_probability_metrics(probability,mask,gt):
    result=e6.probability_metrics(mask.astype(np.float32),gt,0.5)
    result['mae']=float(np.abs(probability-gt.astype(np.float32)).mean())
    return result


def check_prior(prior,mi,ii):
    info=e6.read_json(prior/'run_info.json')
    decision=e6.read_json(prior/'cv_decision.json')
    if (info.get('status')!='COMPLETED' or info.get('run_kind')!='E7B1_CALIBRATION_GATE_CV'
            or decision.get('primary_policy')!='logistic_safe'
            or decision.get('comparator_promoted') is not False
            or set(decision.get('comparator_checks',{}))!=set(('mean_dice_gain','worst_source','mae','failure_tail'))
            or not all(decision['comparator_checks'].values())):
        raise ValueError('Expected completed E7-B1 with passing unpromoted rule comparator')
    if info['manifest_signature']!=mi['manifest_signature']:
        raise ValueError('Prior manifest differs')
    for model in ('e0','e4'):
        if info[model+'_checkpoint_sha256']!=ii[model+'_checkpoint_sha256']:
            raise ValueError('Prior expert checkpoint differs')
    return dict(prior_cv_status=info['cv_status'],prior_comparator_promoted=False,
                info_sha256=e6.file_hash(prior/'run_info.json'),
                decision_sha256=e6.file_hash(prior/'cv_decision.json'))


def signed(config):
    return dict(config,config_signature=e6.signature(config))


def load_locked(output):
    config=e6.read_json(output/'locked_config.json')
    unsigned=dict(config); signature=unsigned.pop('config_signature')
    if e6.signature(unsigned)!=signature:
        raise ValueError('Locked config signature mismatch')
    frozen=e6.read_json(output/'freeze_info.json')
    if (frozen.get('status')!='COMPLETED' or frozen.get('config_signature')!=signature
            or frozen.get('config_sha256')!=e6.file_hash(output/'locked_config.json')):
        raise ValueError('Frozen configuration changed or freeze incomplete')
    if (config.get('experiment')!=EXPERIMENT or config.get('validation_status')!='NOT_EVALUATED'
            or config.get('alpha_e4')!=0.5 or config.get('raw_threshold')!=0.5
            or config.get('e0_threshold')!=0.5 or config.get('gates')!=e6.DEFAULT_GATES
            or config.get('formal_test_allowed') is not False):
        raise ValueError('Protocol parameters changed')
    rule=config['rule']; cal=config['calibrator']
    if rule['quantile']!=0.5 or rule['definition']!='unweighted_image_numpy_median' or rule['operator']!='and_lte':
        raise ValueError('Rule definition changed')
    for key in ('rule_iou_max','rule_e4_area_max'):
        if not np.isfinite(rule[key]) or not 0<=rule[key]<=1:
            raise ValueError('Invalid rule cutpoint')
    if (not np.isfinite(cal['slope']) or not a1.SLOPE_BOUNDS[0]<=cal['slope']<=a1.SLOPE_BOUNDS[1]
            or not np.isfinite(cal['intercept']) or not a1.INTERCEPT_BOUNDS[0]<=cal['intercept']<=a1.INTERCEPT_BOUNDS[1]
            or cal['strength'] not in a2.STRENGTHS):
        raise ValueError('Invalid frozen calibration parameters')
    if config['shared_source_hashes']!=source_hashes():
        raise ValueError('Shared implementation changed after freeze')
    if e6.file_hash(output/'executed_source_snapshot.py')!=config['shared_source_hashes']['run_e7b2.py']:
        raise ValueError('Executed source snapshot changed')
    for name,key in (('calibration_features.csv','calibration_features_sha256'),
                     ('calibration_path.csv','calibration_path_sha256')):
        if e6.file_hash(output/name)!=config[key]:
            raise ValueError('Frozen calibration artifact changed: '+name)
    if e6.file_hash(output/'protocol.json')!=config['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    return config


def freeze(args):
    project=Path(args.project).resolve(); output=e6.resolve(project,args.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite: '+str(output))
    output.mkdir(parents=True)
    info=dict(status='RUNNING',run_kind='E7B2_FREEZE',started_at=e6.utc_now(),
              selection_images_evaluated=0,test_images_evaluated=0)
    e6.write_json(output/'run_info.json',info)
    protocol=dict(experiment=EXPERIMENT,primary_policy='quantile_rule',candidate_count=1,
        rule='IoU <= calibration image median AND E4 predicted area <= calibration image median',
        alpha_e4=0.5,raw_threshold=0.5,e0_threshold=0.5,gates=dict(e6.DEFAULT_GATES),
        calibration_strengths=list(a2.STRENGTHS),seed=args.seed,max_pixels_per_image=args.max_pixels_per_image,
        calibration='Same E7-B0 constrained affine path, fitted on all calibration images',
        fitting='No logistic gate, no grid search, no parameter change after selection',
        phases='Freeze without reading selection files; evaluate one candidate exactly once',
        adaptation='Rule nominated after E7-B1 comparator; selection previously reused by E6/E7-A/E7-A2',
        validation_scope='Adaptive development confirmation, not new independent external validation',
        formal_test_allowed=False,shared_source_hashes=source_hashes())
    e6.write_json(output/'protocol.json',protocol)
    shutil.copy2(Path(__file__),output/'executed_source_snapshot.py')
    try:
        manifest=e6.resolve(project,args.manifest_run); inference=e6.resolve(project,args.inference_run)
        mi,ii,calibration,index=b0.verify_inputs(project,manifest,inference)
        prior=check_prior(e6.resolve(project,args.prior_run),mi,ii)
        features=[]
        for pos,r in enumerate(calibration,1):
            p0,p4,_=a2.load_pair(project,r,inference,index)
            features.append(dict(sample_id=r['sample_id'],group_id=r['group_id'],**b0.predicted_features(p0,p4)))
            if pos%40==0 or pos==len(calibration):
                print(f'[calibration rule features {pos}/{len(calibration)}]',flush=True)
        rule=median_rule(features)
        selected,fit,path,baseline,status=b0.fit_fold(
            project,calibration,inference,index,args.seed,args.max_pixels_per_image,0.5)
        calibrator=dict(selected,calibrator_status=status,training_e0_mae=baseline,unconstrained_fit=fit)
        e6.write_csv(output/'calibration_features.csv',features)
        e6.write_csv(output/'calibration_path.csv',path)
        all_records=e6.read_json(manifest/'development_manifest.json')
        selection=[r for r in all_records if r['split']=='selection']
        if not selection:
            raise ValueError('Selection metadata absent')
        config=signed(dict(experiment=EXPERIMENT,format_version=1,created_at=e6.utc_now(),
            validation_status='NOT_EVALUATED',formal_test_allowed=False,
            rule=rule,calibrator=calibrator,alpha_e4=0.5,raw_threshold=0.5,e0_threshold=0.5,
            gates=dict(e6.DEFAULT_GATES),seed=args.seed,max_pixels_per_image=args.max_pixels_per_image,
            candidate_count=1,calibration_images=len(calibration),
            calibration_patients=len({r['group_id'] for r in calibration}),
            calibration_sample_ids=[r['sample_id'] for r in calibration],
            calibration_group_ids=sorted({r['group_id'] for r in calibration}),
            expected_selection_images=len(selection),expected_selection_patients=len({r['group_id'] for r in selection}),
            manifest_run=args.manifest_run,inference_run=args.inference_run,prior_run=args.prior_run,
            manifest_signature=mi['manifest_signature'],manifest_sha256=mi['manifest_sha256'],
            inference_info_sha256=e6.file_hash(inference/'run_info.json'),
            inference_index_sha256=e6.file_hash(inference/'per_image.csv'),
            e0_checkpoint_sha256=ii['e0_checkpoint_sha256'],e4_checkpoint_sha256=ii['e4_checkpoint_sha256'],
            calibration_features_sha256=e6.file_hash(output/'calibration_features.csv'),
            calibration_path_sha256=e6.file_hash(output/'calibration_path.csv'),
            protocol_sha256=e6.file_hash(output/'protocol.json'),shared_source_hashes=source_hashes(),
            prior=prior,limitation=protocol['adaptation']))
        e6.write_json(output/'locked_config.json',config)
        frozen=dict(status='COMPLETED',run_kind='E7B2_FREEZE',finished_at=e6.utc_now(),
                    config_signature=config['config_signature'],config_sha256=e6.file_hash(output/'locked_config.json'),
                    selection_files_read=0,test_files_read=0,calibrator_status=status)
        e6.write_json(output/'freeze_info.json',frozen)
        info.update(status='FROZEN',finished_at=frozen['finished_at'],config_signature=config['config_signature'])
        e6.write_json(output/'run_info.json',info)
        print('E7-B2 configuration FROZEN:',output/'locked_config.json',flush=True)
        print('Rule:',rule,'calibrator:',status,flush=True)
    except BaseException as error:
        info.update(status='FAILED',finished_at=e6.utc_now(),error=f'{type(error).__name__}: {error}')
        e6.write_json(output/'run_info.json',info)
        raise


def frozen_inputs(project,config):
    manifest=e6.resolve(project,config['manifest_run']); inference=e6.resolve(project,config['inference_run'])
    mi,ii,calibration,index=b0.verify_inputs(project,manifest,inference)
    if (mi['manifest_signature']!=config['manifest_signature'] or mi['manifest_sha256']!=config['manifest_sha256']
            or e6.file_hash(inference/'run_info.json')!=config['inference_info_sha256']
            or e6.file_hash(inference/'per_image.csv')!=config['inference_index_sha256']):
        raise ValueError('Frozen development input changed')
    for model in ('e0','e4'):
        if ii[model+'_checkpoint_sha256']!=config[model+'_checkpoint_sha256']:
            raise ValueError('Frozen expert differs')
    if ({r['sample_id'] for r in calibration}!=set(config['calibration_sample_ids'])
            or {r['group_id'] for r in calibration}!=set(config['calibration_group_ids'])):
        raise ValueError('Frozen calibration membership differs')
    selection=[r for r in e6.read_json(manifest/'development_manifest.json') if r['split']=='selection']
    if (len(selection)!=config['expected_selection_images']
            or len({r['group_id'] for r in selection})!=config['expected_selection_patients']
            or {r['group_id'] for r in selection}&set(config['calibration_group_ids'])):
        raise ValueError('Selection membership/group isolation differs')
    return calibration,selection,inference,index


def check_record_files(project,record,inference,index):
    for kind in ('image','mask'):
        if e6.file_hash(e6.safe_under(project,record[kind]))!=record[kind+'_sha256']:
            raise ValueError('Selection input changed: '+record['sample_id'])
    for model in ('e0','e4'):
        row=index[model,record['sample_id']]
        if e6.file_hash(e6.safe_under(inference,row['probability_path']))!=row['probability_sha256']:
            raise ValueError('Selection probability changed: '+record['sample_id'])


def evaluate_record(project,record,inference,index,config):
    check_record_files(project,record,inference,index)
    p0,p4,gt=a2.load_pair(project,record,inference,index)
    probability,mask,features,selected=apply_prediction(p0,p4,config)
    raw=(0.5*p0+0.5*p4).astype(np.float32)
    cal=config['calibrator']
    fused=a2.transform(raw,cal['slope'],cal['intercept'])
    metrics=dict(e0=e6.probability_metrics(p0,gt,0.5),
                 calibrated_fusion=a2.metrics(fused,raw,gt,0.5),
                 e7b2=mask_probability_metrics(probability,mask,gt))
    row=dict(sample_id=record['sample_id'],group_id=record['group_id'],source=record['source'],
             area_bin=record['area_bin'],gt_area_fraction=record['gt_area_fraction'],uses_fusion=selected,**features)
    for model,result in metrics.items():
        row.update({model+'_'+k:v for k,v in result.items()})
    return row


def evaluate(args):
    project=Path(args.project).resolve(); output=e6.resolve(project,args.run)
    config=load_locked(output)
    info=e6.read_json(output/'run_info.json')
    if info['status']!='FROZEN':
        raise ValueError('Selection evaluation already attempted or run not frozen')
    if config['calibrator']['calibrator_status']!='FEASIBLE':
        raise ValueError('No feasible frozen calibrator; selection blocked')
    destination=output/'selection'
    destination.mkdir()  # Atomic single-attempt claim, before reading selection files.
    info.update(status='RUNNING',run_kind='E7B2_SELECTION',selection_started_at=e6.utc_now())
    e6.write_json(output/'run_info.json',info)
    try:
        _,selection,inference,index=frozen_inputs(project,config)
        rows=[]
        for pos,record in enumerate(selection,1):
            rows.append(evaluate_record(project,record,inference,index,config))
            info['selection_images_evaluated']=len(rows)
            if pos%40==0 or pos==len(selection):
                e6.write_csv(destination/'per_image.csv',rows)
                e6.write_json(output/'run_info.json',info)
                print(f'[fixed selection {pos}/{len(selection)}]',flush=True)
        if len({r['sample_id'] for r in rows})!=len(selection):
            raise ValueError('Duplicate selection image')
        summary=b0.summarize(rows,MODELS)
        e6.write_csv(destination/'summary.csv',b0.flattened(summary))
        e6.write_csv(destination/'by_area.csv',[dict(area_bin=area,**s) for area in
            sorted({r['area_bin'] for r in rows}) for s in b0.flattened(b0.summarize(
                [r for r in rows if r['area_bin']==area],MODELS))])
        e6.write_csv(destination/'by_source.csv',[dict(source=source,**s) for source in
            sorted({r['source'] for r in rows}) for s in b0.flattened(b0.summarize(
                [r for r in rows if r['source']==source],MODELS))])
        patients=[]
        for group_id in sorted({r['group_id'] for r in rows}):
            group=[r for r in rows if r['group_id']==group_id]
            patient=dict(group_id=group_id,count=len(group),routed_images=sum(r['uses_fusion'] for r in group))
            for model in MODELS:
                for metric in ('dice','iou','mae'):
                    patient[model+'_'+metric]=float(np.mean([r[model+'_'+metric] for r in group]))
            patient.update(delta_dice_vs_e0=patient['e7b2_dice']-patient['e0_dice'],
                           delta_mae_vs_e0=patient['e7b2_mae']-patient['e0_mae'])
            patients.append(patient)
        e6.write_csv(destination/'by_patient.csv',patients)
        routed=[r for r in rows if r['uses_fusion']]
        routing=dict(images=len(rows),routed_images=len(routed),route_fraction=len(routed)/len(rows),
            routed_patients=len({r['group_id'] for r in routed}),
            routed_dice_harm=sum(r['e7b2_dice']<r['e0_dice'] for r in routed),
            routed_mae_harm=sum(r['e7b2_mae']>r['e0_mae'] for r in routed),
            routed_strict_safe=sum(r['e7b2_dice']>r['e0_dice'] and r['e7b2_mae']<=r['e0_mae'] for r in routed),
            patient_equal_mean_dice_gain=float(np.mean([r['delta_dice_vs_e0'] for r in patients])),
            patient_equal_mean_mae_delta=float(np.mean([r['delta_mae_vs_e0'] for r in patients])),
            patients_dice_improved=sum(r['delta_dice_vs_e0']>0 for r in patients),
            patients_dice_worse=sum(r['delta_dice_vs_e0']<0 for r in patients))
        e6.write_json(destination/'routing.json',routing)
        bootstrap=a1.cluster_bootstrap([dict(group_id=r['group_id'],delta_dice_vs_e0=
            r['e7b2_dice']-r['e0_dice']) for r in rows],config['seed'],args.bootstrap_repetitions)
        e6.write_json(destination/'patient_bootstrap.json',bootstrap)
        primary=next(r for r in summary if r['model']=='e7b2')
        status='PASS' if primary['passes'] else 'NO_PASS'
        decision=dict(status=status,scope='ADAPTIVE_DEVELOPMENT_SELECTION_ONLY',candidate_count=1,
            checks=primary['checks'],gates=config['gates'],config_signature=config['config_signature'],
            config_sha256=e6.file_hash(output/'locked_config.json'),candidate_metrics=primary,
            baseline_metrics=next(r for r in summary if r['model']=='e0'),
            calibration_group_overlap=0,formal_test_allowed=False,
            limitation='Selection previously reused; single positive-only source; bootstrap conditional, not selection-adjusted.',
            next_step='Review robustness and obtain authorization before any separate formal-test implementation' if status=='PASS'
                      else 'Preserve negative result; do not change rule or gates and rerun selection')
        e6.write_json(destination/'decision.json',decision)
        # The parameter file stays byte-for-byte unchanged after observing selection.
        load_locked(output)
        info.update(status='COMPLETED',finished_at=e6.utc_now(),selection_status=status,
                    selection_images_evaluated=len(rows),selection_patients=len(patients),
                    test_images_evaluated=0,candidate_count=1,routed_images=len(routed))
        e6.write_json(output/'run_info.json',info)
        print('E7-B2 selection:',status,'formal test NOT executed',flush=True)
        print('Output:',output,flush=True)
    except BaseException as error:
        info.update(status='FAILED',finished_at=e6.utc_now(),error=f'{type(error).__name__}: {error}')
        e6.write_json(output/'run_info.json',info)
        raise


def verify(args):
    """Read-only reproduction of an existing evaluation, not a new candidate test."""
    project=Path(args.project).resolve(); output=e6.resolve(project,args.run)
    config=load_locked(output); info=e6.read_json(output/'run_info.json')
    if info['status']!='COMPLETED' or info['test_images_evaluated']!=0:
        raise ValueError('Completed development evaluation required')
    calibration,selection,inference,index=frozen_inputs(project,config)
    features=[]; baseline_maes=[]; fused_maes=[]
    for r in calibration:
        p0,p4,gt=a2.load_pair(project,r,inference,index)
        features.append(b0.predicted_features(p0,p4))
        baseline_maes.append(e6.probability_metrics(p0,gt)['mae'])
        raw=(0.5*p0+0.5*p4).astype(np.float32)
        fused_maes.append(float(np.abs(a2.transform(raw,config['calibrator']['slope'],
            config['calibrator']['intercept'])-gt.astype(np.float32)).mean()))
    if median_rule(features)!=config['rule']:
        raise ValueError('Frozen medians differ from calibration predictions')
    cal=config['calibrator']
    if (abs(float(np.mean(baseline_maes))-cal['training_e0_mae'])>1e-7
            or abs(float(np.mean(fused_maes))-cal['calibration_mae'])>1e-7
            or cal['calibration_mae']>cal['mae_limit']):
        raise ValueError('Frozen calibration MAE constraint differs')
    path=e6.read_csv(output/'calibration_path.csv')
    numeric=[{k:float(r[k]) for k in ('strength','slope','intercept','sampled_nll','calibration_mae')} for r in path]
    chosen=a2.pick_calibrator(numeric,cal['mae_limit'])
    if chosen is None or any(chosen[k]!=cal[k] for k in chosen):
        raise ValueError('Frozen calibrator is not the predeclared feasible NLL choice')
    stored=e6.read_csv(output/'selection/per_image.csv')
    table={r['sample_id']:r for r in stored}
    if len(table)!=len(stored) or set(table)!={r['sample_id'] for r in selection}:
        raise ValueError('Selection output coverage mismatch')
    rows=[]; maximum=0.0
    for pos,record in enumerate(selection,1):
        actual=evaluate_record(project,record,inference,index,config); previous=table[record['sample_id']]
        if actual['uses_fusion']!=(previous['uses_fusion']=='True'):
            raise ValueError('Saved route differs from prediction-only replay')
        for key,value in actual.items():
            if key in ('sample_id','group_id','source','area_bin','uses_fusion'):
                continue
            error=abs(float(value)-float(previous[key])); maximum=max(maximum,error)
            if error>1e-7:
                raise ValueError('Saved metric/feature differs: '+key)
        rows.append(actual)
        if pos%40==0 or pos==len(selection):
            print(f'[selection replay {pos}/{len(selection)}]',flush=True)
    result=next(r for r in b0.summarize(rows,MODELS) if r['model']=='e7b2')
    decision=e6.read_json(output/'selection/decision.json')
    if (result['checks']!=decision['checks'] or ('PASS' if result['passes'] else 'NO_PASS')!=decision['status']
            or decision['config_sha256']!=e6.file_hash(output/'locked_config.json')
            or decision['formal_test_allowed'] is not False):
        raise ValueError('Saved decision differs')
    print(dict(status='VERIFIED',calibration_images=len(calibration),selection_images=len(selection),
               selection_patients=len({r['group_id'] for r in selection}),max_replay_error=maximum,
               frozen_config_unchanged=True,test_images_evaluated=0),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default=str(ROOT))
    sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze')
    f.add_argument('--manifest-run',default='runs/e6_development_seed42')
    f.add_argument('--inference-run',default='runs/e6_development_inference_seed42')
    f.add_argument('--prior-run',default='runs/e7b1_gate_cv_seed42')
    f.add_argument('--output',default='runs/e7b2_rule_selection_seed42')
    f.add_argument('--seed',type=int,default=42)
    f.add_argument('--max-pixels-per-image',type=int,default=8192)
    for command in ('evaluate','verify'):
        c=sub.add_parser(command)
        c.add_argument('--run',default='runs/e7b2_rule_selection_seed42')
        if command=='evaluate':
            c.add_argument('--bootstrap-repetitions',type=int,default=20000)
    return p


if __name__=='__main__':
    p=parser(); args=p.parse_args()
    if args.command=='freeze' and (args.seed<0 or args.max_pixels_per_image<256):
        p.error('Invalid calibration sampling settings')
    if args.command=='evaluate' and args.bootstrap_repetitions<1000:
        p.error('At least 1000 bootstrap repetitions required')
    {'freeze':freeze,'evaluate':evaluate,'verify':verify}[args.command](args)
