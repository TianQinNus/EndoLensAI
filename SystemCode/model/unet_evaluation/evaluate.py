"""Frozen validation-selected checkpoint evaluation; no test-based selection."""
import argparse, csv, hashlib, importlib.util, json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(p,y):
    intersection=np.count_nonzero(p & y); denominator=np.count_nonzero(p)+np.count_nonzero(y); union=np.count_nonzero(p | y)
    return (2*intersection/denominator if denominator else 1., intersection/union if union else 1.)
def main():
    a=argparse.ArgumentParser(); a.add_argument('--selection',type=Path,required=True); a.add_argument('--output',type=Path,required=True); args=a.parse_args()
    torch.set_num_threads(4)
    selection=json.loads(args.selection.read_text()); checkpoint=ROOT/selection['checkpoint']; source=ROOT/selection['source']
    assert sha(checkpoint)==selection['checkpoint_sha256'] and sha(source)==selection['source_sha256']
    spec=importlib.util.spec_from_file_location('selected_model',source); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cp=torch.load(checkpoint,map_location='cpu',weights_only=True); config=cp['config']
    assert config['source_sha256']==sha(source)
    manifest=ROOT.parents[1]/'Miscellaneous/model_artifacts/e2_data_seed42/test.json'; records=json.loads(manifest.read_text())
    assert len(records)==798 and len({r['sample_id'] for r in records})==798
    assert Counter(r['dataset'] for r in records)=={'Kvasir':100,'CVC-ClinicDB':62,'CVC-ColonDB':380,'ETIS-LaribPolypDB':196,'CVC-300':60}
    for r in records:
        for key in ['image','mask']:
            assert sha(ROOT/r[key])==r[key+'_sha256'], r[key]
    device=torch.device('mps'); assert torch.backends.mps.is_available()
    model=module.UNet(config['base']).to(device); model.load_state_dict(cp['model']); model.eval()
    args.output.mkdir(parents=True,exist_ok=False)
    metadata=dict(status='RUNNING',selection=selection,manifest_sha256=sha(manifest),evaluation_source_sha256=sha(Path(__file__)),torch_version=str(torch.__version__),protocol='RGB /255; bilinear input resize; original-size bilinear logits align_corners=False then sigmoid >=0.5; per-image binary Dice/IoU arithmetic mean',test_set='798 entries; previously viewed project test set; not an untouched external holdout')
    def save(): (args.output/'evaluation.json').write_text(json.dumps(metadata,indent=2))
    save(); rows=[]
    try:
        with (args.output/'per_image.csv').open('w') as f, torch.inference_mode():
            writer=csv.DictWriter(f,fieldnames=['sample_id','dataset','dice','iou']);writer.writeheader()
            for r in records:
                with Image.open(ROOT/r['image']) as im: image=im.convert('RGB'); assert image.size==(r['width'],r['height']); image=image.resize((config['size'],config['size']),Image.Resampling.BILINEAR)
                with Image.open(ROOT/r['mask']) as im: y=np.array(im.convert('L'))
                assert y.shape==(r['height'],r['width']); y=y>(0 if y.max()<=1 else 127)
                x=torch.from_numpy((np.array(image,dtype=np.float32).transpose(2,0,1)/255).copy())[None].to(device)
                logits=F.interpolate(model(x),size=y.shape,mode='bilinear',align_corners=False)
                probability=logits.sigmoid()[0,0].cpu().numpy(); assert np.isfinite(probability).all()
                dice,iou=metrics(probability>=.5,y); row=dict(sample_id=r['sample_id'],dataset=r['dataset'],dice=dice,iou=iou);rows.append(row);writer.writerow(row)
                if len(rows)%100==0: print(f'evaluated {len(rows)}/798',flush=True);f.flush()
        groups=[]
        for group in ['ALL','Kvasir','CVC-ClinicDB','CVC-ColonDB','ETIS-LaribPolypDB','CVC-300']:
            subset=[r for r in rows if group=='ALL' or r['dataset']==group]
            groups.append(dict(dataset=group,count=len(subset),dice=float(np.mean([r['dice'] for r in subset])),iou=float(np.mean([r['iou'] for r in subset]))))
        metadata.update(status='COMPLETED',groups=groups);save();print(json.dumps(groups),flush=True)
    except BaseException as e:
        metadata.update(status='FAILED',error=repr(e),processed=len(rows));save();raise
if __name__=='__main__': main()
