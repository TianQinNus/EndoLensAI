"""Descriptive test error audit, CPU only; no threshold/checkpoint tuning."""
import csv, json, importlib.util
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/error_analysis_20261018'
splits=ROOT.parents[1]/'Miscellaneous/model_artifacts/e2_data_seed42'
records=json.loads((splits/'test.json').read_text())
scores={r['sample_id']:float(r['dice']) for r in csv.DictReader((ROOT/'runs/test_selected_gn_wide_20261017/per_image.csv').open())}
summary={}
for name, rows in [('Train1450',json.loads((splits/'full_train.json').read_text()))]+[(d,[r for r in records if r['dataset']==d]) for d in sorted({r['dataset'] for r in records})]:
    areas=np.array([r['gt_foreground_pixels']/(r['height']*r['width']) for r in rows])
    entry=dict(count=len(rows),median_area_fraction=float(np.median(areas)),fraction_area_below_1pct=float(np.mean(areas<.01)),median_height=float(np.median([r['height'] for r in rows])),median_width=float(np.median([r['width'] for r in rows])))
    if name!='Train1450':
        dice=np.array([scores[r['sample_id']] for r in rows]);entry.update(mean_dice=float(dice.mean()),dice_below_01=int((dice<.1).sum()),dice_below_05=int((dice<.5).sum()),area_bins=[])
        for lo,hi in [(0,.01),(.01,.05),(.05,.1),(.1,1.01)]:
            v=dice[(areas>=lo)&(areas<hi)];entry['area_bins'].append(dict(lo=lo,hi=hi,count=len(v),dice=float(v.mean()) if len(v) else None))
    summary[name]=entry
(OUT/'distribution.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
# Fixed selection: 3 worst, 2 near-median, 1 best per hard subset; no cherry-picked model selection.
chosen=[]
for d in ['ETIS-LaribPolypDB','CVC-ColonDB']:
    r=sorted([r for r in records if r['dataset']==d],key=lambda r:(scores[r['sample_id']],r['sample_id']))
    chosen += [r[i] for i in [0,1,2,len(r)//2-1,len(r)//2,len(r)-1]]
selection=json.loads((ROOT/'unet_evaluation/selection_gn_wide_20261017.json').read_text())
import hashlib
assert hashlib.sha256((ROOT/selection['checkpoint']).read_bytes()).hexdigest()==selection['checkpoint_sha256']
assert hashlib.sha256((ROOT/selection['source']).read_bytes()).hexdigest()==selection['source_sha256']
spec=importlib.util.spec_from_file_location('model',ROOT/selection['source']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(2);cp=torch.load(ROOT/selection['checkpoint'],map_location='cpu',weights_only=True);model=m.UNet(cp['config']['base']).eval();model.load_state_dict(cp['model'])
results=[]
for k,r in enumerate(chosen):
    with Image.open(ROOT/r['image']) as im: im=im.convert('RGB'); rgb=np.array(im)
    with Image.open(ROOT/r['mask']) as mask: y=np.array(mask.convert('L'))
    y=y>(0 if y.max()<=1 else 127)
    x=torch.from_numpy((np.array(im.resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/255).copy())[None]
    with torch.inference_mode(): p=F.interpolate(model(x),size=y.shape,mode='bilinear',align_corners=False).sigmoid()[0,0].numpy()>=.5
    tp=int((p&y).sum());fp=int((p&~y).sum());fn=int((~p&y).sum());dice=2*tp/max(int(p.sum()+y.sum()),1)
    row=dict(sample_id=r['sample_id'],stored_mps_dice=scores[r['sample_id']],cpu_dice=dice,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),prediction_area_fraction=float(p.mean()),gt_area_fraction=float(y.mean()),tp=tp,fp=fp,fn=fn)
    results.append(row)
    overlay=rgb.copy(); overlay[y&~p]=[40,110,255];overlay[p&~y]=[255,50,50];overlay[p&y]=[40,220,70]
    tiles=[im,Image.fromarray(y.astype('uint8')*255).convert('RGB'),Image.fromarray(p.astype('uint8')*255).convert('RGB'),Image.fromarray(overlay)]
    canvas=Image.new('RGB',(1024,224),'white');draw=ImageDraw.Draw(canvas);draw.text((4,4),f"{r['sample_id']} Dice={dice:.3f} precision={row['precision']:.3f} recall={row['recall']:.3f}",fill='black')
    for j,t in enumerate(tiles): canvas.paste(t.resize((256,192)),(j*256,30))
    canvas.save(OUT/f'case_{k:02d}.png')
    print(row,flush=True)
for start in [0,6]:
    sheet=Image.new('RGB',(1024,224*6),'white')
    for k in range(6):
        with Image.open(OUT/f'case_{start+k:02d}.png') as im:sheet.paste(im,(0,k*224))
    sheet.save(OUT/f'contact_{start//6}.png')
(OUT/'cases.json').write_text(json.dumps(dict(selection=selection,selection_rule='per subset 3 worst, 2 median, 1 best by existing frozen scores',columns=['image','ground truth','prediction','TP green / FN blue / FP red'],cases=results),indent=2))
