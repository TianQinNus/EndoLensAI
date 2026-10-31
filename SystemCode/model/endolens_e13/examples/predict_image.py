"""Run one image and save the FILTERED native mask and region metadata."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from endolens_model import ModelAdapter,MODEL_VERSION,POSTPROCESS_VERSION

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='auto',choices=['auto','cpu','cuda:0'])
    p.add_argument('--save-probability',action='store_true');args=p.parse_args()
    with Image.open(args.image) as image:rgb=np.asarray(image.convert('RGB'),np.uint8)
    adapter=ModelAdapter(device=args.device);result=adapter.predict(rgb);post=result.postprocess_result
    args.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(result.binary_mask*255,mode='L').save(args.output,format='PNG')
    info={'model_version':MODEL_VERSION,'postprocessing_version':POSTPROCESS_VERSION,'is_candidate':result.is_candidate,
        'image_width':rgb.shape[1],'image_height':rgb.shape[0],'regions':result.regions,'contours':result.contours,
        'foreground_score':post.foreground_score,'retained_components':post.component_count,
        'removed_components':post.removed_component_count,'device':result.device,'inference_time_ms':result.inference_time_ms}
    args.output.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    if args.save_probability:np.save(args.output.with_suffix('.raw_probability.npy'),result.probability_map,allow_pickle=False)
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
