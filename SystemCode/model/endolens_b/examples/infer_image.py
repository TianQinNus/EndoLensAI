"""Single image CLI; model API integration independent of FastAPI/video packages."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from endolens_model import ModelAdapter, MODEL_VERSION


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--image",default=str(Path(__file__).parent/"synthetic_frame.png"))
    parser.add_argument("--output",default="single_frame_output")
    parser.add_argument("--device",choices=("auto","cpu","cuda:0"),default="auto")
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    with Image.open(args.image) as im:frame=np.asarray(im.convert("RGB"),np.uint8)
    adapter=ModelAdapter(device=args.device)
    result=adapter.predict(frame,MODEL_VERSION)
    np.save(output/"probability.npy",result.probability_map,allow_pickle=False)
    Image.fromarray((result.probability_map>=.5).astype(np.uint8)*255,mode="L").save(output/"mask.png")
    (output/"result.json").write_text(json.dumps({"model_version":result.model_version,"device":result.device,
        "inference_time_ms":result.inference_time_ms,"width":frame.shape[1],"height":frame.shape[0],
        "segmentation_threshold":.5,"score_warning":"Foreground scores, NOT calibrated clinical probabilities"},indent=2),encoding="utf-8")
    print("SAVED",output.resolve())


if __name__=="__main__":main()
