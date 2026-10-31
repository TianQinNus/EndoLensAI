"""Offline RGB frame SDK with fixed rotate4 inference and the final mask rule."""
import hashlib,json,os,threading,time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .network import UNet
from .constants import MODEL_VERSION,POSTPROCESS_VERSION
from .postprocess import postprocess,PostprocessResult

ROOT=Path(__file__).resolve().parent.parent
_INFERENCE_LOCK=threading.RLock()

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def validate_frame(frame):
    if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3:
        raise ValueError('frame must be RGB uint8 ndarray(H,W,3); convert BGR explicitly')
    if min(frame.shape[:2])<1 or max(frame.shape[:2])>8192 or frame.shape[0]*frame.shape[1]>16_777_216:
        raise ValueError('Frame dimensions exceed adapter limits')

def preprocess(frame):
    validate_frame(frame)
    image=Image.fromarray(np.ascontiguousarray(frame),mode='RGB')
    array=np.asarray(image.resize((320,320),Image.Resampling.BILINEAR),np.float32)/255
    # Normalization is embedded in UNet; do not apply ImageNet normalization here.
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2,0,1))).unsqueeze(0)

@dataclass(frozen=True)
class PredictionResult:
    probability_map: np.ndarray
    model_version: str
    inference_time_ms: float
    device: str
    postprocess_result: PostprocessResult | None = None
    @property
    def binary_mask(self):
        if self.postprocess_result is None:raise RuntimeError('No filtered result attached')
        return self.postprocess_result.binary_mask
    @property
    def is_candidate(self):
        if self.postprocess_result is None:raise RuntimeError('No filtered result attached')
        return self.postprocess_result.is_candidate
    @property
    def regions(self):
        if self.postprocess_result is None:raise RuntimeError('No filtered result attached')
        return self.postprocess_result.regions
    @property
    def contours(self):
        if self.postprocess_result is None:raise RuntimeError('No filtered result attached')
        return self.postprocess_result.contours

class ModelAdapter:
    def __init__(self,model_dir=None,device='auto',cpu_threads=4):
        self.root=Path(model_dir or ROOT).resolve()
        self.configuration=json.loads((self.root/'model.json').read_text(encoding='utf-8'))
        c=self.configuration
        if c['model_version']!=MODEL_VERSION or c['postprocessing_version']!=POSTPROCESS_VERSION:
            raise ValueError('Unsupported release identity')
        if c['input']['size']!=[320,320] or c['inference']['tta']!='rotate4' or c['candidate_rule']!={
            'pixel_threshold':.5,'min_component_mean_score':.8,'min_region_area_ratio':0.,
            'connectivity':8,'mask_filtering':True,'empty_mask_is_candidate':False,'temporal_confirmation':False}:
            raise ValueError('Released inference/output settings changed')
        relative=Path(c['weights_file'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid weight path')
        self.weights=(self.root/relative).resolve()
        if not self.weights.is_relative_to(self.root) or sha256(self.weights)!=c['weights_sha256']:
            raise ValueError('Weight integrity check failed')
        if device=='auto':device='cuda:0' if torch.cuda.is_available() else 'cpu'
        if device not in ['cpu','cuda','cuda:0']:raise ValueError('Supported devices: auto/cpu/cuda:0')
        if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        self.device=torch.device('cuda:0' if device=='cuda' else device)
        torch.set_num_threads(max(1,min(int(cpu_threads),16)))
        payload=torch.load(self.weights,map_location='cpu',weights_only=True)
        if payload['model_version']!=MODEL_VERSION or payload['source_checkpoint_sha256']!=c['source_checkpoint_sha256']:
            raise ValueError('Weight/config provenance mismatch')
        self.model=UNet();self.model.load_state_dict(payload['state_dict'],strict=True)
        self.model.to(self.device).float().eval();del payload

    def predict(self,frame,model_version=MODEL_VERSION):
        if model_version!=MODEL_VERSION:raise ValueError('Unsupported model_version: '+str(model_version))
        validate_frame(frame)
        with _INFERENCE_LOCK,torch.inference_mode(),torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=True,allow_tf32=False):
            previous=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
            try:
                if self.device.type=='cuda':torch.cuda.synchronize(self.device)
                start=time.perf_counter();x=preprocess(frame).to(self.device)
                with torch.autocast(device_type=self.device.type,enabled=False):
                    logits=sum(torch.rot90(self.model(torch.rot90(x,k,(-2,-1))),-k,(-2,-1)) for k in range(4))/4
                    p=F.interpolate(logits.float(),size=frame.shape[:2],mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy().astype(np.float32)
                if p.shape!=frame.shape[:2] or not np.isfinite(p).all():raise RuntimeError('Invalid model output')
                filtered=postprocess(p);elapsed=(time.perf_counter()-start)*1000
            finally:torch.backends.cuda.matmul.allow_tf32=previous
        return PredictionResult(p,MODEL_VERSION,float(elapsed),str(self.device),filtered)

    def describe(self):
        return {**self.configuration,'device':str(self.device),'torch_version':str(torch.__version__),
            'timing_scope':'preprocessing + 4 forwards + native resize/sigmoid + CPU copy + region filtering; excludes video/queue/storage',
            'clinical_use':False}

_default=None
_LOAD_LOCK=threading.Lock()
def predict(frame,model_version=MODEL_VERSION):
    global _default
    with _LOAD_LOCK:
        if _default is None:_default=ModelAdapter(device=os.getenv('ENDO_DEVICE','auto'))
    return _default.predict(frame,model_version)
