"""Versioned RGB-frame -> native-resolution float32 score-map interface."""
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .network import ResNet34UNet

MODEL_VERSION = "unet-b-expanded-seed42-r98-v1"
ROOT = Path(__file__).resolve().parent.parent
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)
_INFERENCE_LOCK = threading.RLock()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_frame(frame):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be an RGB uint8 ndarray shaped(H,W,3); convert BGR explicitly")
    if min(frame.shape[:2]) < 1 or max(frame.shape[:2]) > 8192 or frame.shape[0]*frame.shape[1] > 16_777_216:
        raise ValueError("frame dimensions exceed adapter limits")


def preprocess(frame):
    validate_frame(frame)
    rgb = Image.fromarray(np.ascontiguousarray(frame), mode="RGB")
    array = np.asarray(rgb.resize((352, 352), Image.Resampling.BILINEAR), np.float32)/255
    array = (array-MEAN)/STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0)


@dataclass(frozen=True)
class PredictionResult:
    probability_map: np.ndarray
    model_version: str
    inference_time_ms: float
    device: str


class ModelAdapter:
    def __init__(self, model_dir=None, device="auto", cpu_threads=4):
        self.root = Path(model_dir or ROOT).resolve()
        self.configuration = json.loads((self.root/"model.json").read_text(encoding="utf-8"))
        if self.configuration["model_version"] != MODEL_VERSION:
            raise ValueError("Unsupported model configuration; no E0/ensemble fallback")
        relative = Path(self.configuration["weights_file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid weight path")
        self.weights = (self.root/relative).resolve()
        if not self.weights.is_relative_to(self.root) or sha256(self.weights) != self.configuration["weights_sha256"]:
            raise ValueError("Weight integrity check failed")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device not in ("cpu", "cuda", "cuda:0"):
            raise ValueError("Supported devices:auto/cpu/cuda:0")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; select cpu explicitly if intended")
        self.device = torch.device("cuda:0" if device == "cuda" else device)
        torch.set_num_threads(max(1, min(int(cpu_threads), 16)))
        payload = torch.load(self.weights, map_location="cpu", weights_only=True)
        if payload["model_version"] != MODEL_VERSION or payload["source_checkpoint_sha256"] != self.configuration["source_checkpoint_sha256"]:
            raise ValueError("Weight/config provenance mismatch")
        self.model = ResNet34UNet()
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.to(self.device).eval()
        del payload

    def predict(self, frame, model_version=MODEL_VERSION):
        if model_version != MODEL_VERSION:
            raise ValueError(f"Unsupported model_version; only {MODEL_VERSION} is released")
        validate_frame(frame)
        with _INFERENCE_LOCK, torch.inference_mode(), torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
            previous = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                x = preprocess(frame).to(self.device)
                logits = self.model(x)
                resized = F.interpolate(logits.float(), size=frame.shape[:2], mode="bilinear", align_corners=False)
                probability = torch.sigmoid(resized)[0, 0].cpu().numpy().astype(np.float32)
                elapsed = (time.perf_counter()-started)*1000
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous
        if (probability.shape != frame.shape[:2] or not np.isfinite(probability).all()
                or probability.min() < 0 or probability.max() > 1):
            raise RuntimeError("Invalid model output")
        return PredictionResult(probability, MODEL_VERSION, float(elapsed), str(self.device))

    def describe(self):
        return {**self.configuration, "device": str(self.device), "torch_version": str(torch.__version__),
                "timing_scope": "adapter preprocessing+transfer+forward+native resize+sigmoid+CPU copy; excludes queue/video/postprocessing",
                "clinical_use": False}


_default = None
_LOAD_LOCK = threading.Lock()


def predict(frame, model_version=MODEL_VERSION):
    global _default
    with _LOAD_LOCK:
        if _default is None:
            _default = ModelAdapter(device=os.getenv("ENDO_DEVICE", "auto"))
    return _default.predict(frame, model_version)
