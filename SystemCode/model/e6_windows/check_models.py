"""Load the real E0/E4 checkpoints and compare one inference with frozen probabilities."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import run_e6 as e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    project = Path(args.project).resolve()

    import torch
    import torch.nn.functional as functional

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    record = e6.read_json(project / "runs/e0_20261004_221345_872444/manifest.json")[0]
    shape = (record["height"], record["width"])
    with Image.open(e6.safe_under(project, record["image"])) as image:
        tensor = e6.normalized_tensor(image.convert("RGB"), 352, torch).to(device)

    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from pvt_timm_compat import install_if_missing

    install_if_missing()
    from lib.pvt import PolypPVT

    e0 = PolypPVT()
    e0.load_state_dict(
        torch.load(project / "model_pth/PolypPVT.pth", map_location="cpu", weights_only=True),
        strict=True,
    )
    e0.to(device).eval()
    with torch.inference_mode():
        p1, p2 = e0(tensor)
        current_e0 = torch.sigmoid(
            functional.interpolate(p1 + p2, size=shape, mode="bilinear", align_corners=False)
        )[0, 0].cpu().numpy().astype(np.float32)
    frozen_e0 = e6.load_probability(
        project / "runs/e0_20261004_221345_872444" / ("probabilities/" + record["sample_id"] + ".npz"),
        shape,
    )
    del e0
    if device.type == "cuda":
        torch.cuda.empty_cache()

    e4_run = project / "runs/e4_final_seed42"
    e4_info = json.loads((e4_run / "run_info.json").read_text(encoding="utf-8"))
    core = project / "e4_windows/core"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    models = importlib.import_module("models")
    checkpoint = torch.load(e4_run / "checkpoints/final.pth", map_location="cpu", weights_only=True)
    e4 = models.make_model(checkpoint["config"], initialization=False)
    e4.load_state_dict(checkpoint["model_state"], strict=True)
    e4.to(device).eval()
    with torch.inference_mode():
        logits = e4(tensor)
        current_e4 = torch.sigmoid(
            functional.interpolate(logits.float(), size=shape, mode="bilinear", align_corners=False)
        )[0, 0].cpu().numpy().astype(np.float32)
    eval_dir = e4_run / "evaluation_cuda_20261006_235221_022843"
    frozen_e4 = e6.load_probability(
        eval_dir / ("probabilities/" + record["sample_id"] + ".npz"), shape
    )

    results = {}
    for name, current, frozen in (("E0", current_e0, frozen_e0), ("E4", current_e4, frozen_e4)):
        difference = np.abs(current.astype(np.float64) - frozen.astype(np.float64))
        results[name] = {
            "mean_absolute_probability_difference": float(difference.mean()),
            "max_absolute_probability_difference": float(difference.max()),
            "fraction_within_1e-4": float((difference <= 1e-4).mean()),
        }
    print("MODEL_PROTOCOL_CHECK", record["sample_id"], json.dumps(results, indent=2))
    for name, result in results.items():
        # E0's frozen probabilities were produced by Torch 2.2 on macOS CPU;
        # Windows CUDA may differ at a very small number of pixels. The mean
        # tolerance guards the protocol while allowing isolated backend noise.
        if result["mean_absolute_probability_difference"] > 2e-5 or result["max_absolute_probability_difference"] > 5e-3:
            raise ValueError(name + " inference differs unexpectedly from the frozen protocol")
    print("MODEL_PROTOCOL_CHECK_PASSED")


if __name__ == "__main__":
    main()
