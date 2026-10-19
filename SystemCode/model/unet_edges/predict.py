"""Predict a polyp region and its inner boundary from an E1 U-Net checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'e1_mac'))
from run_e1 import choose_device, normalize_rgb, code_hashes
from unet import UNet
from e0_protocol import file_hash


def boundary(mask):
    """One-pixel inner boundary, using 8-neighbor erosion and zero padding."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('Expected a 2D mask')
    padded = np.pad(mask, 1, constant_values=False)
    h, w = mask.shape
    eroded = np.ones_like(mask)
    for y in range(3):
        for x in range(3):
            eroded &= padded[y:y+h, x:x+w]
    return mask & ~eroded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if config['model'] != 'standard_padded_unet':
        raise ValueError('Expected a standard E1 U-Net checkpoint')
    if checkpoint['source_hashes'] != code_hashes():
        raise ValueError('Checkpoint and E1 Mac source hashes differ')
    model = UNet(config['base_channels'])
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.to(device).eval()
    with Image.open(args.image) as source:
        rgb = source.convert('RGB')
    with torch.inference_mode():
        logits = model(normalize_rgb(rgb, config['input_size'])[None].to(device))
        logits = F.interpolate(logits, size=(rgb.height, rgb.width), mode='bilinear', align_corners=False)
        probability = logits.sigmoid()[0, 0].cpu().numpy()
    if not np.isfinite(probability).all():
        raise ValueError('Model produced non-finite probabilities')
    mask = probability >= 0.5
    edge = boundary(mask)
    overlay = np.asarray(rgb).copy()
    overlay[edge] = [255, 40, 40]
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'probability.npz', probability=probability)
    Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / 'mask.png')
    Image.fromarray(edge.astype(np.uint8) * 255).save(args.output / 'edge.png')
    Image.fromarray(overlay).save(args.output / 'overlay.png')
    metadata = {'checkpoint': str(args.checkpoint.resolve()),
                'checkpoint_sha256': file_hash(args.checkpoint),
                'image': str(args.image.resolve()), 'image_sha256': file_hash(args.image),
                'predict_source_sha256': file_hash(Path(__file__)),
                'device': str(device), 'config': config, 'epoch': checkpoint['epoch'],
                'threshold': 0.5, 'boundary': 'mask minus 3x3 erosion, zero padding',
                'smoke_only': config['smoke'], 'width': rgb.width, 'height': rgb.height}
    (args.output / 'prediction.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('SMOKE ONLY: not a trained quality model' if config['smoke'] else 'Prediction complete')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
