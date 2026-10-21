"""Photo in, original-size photo with predicted mask outlines out."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
import torch.nn.functional as F
try:
    from .model import UNet
except ImportError:
    from model import UNet

ROOT = Path(__file__).resolve().parent


def mask_edge(mask, width=2):
    """Inner morphological boundary, including holes and image-border objects."""
    if not isinstance(width, int) or not 1 <= width <= 20:
        raise ValueError('Line width must be an integer from 1 to 20')
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('Mask must be 2D')
    padded = Image.fromarray(np.pad(mask.astype(np.uint8)*255, width))
    eroded = np.asarray(padded.filter(ImageFilter.MinFilter(2*width+1)))
    eroded = eroded[width:-width, width:-width] > 0
    return mask & ~eroded


class PolypOutline:
    def __init__(self, device='cpu', threads=2):
        # CPU default avoids competing with the ongoing MPS training job.
        torch.set_num_threads(threads)
        self.device = torch.device(device)
        self.info = json.loads((ROOT/'model_info.json').read_text())
        for name, key in [('best.pt', 'checkpoint_sha256'), ('model.py', 'model_sha256')]:
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != self.info[key]:
                raise ValueError(f'File hash mismatch: {name}')
        cp = torch.load(ROOT/'best.pt', map_location='cpu', weights_only=True)
        if cp['config']['source_sha256'] != self.info['source_sha256']:
            raise ValueError('Checkpoint source mismatch')
        self.size = cp['config']['size']
        self.model = UNet(cp['config']['base']).to(self.device)
        self.model.load_state_dict(cp['model'], strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict_mask(self, image):
        image = image.convert('RGB')
        resized = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.array(resized, dtype=np.float32).transpose(2, 0, 1).copy()/255)[None]
        logits = self.model(x.to(self.device))
        logits = F.interpolate(logits, size=(image.height, image.width), mode='bilinear', align_corners=False)
        return (logits.sigmoid()[0, 0] >= .5).cpu().numpy()

    def annotate(self, image, line_width=2, color=(0, 255, 0)):
        image = ImageOps.exif_transpose(image).convert('RGB')
        edge = mask_edge(self.predict_mask(image), line_width)
        pixels = np.array(image).copy()
        pixels[edge] = color
        return Image.fromarray(pixels)

    def predict(self, input_path, output_path, line_width=2):
        source, target = Path(input_path), Path(output_path)
        if target.exists():
            raise FileExistsError(f'Output already exists: {target}')
        if target.suffix.lower() != '.png':
            raise ValueError('Use a .png output to preserve original pixels outside the outline')
        with Image.open(source) as image:
            result = self.annotate(image, line_width)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            result.save(f, format='PNG')
        return target


def main():
    parser = argparse.ArgumentParser(description='输入照片，输出仅叠加绿色息肉轮廓的原尺寸 PNG')
    parser.add_argument('image', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('--line-width', type=int, default=2, choices=range(1, 21), metavar='1..20')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    output = args.output or args.image.with_name(args.image.stem+'_outline.png')
    try:
        saved = PolypOutline(args.device).predict(args.image, output, args.line_width)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(saved.resolve())


if __name__ == '__main__':
    main()
