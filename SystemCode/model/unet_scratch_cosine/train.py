"""Independent U-Net training from random weights; no existing model imports."""
import argparse
import hashlib
import math
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT.parents[1] / 'Miscellaneous/model_artifacts/e2_data_seed42'


def double_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        channels = [base * 2**i for i in range(5)]
        self.encoder = nn.ModuleList([double_conv(3, base)] +
                                    [double_conv(a, b) for a, b in zip(channels, channels[1:])])
        self.upsample = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i-1], 2, 2)
                                      for i in range(4, 0, -1)])
        self.decoder = nn.ModuleList([double_conv(channels[i], channels[i-1]) for i in range(4, 0, -1)])
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(nn.functional.max_pool2d(x, 2) if i else x)
            skips.append(x)
        for up, block, skip in zip(self.upsample, self.decoder, reversed(skips[:-1])):
            x = block(torch.cat([up(x), skip], dim=1))
        return self.output(x)


def loss_fn(logits, y):
    p = logits.sigmoid()
    dice = (2 * (p*y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return nn.functional.binary_cross_entropy_with_logits(logits, y) + 1 - dice.mean()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Pairs(Dataset):
    def __init__(self, records, size, augment):
        self.records, self.size, self.augment = records, size, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(ROOT / r['image']) as im:
            image = im.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR)
        with Image.open(ROOT / r['mask']) as im:
            a = np.array(im.convert('L'))
        mask = Image.fromarray((a > (0 if a.max() <= 1 else 127)).astype(np.uint8)*255)
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        if self.augment:
            for op in [Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM]:
                if random.random() < .5:
                    image, mask = image.transpose(op), mask.transpose(op)
        x = np.array(image, dtype=np.float32).transpose(2, 0, 1)/255
        y = (np.array(mask) > 127).astype(np.float32)[None]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def save_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, data):
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def epoch_lr(epoch, epochs, initial):
    """Cosine decay from initial LR to 1% at the final epoch."""
    progress = (epoch - 1) / max(epochs - 1, 1)
    return initial * (0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--base', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default='mps')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.size, args.base, args.batch_size, args.lr) <= 0 or args.size % 16:
        parser.error('Positive settings required; size must be divisible by 16')
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Apple GPU unavailable in this process')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    config = {k: v for k, v in vars(args).items() if k not in ['output', 'resume']}
    config.update(seed=42, initialization='random; no pretrained weights', input='RGB /255',
                  schedule='epoch cosine 1e-4 to 1e-6 over 100 epochs', loss='BCE + soft Dice', selection='maximum validation mean binary Dice',
                  validation_resolution=args.size, threshold=0.5, torch_version=str(torch.__version__),
                  source_sha256=sha(Path(__file__)))
    records = {}
    for name in ['dev_train', 'validation']:
        records[name] = json.loads((SPLITS / (name+'.json')).read_text())
        config[name+'_sha256'] = sha(SPLITS / (name+'.json'))
        for r in records[name]:
            for field in ['image', 'mask']:
                if sha(ROOT / r[field]) != r[field+'_sha256']:
                    raise ValueError('Data hash mismatch: ' + r[field])
        save_json(output / (name+'.json'), records[name])
    train_ids = {r['decoded_rgb_sha256'] for r in records['dev_train']}
    if train_ids & {r['decoded_rgb_sha256'] for r in records['validation']}:
        raise ValueError('Duplicate RGB content across train/validation')
    model = UNet(args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start, best, history = 1, -1., []
    if args.resume:
        checkpoint = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Resume configuration/source/data mismatch')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, best, history = checkpoint['epoch']+1, checkpoint['best_dice'], checkpoint['history']
        random.setstate(checkpoint['python_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
    save_json(output/'config.json', config)
    train = DataLoader(Pairs(records['dev_train'], args.size, True), batch_size=args.batch_size, shuffle=True)
    validation = DataLoader(Pairs(records['validation'], args.size, False), batch_size=args.batch_size)
    status = dict(status='RUNNING', pid=os.getpid(), device=str(device), train_count=len(train.dataset),
                  validation_count=len(validation.dataset), parameters=sum(p.numel() for p in model.parameters()))
    save_json(output/'status.json', status)
    print(json.dumps(status), flush=True)
    try:
        for epoch in range(start, args.epochs+1):
            learning_rate = epoch_lr(epoch, args.epochs, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            begin = time.time()
            model.train()
            total = 0.
            for step, (x, y) in enumerate(train, 1):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                optimizer.step()
                total += loss.item()*len(x)
                if step == 1 or step % 25 == 0:
                    print(f'epoch {epoch}/{args.epochs} step {step}/{len(train)} loss={loss.item():.5f}', flush=True)
                    save_json(output/'status.json', dict(status, epoch=epoch, step=step, updated_at=time.time()))
            model.eval()
            dice_sum, iou_sum, val_loss = 0., 0., 0.
            with torch.inference_mode():
                for x, y in validation:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    val_loss += loss_fn(logits, y).item()*len(x)
                    p = logits.sigmoid() >= .5
                    target = y.bool()
                    intersection = (p & target).sum((1, 2, 3)).float()
                    denominator = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
                    union = (p | target).sum((1, 2, 3))
                    dice_sum += torch.where(denominator > 0, 2*intersection/denominator.clamp_min(1), 1.).sum().item()
                    iou_sum += torch.where(union > 0, intersection/union.clamp_min(1), 1.).sum().item()
            row = dict(epoch=epoch, learning_rate=learning_rate, train_loss=total/len(train.dataset), val_loss=val_loss/len(validation.dataset),
                       val_dice=dice_sum/len(validation.dataset), val_iou=iou_sum/len(validation.dataset), seconds=time.time()-begin)
            improved = row['val_dice'] > best
            best = max(best, row['val_dice'])
            history.append(row)
            weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if improved:
                save_checkpoint(output/'best.pt', dict(model=weights, config=config, epoch=epoch, val_dice=best))
            save_checkpoint(output/'last.pt', dict(model=weights, optimizer=optimizer.state_dict(), config=config,
                            epoch=epoch, best_dice=best, history=history, python_rng=random.getstate(), torch_rng=torch.get_rng_state()))
            save_json(output/'history.json', history)
            status.update(completed_epochs=epoch, best_val_dice=best, last_epoch=row)
            save_json(output/'status.json', status)
            print(json.dumps(row), flush=True)
        status['status'] = 'COMPLETED'
    except BaseException as error:
        status.update(status='FAILED', error=repr(error))
        raise
    finally:
        save_json(output/'status.json', status)


if __name__ == '__main__':
    main()
