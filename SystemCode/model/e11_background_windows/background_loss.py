"""Bounded training-only background emphasis; no dataset mining or GT leakage."""
import torch
import torch.nn.functional as F

STRENGTH = 1.0
POWER = 2.0


def loss_parts(logits, target, strength=STRENGTH):
    """Per-image mean BCE+Dice plus detached, probability-weighted background BCE.

    Foreground BCE weight stays1. Background BCE weight is1+strength*stopgrad(p)^2.
    Reduction stays over ALL pixels, not renormalized over the foreground/background.
    This is a fixed experimental heuristic, not a claim of a new loss family.
    """
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1 or not 0 <= strength <= 1:
        raise ValueError("Require aligned N1HW logits/targets and bounded strength")
    logits, target = logits.float(), target.float()
    if not bool(torch.isfinite(target).all()) or not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("Require finite binary training targets")
    axes = (1, 2, 3)
    element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    base_bce = element.mean(axes)
    p = logits.sigmoid()
    extra = strength * ((1 - target) * p.detach().square() * element).mean(axes)
    dice = 1 - (2 * (p * target).sum(axes) + 1) / (p.sum(axes) + target.sum(axes) + 1)
    return base_bce + extra + dice, base_bce + extra, dice, base_bce, extra


def enrich_rows(rows):
    """Existing TP/FP/FN only; no changed threshold, probability or inference."""
    result = []
    for r in rows:
        row = dict(r)
        tp, fp, fn = (int(r[k]) for k in ("tp", "fp", "fn"))
        pixels = int(r["height"]) * int(r["width"])
        if min(tp, fp, fn) < 0 or tp + fn <= 0 or pixels <= 0 or tp + fp + fn > pixels:
            raise ValueError("Positive-only validation requires valid counts/dimensions")
        row.update(precision=tp / (tp + fp) if tp + fp else 0., recall=tp / (tp + fn),
                   fp_image_fraction=fp / pixels, fn_image_fraction=fn / pixels)
        result.append(row)
    return result
