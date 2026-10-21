"""Bounded, independent temperature scaling; deterministic image-equal NLL."""
import hashlib
from collections import defaultdict
import numpy as np

BOUNDS = (.25, 4.)
EPSILON = 1e-12
PIXELS_PER_IMAGE = 8192


def logits(probability):
    p = np.asarray(probability, np.float64)
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("Invalid probability")
    p = np.clip(p, EPSILON, 1 - EPSILON)
    lower = np.log(EPSILON) - np.log1p(-EPSILON)
    # Symmetric endpoint cap avoids float64 cancellation at1-epsilon creating
    # different magnitudes for original cached0 and1.
    return np.clip(np.log(p) - np.log1p(-p), lower, -lower)


def sigmoid(value):
    x = np.asarray(value, np.float64)
    result = np.empty_like(x)
    positive = x >= 0
    result[positive] = 1 / (1 + np.exp(-x[positive]))
    negative = np.exp(x[~positive])
    result[~positive] = negative / (1 + negative)
    return result


def apply_temperature(probability, temperature):
    if not BOUNDS[0] <= temperature <= BOUNDS[1] or probability.dtype != np.float32:
        raise ValueError("Require float32 input and predeclared positive temperature bounds")
    if temperature == 1:
        logits(probability)  # validate; identity preserves original endpoints exactly
        return probability.copy()
    q = sigmoid(logits(probability) / temperature).astype(np.float32)
    # Positive temperature mathematically preserves the 0.5 decision boundary.
    # Avoid float32 rounding moving a value just below0.5 onto0.5. No GT involved.
    q[(probability < .5) & (q >= .5)] = np.nextafter(np.float32(.5), np.float32(0))
    q[(probability >= .5) & (q < .5)] = np.float32(.5)
    return q


def assign_folds(records, folds=5, seed=42):
    if any(r["split"] != "calibration" for r in records):
        raise ValueError("Fold construction accepts calibration only")
    groups = defaultdict(list)
    for r in records:
        groups[r["group_id"]].append(r)
    if len(groups) < folds or folds < 2:
        raise ValueError("Too few independent patient groups")
    rng = np.random.default_rng(seed)
    names = sorted(groups)
    rng.shuffle(names)
    names.sort(key=lambda name: -len(groups[name]))
    loads, patients, assignments = [0] * folds, [0] * folds, {}
    for name in names:
        minimum = min((loads[i], patients[i]) for i in range(folds))
        options = [i for i in range(folds) if (loads[i], patients[i]) == minimum]
        fold = int(rng.choice(options))
        assignments[name] = fold
        loads[fold] += len(groups[name])
        patients[fold] += 1
    return [dict(r, cv_fold=assignments[r["group_id"]]) for r in records]


def sample_pair(p_b, p0, gt, sample_id, seed=42, maximum=PIXELS_PER_IMAGE):
    if p_b.shape != p0.shape or p_b.shape != gt.shape or maximum < 1:
        raise ValueError("Invalid paired calibration sampling")
    derived = int.from_bytes(hashlib.sha256(f"E10C/{seed}/{sample_id}".encode()).digest()[:8], "little")
    indices = np.random.default_rng(derived).choice(gt.size, min(maximum, gt.size), replace=False)
    return {"b_logits": logits(p_b.reshape(-1)[indices]), "e0_logits": logits(p0.reshape(-1)[indices]),
            "targets": gt.reshape(-1)[indices].astype(np.float64),
            "sample_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()}


def fit_temperature(samples, model):
    if not samples or model not in ("b", "e0"):
        raise ValueError("Empty samples or unknown model")
    arrays = [np.asarray(r[model + "_logits"], np.float64) for r in samples]
    targets = [np.asarray(r["targets"], np.float64) for r in samples]
    if any(x.shape != y.shape or x.size == 0 for x, y in zip(arrays, targets)):
        raise ValueError("Sample shapes differ or are empty")
    z, y = np.concatenate(arrays), np.concatenate(targets)
    w = np.concatenate([np.full(x.size, 1 / (len(arrays) * x.size)) for x in arrays])
    if not np.isfinite(z).all() or not np.isin(y, (0, 1)).all() or y.min() == y.max():
        raise ValueError("Calibration needs finite logits and both binary target classes")
    def objective(scale):
        return float(np.sum(w * (np.logaddexp(0, scale * z) - y * scale * z)))
    def derivative(scale):
        return float(np.sum(w * z * (sigmoid(scale * z) - y)))
    # NLL is convex in positive inverse temperature s=1/T. Solve its derivative
    # by fixed bisection, not a Dice-selected hyperparameter grid.
    lower, upper = 1 / BOUNDS[1], 1 / BOUNDS[0]
    boundary = None
    if derivative(lower) >= 0:
        scale, boundary = lower, "T_upper"
    elif derivative(upper) <= 0:
        scale, boundary = upper, "T_lower"
    else:
        for _ in range(64):
            middle = (lower + upper) / 2
            if derivative(middle) > 0:
                upper = middle
            else:
                lower = middle
        scale = (lower + upper) / 2
    before, after = objective(1), objective(scale)
    if after > before + 1e-10:
        raise ValueError("Bounded NLL fit worsened feasible identity")
    return {"temperature": float(1 / scale), "inverse_temperature": float(scale),
            "image_count": len(arrays), "sampled_pixels": int(z.size), "boundary": boundary,
            "sampled_image_equal_nll_before": before, "sampled_image_equal_nll_after": after,
            "objective": "uniform natural-prevalence pixels per image; each image equal total NLL weight"}
