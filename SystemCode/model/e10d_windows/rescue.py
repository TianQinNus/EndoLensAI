"""Prediction-only regions, non-erasing fusion and fixed small diagnostic gates."""
from collections import defaultdict
import numpy as np

FEATURES = ("e0_flip_retention_min", "b_flip_support_mean", "b_original_mean",
            "e0_original_mean", "log_area_fraction")
BASE_FEATURES = FEATURES[2:]
ACCEPT = .8
RIDGE = .01


def components(mask):
    """Deterministic 8-connected components via row runs; no optional dependencies."""
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Require a two-dimensional bool mask")
    height, width = mask.shape
    parent, runs, previous = [], [], []
    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for y in range(height):
        padded = np.pad(mask[y].astype(np.int8), (1, 1))
        edges = np.diff(padded)
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
        current, left = [], 0
        for start, end in zip(starts.tolist(), ends.tolist()):
            k = len(parent)
            parent.append(k)
            runs.append((y, start, end))
            while left < len(previous) and previous[left][1] < start - 1:
                left += 1
            j = left
            while j < len(previous) and previous[j][0] <= end + 1:
                a, b = root(k), root(previous[j][2])
                parent[max(a, b)] = min(a, b)
                j += 1
            current.append((start, end, k))
        previous = current
    groups = defaultdict(list)
    for k, (y, start, end) in enumerate(runs):
        groups[root(k)].append(np.arange(y * width + start, y * width + end + 1, dtype=np.int64))
    return [np.concatenate(groups[k]) for k in sorted(groups)]


def validate_pair(pb, p0):
    if (pb.shape != p0.shape or pb.ndim != 2 or pb.dtype != np.float32 or p0.dtype != np.float32
            or not np.isfinite(pb).all() or not np.isfinite(p0).all()
            or min(pb.min(), p0.min()) < 0 or max(pb.max(), p0.max()) > 1):
        raise ValueError("Require aligned finite float32 probabilities in [0,1]")


def candidate_regions(pb, p0):
    validate_pair(pb, p0)
    return components((p0 >= .5) & (pb < .5))


def region_features(pb, p0, b_views, e0_views, indices):
    """No GT argument: all gate inputs are available at actual inference."""
    validate_pair(pb, p0)
    if len(b_views) != 2 or len(e0_views) != 2 or len(indices) == 0:
        raise ValueError("Exactly hflip/vflip views and a nonempty component are required")
    for q in (*b_views, *e0_views):
        validate_pair(pb, q)
    return dict(zip(FEATURES, (
        min(float((q.reshape(-1)[indices] >= .5).mean()) for q in e0_views),
        float(np.mean([q.reshape(-1)[indices].mean() for q in b_views])),
        float(pb.reshape(-1)[indices].mean()), float(p0.reshape(-1)[indices].mean()),
        float(np.log(len(indices) / pb.size)))))


def supplement(pb, p0, regions):
    validate_pair(pb, p0)
    result = pb.copy()
    flat, pflat = result.reshape(-1), p0.reshape(-1)
    for indices in regions:
        flat[indices] = np.maximum(flat[indices], pflat[indices])
    if np.any(result < pb) or np.any((pb >= .5) & (result < .5)):
        raise AssertionError("Non-erasing invariant violated")
    return result


def isolated_label(pb, p0, gt, indices):
    """GT used only as training label / held-out diagnostic, never as a feature."""
    if gt.shape != pb.shape or gt.dtype != np.bool_:
        raise ValueError("Invalid binary target")
    pred = pb >= .5
    tp = int((pred & gt).sum())
    denominator = int(pred.sum() + gt.sum())
    if denominator == 0:
        raise ValueError("Diagnostic requires positive GT images")
    truth = gt.reshape(-1)[indices]
    added_tp, added_fp = int(truth.sum()), int((~truth).sum())
    dice_delta = 2 * (tp + added_tp) / (denominator + len(indices)) - 2 * tp / denominator
    increase = np.maximum(0., p0.reshape(-1)[indices].astype(np.float64) - pb.reshape(-1)[indices])
    mae_delta = float(np.sum(increase * (1 - 2 * truth.astype(np.float64))) / pb.size)
    return {"helpful": int(dice_delta > 1e-6 and mae_delta <= 1e-12),
            "isolated_dice_delta": float(dice_delta), "isolated_mae_delta": mae_delta,
            "added_tp": added_tp, "added_fp": added_fp}


def sigmoid(x):
    x = np.asarray(x, np.float64)
    return np.exp(-np.logaddexp(0., -x))


def component_weights(rows):
    """Equal patients, equal images with candidates within patient, equal components."""
    if not rows:
        return np.empty(0, np.float64)
    patient_images, counts = defaultdict(set), defaultdict(int)
    for r in rows:
        patient_images[r["group_id"]].add(r["sample_id"])
        counts[r["sample_id"]] += 1
    return np.array([1 / (len(patient_images) * len(patient_images[r["group_id"]]) * counts[r["sample_id"]])
                     for r in rows], np.float64)


def fit_gate(training, features=FEATURES):
    """One fixed ridge logistic fit, train-only normalization, no class resampling."""
    if features not in (FEATURES, BASE_FEATURES):
        raise ValueError("Only predeclared full/ablated features allowed")
    if not training:
        return {"features": list(features), "constant": 0., "training_components": 0}
    x = np.array([[r[k] for k in features] for r in training], np.float64)
    y, w = np.array([r["helpful"] for r in training], np.float64), component_weights(training)
    if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("Invalid features/labels")
    if len(set(y)) == 1:
        return {"features": list(features), "constant": float((sum(y) + 1) / (len(y) + 2)),
                "training_components": len(training)}
    mean = (x * w[:, None]).sum(axis=0)
    scale = np.maximum(np.sqrt(((x - mean) ** 2 * w[:, None]).sum(axis=0)), 1e-6)
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    beta = np.zeros(design.shape[1])
    penalty = np.diag([1e-8] + [RIDGE] * len(features))
    def objective(value):
        z = design @ value
        return float(w @ (np.logaddexp(0, z) - y * z) + .5 * value @ penalty @ value)
    converged = False
    for iteration in range(60):
        z, p = design @ beta, sigmoid(design @ beta)
        gradient = design.T @ (w * (p - y)) + penalty @ beta
        if np.max(np.abs(gradient)) < 1e-9:
            converged = True
            break
        hessian = design.T @ ((w * p * (1 - p))[:, None] * design) + penalty
        step = np.linalg.solve(hessian, gradient)
        rate, before = 1., objective(beta)
        for _ in range(40):
            proposed = beta - rate * step
            if objective(proposed) <= before - 1e-4 * rate * float(gradient @ step):
                beta = proposed
                break
            rate *= .5
        else:
            raise ValueError("Fixed logistic optimization failed line search")
    if not converged:
        raise ValueError("Fixed logistic optimization did not converge; no fallback tuning")
    return {"features": list(features), "mean": mean.tolist(), "scale": scale.tolist(),
            "beta": beta.tolist(), "training_components": len(training), "iterations": iteration + 1,
            "objective": objective(beta), "ridge": RIDGE}


def score_gate(rows, fitted):
    if not rows:
        return np.empty(0, np.float64)
    if "constant" in fitted:
        return np.full(len(rows), fitted["constant"], np.float64)
    # Only explicit prediction features read here: labels/GT cannot affect scores.
    x = np.array([[r[k] for k in fitted["features"]] for r in rows], np.float64)
    z = (x - np.array(fitted["mean"])) / np.array(fitted["scale"])
    return sigmoid(np.column_stack((np.ones(len(rows)), z)) @ np.array(fitted["beta"]))


def discrimination(rows, score_key):
    if not rows:
        return {"components": 0, "weighted_auc": None, "admitted": 0, "precision": None, "admitted_groups": 0}
    y, scores = np.array([r["helpful"] for r in rows]), np.array([r[score_key] for r in rows])
    w, admitted = component_weights(rows), scores >= ACCEPT
    positive, negative = float(w[y == 1].sum()), float(w[y == 0].sum())
    auc = None
    if positive and negative:
        order = np.argsort(scores, kind="stable")
        before, numerator = 0., 0.
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and scores[order[end]] == scores[order[start]]:
                end += 1
            part = order[start:end]
            pos, neg = w[part][y[part] == 1].sum(), w[part][y[part] == 0].sum()
            numerator += pos * (before + .5 * neg)
            before += neg
            start = end
        auc = float(numerator / (positive * negative))
    return {"components": len(rows), "helpful": int(y.sum()), "weighted_auc": auc,
            "admitted": int(admitted.sum()), "precision": float(y[admitted].mean()) if admitted.any() else None,
            "weighted_precision": float(w[admitted] @ y[admitted] / w[admitted].sum()) if admitted.any() else None,
            "admitted_groups": len({r["group_id"] for r, keep in zip(rows, admitted) if keep})}
