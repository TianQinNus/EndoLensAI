#!/usr/bin/env python3
"""E7-A: fuse raw E0/E4 probabilities, then apply affine logit calibration.

Workflow:
  select      fit one affine calibrator per alpha on the frozen calibration split,
              choose alpha/threshold on the frozen selection split, and lock config
  apply-test  apply a PASS config once to the frozen 798-image test set

E7-A reuses the E6 probability archives. It performs no neural-network training or
inference and never reads the frozen test probabilities during selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
E6_DIR = PROJECT_ROOT / "e6_windows"
if str(E6_DIR) not in sys.path:
    sys.path.insert(0, str(E6_DIR))
import run_e6 as e6  # noqa: E402


ALPHAS = tuple(round(index * 0.05, 2) for index in range(13))
THRESHOLDS = e6.THRESHOLDS
DEFAULT_GATES = dict(e6.DEFAULT_GATES)
SLOPE_BOUNDS = (0.01, 20.0)
INTERCEPT_BOUNDS = (-20.0, 20.0)


def sigmoid(logits):
    logits = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1.0 + exp_logits)
    return result


def affine_calibrate(probability, slope, intercept):
    if not np.isfinite(slope) or not SLOPE_BOUNDS[0] <= slope <= SLOPE_BOUNDS[1]:
        raise ValueError("Affine slope is invalid")
    if not np.isfinite(intercept) or not INTERCEPT_BOUNDS[0] <= intercept <= INTERCEPT_BOUNDS[1]:
        raise ValueError("Affine intercept is invalid")
    logits = slope * e6.probability_to_logit(probability) + intercept
    return sigmoid(logits).astype(np.float32)


def nll_from_logits(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return float(np.mean(np.maximum(logits, 0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))))


def fit_affine(probability, targets, maximum_iterations=80):
    probability = np.asarray(probability, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if probability.size != targets.size or probability.size == 0:
        raise ValueError("Calibration probability/target arrays are empty or mismatched")
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError("Calibration probability is invalid")
    if targets.min() == targets.max() or not np.isin(targets, (0, 1)).all():
        raise ValueError("Calibration targets must contain both binary classes")

    source_logits = e6.probability_to_logit(probability)
    slope, intercept = 1.0, 0.0
    converged = False
    iterations = 0

    for iterations in range(1, maximum_iterations + 1):
        logits = slope * source_logits + intercept
        calibrated = sigmoid(logits)
        residual = calibrated - targets
        weights = np.maximum(calibrated * (1.0 - calibrated), 1e-10)
        gradient = np.array(
            [np.mean(residual * source_logits), np.mean(residual)], dtype=np.float64
        )
        hessian = np.array(
            [
                [np.mean(weights * source_logits * source_logits), np.mean(weights * source_logits)],
                [np.mean(weights * source_logits), np.mean(weights)],
            ],
            dtype=np.float64,
        )
        hessian += np.eye(2, dtype=np.float64) * 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        current_loss = nll_from_logits(logits, targets)
        accepted = False
        scale = 1.0
        for _ in range(30):
            candidate_slope = float(np.clip(slope - scale * step[0], *SLOPE_BOUNDS))
            candidate_intercept = float(np.clip(intercept - scale * step[1], *INTERCEPT_BOUNDS))
            candidate_loss = nll_from_logits(
                candidate_slope * source_logits + candidate_intercept, targets
            )
            if candidate_loss <= current_loss + 1e-12:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break

        change = max(abs(candidate_slope - slope), abs(candidate_intercept - intercept))
        slope, intercept = candidate_slope, candidate_intercept
        if change < 1e-8 or np.linalg.norm(gradient, ord=np.inf) < 1e-9:
            converged = True
            break

    before = probability.astype(np.float64)
    after = affine_calibrate(probability, slope, intercept).astype(np.float64)
    return {
        "slope": slope,
        "intercept": intercept,
        "sampled_pixels": int(probability.size),
        "iterations": iterations,
        "converged": converged,
        "nll_before": nll_from_logits(source_logits, targets),
        "nll_after": nll_from_logits(slope * source_logits + intercept, targets),
        "brier_before": float(np.mean((before - targets) ** 2)),
        "brier_after": float(np.mean((after - targets) ** 2)),
        "mae_before": float(np.mean(np.abs(before - targets))),
        "mae_after": float(np.mean(np.abs(after - targets))),
    }


def deterministic_pair_sample(e0_probability, e4_probability, ground_truth, sample_id, seed, maximum):
    e0_flat = e0_probability.reshape(-1)
    e4_flat = e4_probability.reshape(-1)
    target_flat = ground_truth.reshape(-1).astype(np.float64)
    if e0_flat.shape != e4_flat.shape or e0_flat.shape != target_flat.shape:
        raise ValueError("Probability/target shapes differ during calibration sampling")
    if e0_flat.size <= maximum:
        indices = np.arange(e0_flat.size)
    else:
        token = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(token[:8], "little"))
        indices = np.sort(rng.choice(e0_flat.size, size=maximum, replace=False))
    return e0_flat[indices], e4_flat[indices], target_flat[indices]


def cluster_bootstrap(rows, seed=42, repetitions=20000):
    if not rows:
        raise ValueError("Patient bootstrap needs rows")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(float(row["delta_dice_vs_e0"]))
    groups = sorted(grouped)
    sums = np.array([np.sum(grouped[group]) for group in groups], dtype=np.float64)
    counts = np.array([len(grouped[group]) for group in groups], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    batch = 1000
    for start in range(0, repetitions, batch):
        stop = min(repetitions, start + batch)
        indices = rng.integers(0, len(groups), size=(stop - start, len(groups)))
        means[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "unit": "patient_group",
        "group_count": len(groups),
        "image_count": len(rows),
        "mean_delta": float(np.mean([row["delta_dice_vs_e0"] for row in rows])),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "repetitions": repetitions,
        "seed": seed,
    }


def verify_e6_failure(e6_selection_run, manifest_info, inference_info):
    info = e6.read_json(e6_selection_run / "run_info.json")
    config = e6.read_json(e6_selection_run / "selection_config.json")
    recorded_signature = config.pop("config_signature")
    if e6.signature(config) != recorded_signature:
        raise ValueError("E6 selection config changed")
    config["config_signature"] = recorded_signature
    if info.get("status") != "COMPLETED" or info.get("selection_status") != "NO_PASS":
        raise ValueError("E7-A expects the completed E6 NO_PASS result")
    if config.get("validation_status") != "NO_PASS":
        raise ValueError("E6 config is not the expected NO_PASS config")
    if config.get("development_manifest_signature") != manifest_info.get("manifest_signature"):
        raise ValueError("E6 selection used a different development manifest")
    if config.get("e0_checkpoint_sha256") != inference_info.get("e0_checkpoint_sha256"):
        raise ValueError("E6/E7-A E0 checkpoint mismatch")
    if config.get("e4_checkpoint_sha256") != inference_info.get("e4_checkpoint_sha256"):
        raise ValueError("E6/E7-A E4 checkpoint mismatch")
    return config


def calibration_samples(project, records, inference_run, index, seed, maximum):
    e0_samples, e4_samples, target_samples = [], [], []
    for record in records:
        ground_truth = e6.read_mask(e6.safe_under(project, record["mask"]))
        e0_probability = e6.load_probability(
            e6.probability_path(inference_run, index, "e0", record["sample_id"]),
            ground_truth.shape,
        )
        e4_probability = e6.load_probability(
            e6.probability_path(inference_run, index, "e4", record["sample_id"]),
            ground_truth.shape,
        )
        e0_sample, e4_sample, target_sample = deterministic_pair_sample(
            e0_probability,
            e4_probability,
            ground_truth,
            record["sample_id"],
            seed,
            maximum,
        )
        e0_samples.append(e0_sample)
        e4_samples.append(e4_sample)
        target_samples.append(target_sample)
    return np.concatenate(e0_samples), np.concatenate(e4_samples), np.concatenate(target_samples)


def fit_alpha_calibrators(e0_sample, e4_sample, targets):
    fitted = {}
    for alpha in ALPHAS:
        fused = ((1.0 - alpha) * e0_sample + alpha * e4_sample).astype(np.float32)
        fit = fit_affine(fused, targets)
        fit["alpha_e4"] = alpha
        fitted[alpha] = fit
        print(
            f"[calibrate alpha={alpha:.2f}] slope={fit['slope']:.6f} "
            f"intercept={fit['intercept']:.6f} NLL {fit['nll_before']:.6f}->{fit['nll_after']:.6f}",
            flush=True,
        )
    return fitted


def selected_per_image(project, records, inference_run, index, selected, fitted):
    rows = []
    alpha = selected["alpha"]
    threshold = selected["threshold"]
    calibrator = fitted[alpha]
    for record in records:
        ground_truth = e6.read_mask(e6.safe_under(project, record["mask"]))
        e0_probability = e6.load_probability(
            e6.probability_path(inference_run, index, "e0", record["sample_id"]), ground_truth.shape
        )
        e4_probability = e6.load_probability(
            e6.probability_path(inference_run, index, "e4", record["sample_id"]), ground_truth.shape
        )
        fused = ((1.0 - alpha) * e0_probability + alpha * e4_probability).astype(np.float32)
        calibrated = affine_calibrate(fused, calibrator["slope"], calibrator["intercept"])
        baseline = e6.probability_metrics(e0_probability, ground_truth, 0.5)
        candidate = e6.probability_metrics(calibrated, ground_truth, threshold)
        rows.append(
            {
                "sample_id": record["sample_id"],
                "group_id": record["group_id"],
                "source": record["source"],
                "area_bin": record["area_bin"],
                "e0_dice": baseline["dice"],
                "e7a_dice": candidate["dice"],
                "delta_dice_vs_e0": candidate["dice"] - baseline["dice"],
                "e0_iou": baseline["iou"],
                "e7a_iou": candidate["iou"],
                "e0_mae": baseline["mae"],
                "e7a_mae": candidate["mae"],
            }
        )
    return rows


def area_summary(rows):
    result = []
    for area in ("small_lt1pct", "medium_1to10pct", "large_ge10pct"):
        group = [row for row in rows if row["area_bin"] == area]
        if not group:
            continue
        result.append(
            {
                "area_bin": area,
                "count": len(group),
                "e0_dice": float(np.mean([row["e0_dice"] for row in group])),
                "e7a_dice": float(np.mean([row["e7a_dice"] for row in group])),
                "delta_dice_vs_e0": float(np.mean([row["delta_dice_vs_e0"] for row in group])),
                "e0_mae": float(np.mean([row["e0_mae"] for row in group])),
                "e7a_mae": float(np.mean([row["e7a_mae"] for row in group])),
            }
        )
    return result


def command_select(args):
    project = Path(args.project).expanduser().resolve()
    manifest_run = e6.resolve(project, args.manifest_run)
    inference_run = e6.resolve(project, args.inference_run)
    e6_selection_run = e6.resolve(project, args.e6_selection_run)
    output = e6.resolve(project, args.output)
    if output.exists():
        raise FileExistsError("Output already exists: " + str(output))
    output.mkdir(parents=True)
    info = {"status": "RUNNING", "run_kind": "E7A_SELECT", "started_at": e6.utc_now()}
    e6.write_json(output / "run_info.json", info)
    try:
        manifest_info, records = e6.verify_development_manifest(project, manifest_run)
        inference_info, index = e6.verify_inference(project, manifest_info, records, inference_run)
        e6_config = verify_e6_failure(e6_selection_run, manifest_info, inference_info)
        calibration_records = [record for record in records if record["split"] == "calibration"]
        selection_records = [record for record in records if record["split"] == "selection"]

        e0_sample, e4_sample, targets = calibration_samples(
            project,
            calibration_records,
            inference_run,
            index,
            args.seed,
            args.max_pixels_per_image,
        )
        fitted = fit_alpha_calibrators(e0_sample, e4_sample, targets)

        baseline_acc = e6.empty_accumulator()
        candidate_acc = {
            (alpha, threshold): e6.empty_accumulator()
            for alpha in ALPHAS
            for threshold in THRESHOLDS
        }
        for position, record in enumerate(selection_records, 1):
            ground_truth = e6.read_mask(e6.safe_under(project, record["mask"]))
            e0_probability = e6.load_probability(
                e6.probability_path(inference_run, index, "e0", record["sample_id"]),
                ground_truth.shape,
            )
            e4_probability = e6.load_probability(
                e6.probability_path(inference_run, index, "e4", record["sample_id"]),
                ground_truth.shape,
            )
            e6.add_metrics(
                baseline_acc,
                record["source"],
                e6.probability_metrics(e0_probability, ground_truth, 0.5),
            )
            for alpha in ALPHAS:
                fused = ((1.0 - alpha) * e0_probability + alpha * e4_probability).astype(np.float32)
                calibrator = fitted[alpha]
                calibrated = affine_calibrate(fused, calibrator["slope"], calibrator["intercept"])
                for threshold in THRESHOLDS:
                    e6.add_metrics(
                        candidate_acc[(alpha, threshold)],
                        record["source"],
                        e6.probability_metrics(calibrated, ground_truth, threshold),
                    )
            if position % 25 == 0 or position == len(selection_records):
                print(f"[search {position}/{len(selection_records)}] {record['sample_id']}", flush=True)

        baseline = e6.finish_accumulator(baseline_acc)
        candidates = []
        by_source_rows = []
        for alpha in ALPHAS:
            for threshold in THRESHOLDS:
                candidate = e6.finish_accumulator(candidate_acc[(alpha, threshold)])
                candidate.update(
                    alpha=alpha,
                    threshold=threshold,
                    slope=fitted[alpha]["slope"],
                    intercept=fitted[alpha]["intercept"],
                )
                e6.gate_candidate(candidate, baseline, DEFAULT_GATES)
                candidates.append(candidate)
                for source, score in sorted(candidate["source_dice"].items()):
                    by_source_rows.append(
                        {
                            "alpha": alpha,
                            "threshold": threshold,
                            "source": source,
                            "e7a_dice": score,
                            "e0_dice": baseline["source_dice"][source],
                            "delta_vs_e0": score - baseline["source_dice"][source],
                        }
                    )
        selected, status = e6.choose_candidate(candidates)
        flat_rows = [
            {key: value for key, value in candidate.items() if key not in ("source_dice", "checks")}
            for candidate in candidates
        ]
        calibration_rows = [fitted[alpha] for alpha in ALPHAS]
        per_image_rows = selected_per_image(
            project, selection_records, inference_run, index, selected, fitted
        )
        bootstrap = cluster_bootstrap(per_image_rows, seed=args.seed, repetitions=args.bootstrap_repetitions)
        area_rows = area_summary(per_image_rows)
        e6.write_csv(output / "affine_calibration.csv", calibration_rows)
        e6.write_csv(output / "grid_search.csv", flat_rows)
        e6.write_csv(output / "grid_by_source.csv", by_source_rows)
        e6.write_csv(output / "selected_per_image.csv", per_image_rows)
        e6.write_csv(output / "selected_by_area.csv", area_rows)
        e6.write_json(output / "patient_bootstrap.json", bootstrap)

        config = {
            "format_version": 1,
            "experiment": "E7A_POST_FUSION_AFFINE_CALIBRATION",
            "validation_status": status,
            "selection_rule": (
                "Fit slope/intercept separately for each alpha on calibration only. Among candidates "
                "passing the unchanged E6 gates, maximize selection Dice, then worst-source delta, "
                "then minimize MAE, failure tail, alpha, and distance from threshold 0.5. If none "
                "pass, record the nearest candidate and block formal test."
            ),
            "alpha_e4": selected["alpha"],
            "weight_e0": 1.0 - selected["alpha"],
            "threshold": selected["threshold"],
            "affine_slope": fitted[selected["alpha"]]["slope"],
            "affine_intercept": fitted[selected["alpha"]]["intercept"],
            "gates": DEFAULT_GATES,
            "selection_baseline_e0": baseline,
            "selected_validation_metrics": selected,
            "patient_bootstrap": bootstrap,
            "development_manifest_signature": manifest_info["manifest_signature"],
            "development_manifest_sha256": manifest_info["manifest_sha256"],
            "development_inference_run_info_sha256": e6.file_hash(inference_run / "run_info.json"),
            "e6_no_pass_config_sha256": e6.file_hash(e6_selection_run / "selection_config.json"),
            "e6_no_pass_config_signature": e6_config["config_signature"],
            "e0_checkpoint_sha256": inference_info["e0_checkpoint_sha256"],
            "e4_checkpoint_sha256": inference_info["e4_checkpoint_sha256"],
            "search_space": {"alpha_e4": list(ALPHAS), "threshold": list(THRESHOLDS)},
            "calibration_partition_count": len(calibration_records),
            "selection_partition_count": len(selection_records),
            "created_at": e6.utc_now(),
        }
        config["config_signature"] = e6.signature(config)
        e6.write_json(output / "selection_config.json", config)
        info.update(
            {
                "status": "COMPLETED",
                "finished_at": e6.utc_now(),
                "selection_status": status,
                "selected": {
                    "alpha_e4": selected["alpha"],
                    "threshold": selected["threshold"],
                    "affine_slope": fitted[selected["alpha"]]["slope"],
                    "affine_intercept": fitted[selected["alpha"]]["intercept"],
                },
                "config_signature": config["config_signature"],
            }
        )
        e6.write_json(output / "run_info.json", info)
        print("E7-A selection status:", status)
        print("Locked configuration:", output / "selection_config.json")
        if status != "PASS":
            print("No candidate passed the unchanged gates. Formal test is blocked.")
    except BaseException as error:
        info.update(status="FAILED", finished_at=e6.utc_now(), error=f"{type(error).__name__}: {error}")
        e6.write_json(output / "run_info.json", info)
        raise


def summarize_test(rows):
    result = []
    for dataset in ("ALL",) + e6.DATASETS:
        group = rows if dataset == "ALL" else [row for row in rows if row["dataset"] == dataset]
        item = {"dataset": dataset, "count": len(group)}
        for model in ("e0", "e4", "e5", "e7a"):
            for metric in ("dice", "iou", "mae"):
                item[f"{model}_{metric}"] = float(np.mean([row[f"{model}_{metric}"] for row in group]))
            dice = np.array([row[f"{model}_dice"] for row in group])
            item[f"{model}_median_dice"] = float(np.median(dice))
            item[f"{model}_dice_lt05"] = int((dice < 0.5).sum())
            item[f"{model}_zero_dice"] = int((dice == 0).sum())
        for reference in ("e0", "e5"):
            for metric in ("dice", "iou", "mae"):
                item[f"delta_{metric}_vs_{reference}"] = (
                    item[f"e7a_{metric}"] - item[f"{reference}_{metric}"]
                )
        result.append(item)
    return result


def command_apply_test(args):
    project = Path(args.project).expanduser().resolve()
    config_path = e6.resolve(project, args.config)
    output = e6.resolve(project, args.output)
    e0_run = e6.resolve(project, args.e0_run)
    e4_run = e6.resolve(project, args.e4_run)
    e4_eval = e6.resolve(project, args.e4_eval)
    if output.exists():
        raise FileExistsError("Output already exists: " + str(output))
    config = e6.read_json(config_path)
    recorded_signature = config.pop("config_signature")
    if e6.signature(config) != recorded_signature:
        raise ValueError("Locked E7-A selection config changed")
    config["config_signature"] = recorded_signature
    if config.get("validation_status") != "PASS" and not args.allow_exploratory:
        raise ValueError("Validation status is NO_PASS; formal test is blocked")

    output.mkdir(parents=True)
    info = {
        "status": "RUNNING",
        "run_kind": "FULL_E7A" if config["validation_status"] == "PASS" else "EXPLORATORY_E7A",
        "started_at": e6.utc_now(),
        "config_signature": recorded_signature,
    }
    e6.write_json(output / "run_info.json", info)
    try:
        manifest, e0_rows, e4_rows, e0_info, e4_info = e6.validate_test_sources(
            project, e0_run, e4_run, e4_eval
        )
        if config["e0_checkpoint_sha256"] != e0_info["checkpoint_sha256"]:
            raise ValueError("Selection used a different E0 checkpoint")
        if config["e4_checkpoint_sha256"] != e4_info["final_checkpoint_sha256"]:
            raise ValueError("Selection used a different E4 checkpoint")

        rows = []
        fingerprints = []
        for position, record in enumerate(manifest, 1):
            sample_id = record["sample_id"]
            image_path = e6.safe_under(project, record["image"])
            mask_path = e6.safe_under(project, record["mask"])
            if e6.file_hash(image_path) != record["image_sha256"] or e6.file_hash(mask_path) != record["mask_sha256"]:
                raise ValueError("Frozen test file changed: " + sample_id)
            ground_truth = e6.read_mask(mask_path)
            e0_path = e6.safe_under(e0_run, e0_rows[sample_id]["probability_path"])
            e4_path = e6.safe_under(e4_eval, e4_rows[sample_id]["probability_path"])
            e0_probability = e6.load_probability(e0_path, ground_truth.shape)
            e4_probability = e6.load_probability(e4_path, ground_truth.shape)
            e5_probability = (0.5 * e0_probability + 0.5 * e4_probability).astype(np.float32)
            alpha = config["alpha_e4"]
            fused = ((1.0 - alpha) * e0_probability + alpha * e4_probability).astype(np.float32)
            e7a_probability = affine_calibrate(
                fused, config["affine_slope"], config["affine_intercept"]
            )
            metrics = {
                "e0": e6.probability_metrics(e0_probability, ground_truth, 0.5),
                "e4": e6.probability_metrics(e4_probability, ground_truth, 0.5),
                "e5": e6.probability_metrics(e5_probability, ground_truth, 0.5),
                "e7a": e6.probability_metrics(e7a_probability, ground_truth, config["threshold"]),
            }
            probability_relative = Path("probabilities") / (sample_id + ".npz")
            mask_relative = Path("masks") / (sample_id + ".png")
            for relative in (probability_relative, mask_relative):
                (output / relative.parent).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output / probability_relative, probability=e7a_probability)
            Image.fromarray(
                (e7a_probability >= config["threshold"]).astype(np.uint8) * 255
            ).save(output / mask_relative)
            row = {
                "sample_id": sample_id,
                "dataset": record["dataset"],
                "height": ground_truth.shape[0],
                "width": ground_truth.shape[1],
                "gt_area_fraction": float(ground_truth.mean()),
            }
            for model, model_metrics in metrics.items():
                row.update({f"{model}_{key}": value for key, value in model_metrics.items()})
            row.update(
                {
                    "delta_dice_vs_e0": metrics["e7a"]["dice"] - metrics["e0"]["dice"],
                    "delta_dice_vs_e5": metrics["e7a"]["dice"] - metrics["e5"]["dice"],
                    "probability_path": probability_relative.as_posix(),
                    "mask_path": mask_relative.as_posix(),
                }
            )
            rows.append(row)
            fingerprints.append(
                {
                    "sample_id": sample_id,
                    "e0_probability_sha256": e6.file_hash(e0_path),
                    "e4_probability_sha256": e6.file_hash(e4_path),
                    "e7a_probability_sha256": e6.file_hash(output / probability_relative),
                }
            )
            if position % 50 == 0 or position == len(manifest):
                print(f"[E7-A test {position}/{len(manifest)}] {sample_id}", flush=True)

        e6.write_csv(output / "per_image.csv", rows)
        e6.write_csv(output / "input_fingerprints.csv", fingerprints)
        summary = summarize_test(rows)
        e6.write_csv(output / "summary.csv", summary)
        all_summary = summary[0]
        gates = config["gates"]
        dataset_deltas = {
            row["dataset"]: row["delta_dice_vs_e0"]
            for row in summary
            if row["dataset"] != "ALL"
        }
        checks = {
            "mean_dice_gain": all_summary["delta_dice_vs_e0"] >= gates["min_mean_dice_gain"],
            "all_datasets": min(dataset_deltas.values()) >= gates["min_source_dice_delta"],
            "mae": all_summary["delta_mae_vs_e0"] <= gates["max_mae_increase"],
            "failure_tail": (
                all_summary["e7a_dice_lt05"] - all_summary["e0_dice_lt05"]
                <= gates["max_dice_lt05_increase"]
            ),
        }
        decision = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "gates": gates,
            "dataset_dice_deltas": dataset_deltas,
            "paired_bootstrap": {
                "e7a_minus_e0_dice": e6.paired_bootstrap(
                    [row["e7a_dice"] - row["e0_dice"] for row in rows]
                ),
                "e7a_minus_e5_dice": e6.paired_bootstrap(
                    [row["e7a_dice"] - row["e5_dice"] for row in rows]
                ),
            },
            "interpretation": (
                "PASS means the predeclared held-out test gates were met. FAIL does not authorize "
                "retuning on the test set."
            ),
        }
        e6.write_json(output / "decision.json", decision)
        e6.write_json(output / "locked_config.json", config)
        e6.write_json(output / "manifest.json", manifest)
        info.update(
            {
                "status": "COMPLETED",
                "finished_at": e6.utc_now(),
                "processed_count": len(rows),
                "decision": decision["status"],
                "e0_checkpoint_sha256": e0_info["checkpoint_sha256"],
                "e4_checkpoint_sha256": e4_info["final_checkpoint_sha256"],
                "test_manifest_sha256": e6.file_hash(e0_run / "manifest.json"),
                "timing": "Offline probability fusion/calibration/audit/output; not neural inference FPS",
            }
        )
        e6.write_json(output / "run_info.json", info)
        print("FULL E7-A complete. Decision:", decision["status"])
        print("Output:", output)
    except BaseException as error:
        info.update(status="FAILED", finished_at=e6.utc_now(), error=f"{type(error).__name__}: {error}")
        e6.write_json(output / "run_info.json", info)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(PROJECT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--manifest-run", default="runs/e6_development_seed42")
    select.add_argument("--inference-run", default="runs/e6_development_inference_seed42")
    select.add_argument("--e6-selection-run", default="runs/e6_selection_seed42")
    select.add_argument("--output", default="runs/e7a_selection_seed42")
    select.add_argument("--seed", type=int, default=42)
    select.add_argument("--max-pixels-per-image", type=int, default=8192)
    select.add_argument("--bootstrap-repetitions", type=int, default=20000)

    apply_test = subparsers.add_parser("apply-test")
    apply_test.add_argument("--config", default="runs/e7a_selection_seed42/selection_config.json")
    apply_test.add_argument("--output", default="runs/e7a_final_seed42")
    apply_test.add_argument("--e0-run", default="runs/e0_20261004_221345_872444")
    apply_test.add_argument("--e4-run", default="runs/e4_final_seed42")
    apply_test.add_argument(
        "--e4-eval", default="runs/e4_final_seed42/evaluation_cuda_20261006_235221_022843"
    )
    apply_test.add_argument("--allow-exploratory", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "seed") and args.seed < 0:
        parser.error("seed must be nonnegative")
    if hasattr(args, "max_pixels_per_image") and args.max_pixels_per_image < 256:
        parser.error("max-pixels-per-image must be at least 256")
    if hasattr(args, "bootstrap_repetitions") and args.bootstrap_repetitions < 1000:
        parser.error("bootstrap-repetitions must be at least 1000")
    {"select": command_select, "apply-test": command_apply_test}[args.command](args)


if __name__ == "__main__":
    main()
