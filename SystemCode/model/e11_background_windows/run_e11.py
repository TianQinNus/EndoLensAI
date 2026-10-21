"""E11: one bounded background-loss revision of B; user-run long training."""
import argparse
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "e8_repeat_windows"))
import run_e8_repeat as repeat
import train_e8 as trainer
from common import (read_json, write_json, write_csv, file_hash, signature, metric_summary,
                    validation_score, code_hashes as parent_hashes)
from engine import require_cuda, environment
from background_loss import loss_parts, enrich_rows, STRENGTH, POWER

PLAN = PROJECT / "runs/e11_background_plan.json"
OUTPUT = PROJECT / "runs/e11_background_summary"
SEEDS = (42, 17, 2024)
SOURCES = repeat.SOURCES
ARM = "B_background"
METRICS = (*repeat.METRICS, "macro_fp_image_fraction", "precision", "recall", "small_recall")
GATES = {"min_macro_dice_gain_vs_B_each_seed": .003, "max_source_dice_loss_vs_B": .003,
         "max_source_mae_increase_vs_B": .003, "min_relative_macro_fp_reduction": .05,
         "max_each_source_fp_increase": 0., "max_each_source_recall_loss": .01,
         "max_small_recall_loss": .01, "max_each_source_failure_count_increase": 0,
         "require_all_three_seeds": True}


def stage_hashes():
    return {**parent_hashes(), **{p.relative_to(PROJECT).as_posix(): file_hash(p) for p in sorted(PACKAGE.glob("*.py"))}}


def make_config(seed, smoke=False):
    if seed not in SEEDS:
        raise ValueError("Only seeds42/17/2024 permitted")
    config = trainer.make_config("B_expanded", seed, smoke)
    config.update(stage="E11_BACKGROUND", background_strength=STRENGTH, background_power=POWER,
                  background_detach=True, background_weight_max=2.,
                  loss="Original BCE+per-image soft Dice + mean_all_pixels((1-y)*stopgrad(sigmoid(z))^2*BCE); lambda1 constant")
    return config


@contextmanager
def training_adapter():
    """Reuse original loop/optimizer/sampler/augmentation, adapting loss and reporting only."""
    original = (trainer.per_image_loss, trainer.code_hashes, trainer.write_json, trainer.evaluate_records)
    pending = {"seen": 0, "base": 0., "extra": 0.}
    def weighted_loss(logits, target):
        total, weighted, dice, base, extra = loss_parts(logits, target)
        pending["seen"] += len(logits)
        pending["base"] += float(base.detach().sum().item())
        pending["extra"] += float(extra.detach().sum().item())
        return total, weighted, dice
    def stage_writer(path, value):
        if Path(path).name == "run_info.json":
            value.update(run_kind="E11_GPU_SMOKE" if value["config"]["smoke"] else "E11_DEVELOPMENT",
                         stage="E11_BACKGROUND", note="Only bounded pixelwise background BCE weighting changed; no C6/FAR/E0 metrics or fusion. Original B preserved.")
            if pending["seen"]:
                value["loss_diagnostics_last_round"] = {"round": value["completed_epochs"], "samples": pending["seen"],
                    "base_bce": pending["base"] / pending["seen"], "background_extra": pending["extra"] / pending["seen"],
                    "foreground_bce_weight": 1., "background_bce_weight_range": [1., 2.]}
                pending.update(seen=0, base=0., extra=0.)
        original[2](path, value)
    def evaluation(*args, **kwargs):
        summary, rows = original[3](*args, **kwargs)
        if kwargs.get("output") is not None:
            raise ValueError("Training adapter does not support inference exports/test")
        return summary, enrich_rows(rows)
    trainer.per_image_loss, trainer.code_hashes = weighted_loss, stage_hashes
    trainer.write_json, trainer.evaluate_records = stage_writer, evaluation
    try:
        yield
    finally:
        trainer.per_image_loss, trainer.code_hashes, trainer.write_json, trainer.evaluate_records = original


def run_path(seed, smoke=False):
    return PROJECT / "runs" / f"e11_{'smoke_' if smoke else ''}B_background_seed{seed}"


def protect_file(path, protected):
    protected[path.relative_to(PROJECT).as_posix()] = file_hash(path)


def prepare():
    if PLAN.exists():
        check()
        return
    if OUTPUT.exists() or any(run_path(s).exists() or run_path(s, True).exists() for s in SEEDS):
        raise ValueError("E11 output predates protocol; preserve and inspect")
    repeat.verify()
    _, bundle, manifests, protocol = repeat.check()
    protected = {}
    for seed in SEEDS:
        path, _, _, _ = repeat.verified_run("B_expanded", seed, bundle, manifests, protocol)
        for name in ("run_info.json", "checkpoints/best.pth", "train_history.csv", "validation_best_per_image.csv"):
            protect_file(path / name, protected)
    for path in (repeat.PLAN_PATH, repeat.OUTPUT / "audit.json", repeat.OUTPUT / "stability.json",
                 repeat.e8.PROTOCOL_PATH, PROJECT / "e8_repeat_windows/run_e8_repeat.py"):
        protect_file(path, protected)
    # Historical fusion failures remain unchanged, without reusing their masks for training.
    for name in ("runs/e10b_fusion_plan.json", "runs/e10b_fusion_seed42/locked_config.json",
                 "runs/e10c_temperature_plan.json", "runs/e10c_temperature_seed42/cross_validation/decision.json",
                 "runs/e10d_rescue_compat_plan.json", "runs/e10d_rescue_compat_seed42/diagnosis/decision.json",
                 "runs/e10d_rescue_compat_seed42/diagnosis/audit.json"):
        protect_file(PROJECT / name, protected)
    if read_json(PROJECT / "runs/e10d_rescue_compat_seed42/diagnosis/decision.json")["status"] != "NO_FEASIBILITY_PASS_RETAIN_B":
        raise ValueError("Require preserved E10-D diagnosis before E11")
    records = manifests["original_train"] + manifests["new_train"]
    if {r["dataset"] for r in records} != {"original_training", *(f"PolypGen-C{i}" for i in range(1, 5))}:
        raise ValueError("Only original training/C1-C4 may supply training targets")
    plan = {"stage": "E11_BOUNDED_BACKGROUND_SUPERVISION", "frozen_at_singapore": repeat.now(),
            "source_hashes": stage_hashes(), "protected_artifacts": protected,
            "bundle_signature": bundle["bundle_signature"], "parent_protocol_signature": protocol["protocol_signature"],
            "configs": {str(s): make_config(s) for s in SEEDS}, "seeds": list(SEEDS), "gates": GATES,
            "data_counts": bundle["counts"], "comparator": "Same-seed original E8 B, existing three verified complete runs, never overwrite or choose the best seed",
            "training_change": "Background BCE weight1+stopgrad(p)^2 bounded1..2; foreground BCE1; original Dice unchanged. Constant lambda1, no tuning. Soft pixel-level emphasis, NOT image-level difficulty sampling or top-k selection.",
            "training_targets": "Only each sampled training image's existing binary GT and current model probability. No validation/FAR/C5/C6 scores/masks/crops/negative frames enter training or weights.",
            "controls": "Same ImageNet encoder initializer/random decoder, same 50/50 source mass and C1-C4 equal mass, 352/strong augmentation, optimizer/lr, 100x1160 draws, batch2/accum4/AMP. No B-checkpoint finetuning or added training budget.",
            "selection": "Unchanged earliest highest0.5*oldDice+0.5*C5Dice; safety is assessed afterwards, not used to cherry-pick a different checkpoint",
            "fp_metric": "Image mean FP/(H*W) on positive validation; macro is half each source mean. NOT negative-frame false alarm rate or specificity.",
            "recall_protection": "Image-equal TP/(TP+FN); each source loss<=.01; smallGT<1% image-equal recall loss<=.01. Failure counts protected each source.",
            "decision": "All three seeds must satisfy all new E11 gates. If any fails retain original B; no lambda/power/threshold/seed/checkpoint search. Old E8-R/other NO_PASS outcomes unchanged.",
            "deployment_seed": 42, "data_role": "Adaptive development after FAR error inspection. Fixed old/C5 validation already reused; not independent confirmation. FAR never becomes training or new blind test.",
            "restrictions": {"new_data": 0, "negative_frame_training": 0, "E0_training_or_inference": 0,
                             "FAR_new_model_evaluations": 0, "C6_evaluations": 0, "fusion": 0,
                             "formal_runs_to_be_started_by_user": 3},
            "visual_review": "Optional cached FAR30 low-Dice plates only for error/label review; no new inference, no transfer to training. Spatial GT band5 at352 is descriptive only, never an input to training."}
    plan["plan_signature"] = signature(plan)
    write_json(PLAN, plan)
    print("E11_PLAN_FROZEN; THREE_FORMAL_RUNS_NOT_STARTED", flush=True)


def check():
    plan = read_json(PLAN)
    if (signature({k: v for k, v in plan.items() if k != "plan_signature"}) != plan["plan_signature"]
            or plan["source_hashes"] != stage_hashes() or plan["gates"] != GATES
            or plan["configs"] != {str(s): make_config(s) for s in SEEDS}):
        raise ValueError("E11 frozen code/config/gates changed")
    bundle, manifests, protocol = repeat.e8.preflight()
    if bundle["bundle_signature"] != plan["bundle_signature"] or protocol["protocol_signature"] != plan["parent_protocol_signature"]:
        raise ValueError("Parent data/protocol differs")
    for name, expected in plan["protected_artifacts"].items():
        if file_hash(PROJECT / name) != expected:
            raise ValueError("Protected B/historical artifact changed: " + name)
    print("E11_CHECK_PASS; TRAINING_ONLY_BACKGROUND_WEIGHTS; B/E0/C6/FAR_FROZEN", flush=True)
    return plan, bundle, manifests, protocol


def validate_rows(rows, expected):
    indexed = {r["sample_id"]: r for r in expected}
    if len(rows) != len(indexed) or {r["sample_id"] for r in rows} != set(indexed):
        raise ValueError("Incomplete/duplicate validation membership")
    for r in rows:
        original = indexed[r["sample_id"]]
        if r["dataset"] != original["dataset"]:
            raise ValueError("Validation source differs")
        for key in ("dice", "iou", "mae"):
            if not np.isfinite(float(r[key])) or not 0 <= float(r[key]) <= 1:
                raise ValueError("Invalid validation metric")
    enriched = enrich_rows(rows)
    for old, new in zip(rows, enriched):
        tp, fp, fn = (int(old[k]) for k in ("tp", "fp", "fn"))
        if not np.isclose(float(old["dice"]), 2*tp/(2*tp+fp+fn), rtol=0, atol=1e-12):
            raise ValueError("Dice/count mismatch")
        for key in ("precision", "recall", "fp_image_fraction", "fn_image_fraction"):
            if key in old and not np.isclose(float(old[key]), new[key], rtol=0, atol=1e-12):
                raise ValueError("Derived background/recall metric mismatch")
    return enriched


def summarize_rows(rows):
    rows = enrich_rows(rows)
    small = [r for r in rows if float(r["gt_area_fraction"]) < .01]
    if not small or any(not any(r["dataset"] == source for r in rows) for source in SOURCES):
        raise ValueError("Require both sources and prespecified small-target stratum")
    return {"macro_dice": validation_score(rows), **metric_summary(rows),
            "macro_fp_image_fraction": float(np.mean([np.mean([r["fp_image_fraction"] for r in rows if r["dataset"] == s]) for s in SOURCES])),
            "precision": float(np.mean([r["precision"] for r in rows])), "recall": float(np.mean([r["recall"] for r in rows])),
            "small_recall": float(np.mean([r["recall"] for r in small]))}


def verified_run(seed, plan, bundle, manifests, smoke=False):
    path, config = run_path(seed, smoke), make_config(seed, smoke)
    info = read_json(path / "run_info.json")
    if info["status"] != "TRAINED" or info["run_kind"] != ("E11_GPU_SMOKE" if smoke else "E11_DEVELOPMENT"):
        raise ValueError("Require complete E11 run")
    if info["config"] != config or info["source_hashes"] != plan["source_hashes"] or info["bundle_signature"] != bundle["bundle_signature"]:
        raise ValueError("E11 run provenance differs")
    for name, key in (("checkpoints/best.pth", "best_checkpoint_sha256"), ("validation_best_per_image.csv", "best_validation_csv_sha256")):
        if file_hash(path / name) != info[key]:
            raise ValueError("Committed result changed")
    cp = torch.load(path / "checkpoints/best.pth", map_location="cpu", weights_only=True)
    if any(cp[key] != info[key] for key in ("config", "source_hashes", "bundle_signature")) or cp["epoch"] != info["best_epoch"]:
        raise ValueError("Checkpoint provenance differs")
    del cp
    baseline_path = repeat.e8.run_path("B_expanded", seed)
    base = read_json(baseline_path / "run_info.json")
    if (info["environment"]["versions"] != base["environment"]["versions"] or info["environment"].get("gpu") != base["environment"].get("gpu")
            or info["initial_encoder_sha256"] != base["initial_encoder_sha256"] or info["model_device"] != "cuda:0"
            or not info["amp_enabled"] or any(info["last_device_proof"][k] != "cuda:0" for k in ("model", "images", "masks", "logits"))):
        raise ValueError("Environment/ImageNet/GPU proof mismatch")
    training_records = manifests["original_train"] + manifests["new_train"]
    validation = manifests["validation"]
    if smoke:
        validation = [r for source in SOURCES for r in [x for x in validation if x["dataset"] == source][:2]]
    if (read_json(path / "train_manifest.json") != training_records or read_json(path / "validation_manifest.json") != validation
            or info["training_unique_images"] != 2269 or info["completed_epochs"] != config["epochs"] or info["formal_holdout_evaluations"] != 0):
        raise ValueError("Data role or budget differs")
    history = repeat.read_csv(path / "train_history.csv")
    if len(history) != config["epochs"] or [int(r["round"]) for r in history] != list(range(1, config["epochs"]+1)):
        raise ValueError("Incomplete round history")
    parent_history = repeat.read_csv(baseline_path / "train_history.csv")
    probs, _ = trainer.sampling_probabilities(training_records, np.full(len(training_records), np.nan), config, 1)
    updates_per_round = int(np.ceil(config["samples_per_round"] / (config["batch_size"]*config["accumulation"])))
    for r in history:
        epoch = int(r["round"])
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 808]))
        draws = rng.choice(len(training_records), config["samples_per_round"], replace=True, p=probs)
        if (trainer.hashlib_sha256_indices(draws) != r["sample_indices_sha256"] or int(r["hard_boosted_images"]) != 0
                or int(r["train_draws"]) != config["samples_per_round"]
                or int(r["optimizer_updates"])+int(r["amp_skipped_updates"]) != updates_per_round):
            raise ValueError("Sampler/budget/hard-image mining mismatch")
        if not smoke and r["sample_indices_sha256"] != parent_history[epoch-1]["sample_indices_sha256"]:
            raise ValueError("Training draw sequence differs from same-seed original B")
    for key, total in (("optimizer_updates", "optimizer_updates_total"), ("amp_skipped_updates", "amp_skipped_updates_total")):
        if sum(int(r[key]) for r in history) != info[total]:
            raise ValueError("Update total mismatch")
    earliest = max(history, key=lambda r: float(r["validation_macro_dice"]))
    if int(earliest["round"]) != info["best_epoch"]:
        raise ValueError("Not earliest macro-Dice maximum")
    rows = validate_rows(repeat.read_csv(path / "validation_best_per_image.csv"), validation)
    if not np.isclose(validation_score(rows), info["best_validation_macro_dice"], rtol=0, atol=1e-12):
        raise ValueError("Best validation score mismatch")
    diagnostic = info["loss_diagnostics_last_round"]
    if diagnostic["samples"] != config["samples_per_round"] or diagnostic["background_extra"] <= 0 or diagnostic["foreground_bce_weight"] != 1:
        raise ValueError("No active bounded background supervision proof")
    return path, info, rows, {"arm": ARM, "seed": seed, "best_round": info["best_epoch"], **(summarize_rows(rows) if not smoke else {}),
                             "round100_macro_dice": float(history[-1]["validation_macro_dice"]),
                             "actual_optimizer_updates": info["optimizer_updates_total"], "amp_skipped_updates": info["amp_skipped_updates_total"]}


def continuation_action(path, config, hashes, bundle_signature):
    if not path.exists():
        return "train"
    info = read_json(path / "run_info.json")
    if info["config"] != config or info["source_hashes"] != hashes or info["bundle_signature"] != bundle_signature:
        raise ValueError("Existing run differs; never overwrite")
    if info["status"] == "TRAINED":
        return "skip"
    if info["status"] in ("TRAINING", "INITIALIZING"):
        raise ValueError("May still be active; inspect terminal before recovery. Never launch a second process.")
    if info["status"] not in ("FAILED", "INTERRUPTED") or not (path / "checkpoints/last.pth").is_file():
        raise ValueError("No safe committed-round resume; preserve files and request guidance")
    return "resume"


def train_one(seed, workers=2, smoke=False):
    if smoke and seed != 42:
        raise ValueError("Smoke only seed42")
    with repeat.serial_training_lock():
        plan, bundle, manifests, _ = check()
        device = require_cuda()
        baseline = read_json(repeat.e8.run_path("B_expanded", seed) / "run_info.json")
        env = environment(device)
        if env["versions"] != baseline["environment"]["versions"] or env.get("gpu") != baseline["environment"].get("gpu"):
            raise ValueError("Environment differs from paired B")
        config, path = make_config(seed, smoke), run_path(seed, smoke)
        action = continuation_action(path, config, plan["source_hashes"], bundle["bundle_signature"])
        if action == "skip":
            verified_run(seed, plan, bundle, manifests, smoke)
            print("EXISTING_COMPLETE_VERIFIED_PRESERVED", path, flush=True)
            return
        if not smoke:
            verified_run(42, plan, bundle, manifests, smoke=True)
        print("E11_TRAINING_START", seed, action, "SMOKE" if smoke else "USER100_ROUNDS", flush=True)
        with training_adapter():
            trainer.train(path, config, bundle, manifests, workers=workers, resume=action == "resume")
        verified_run(seed, plan, bundle, manifests, smoke)


def development_gates(candidate_rows, base_rows):
    candidate, base = summarize_rows(candidate_rows), summarize_rows(base_rows)
    gates = {"macro_dice_gain": candidate["macro_dice"] >= base["macro_dice"] + .003 - 1e-12,
             "macro_fp_reduced_5pct": base["macro_fp_image_fraction"] > 0 and candidate["macro_fp_image_fraction"] <= .95*base["macro_fp_image_fraction"] + 1e-12,
             "small_target_recall_protected": candidate["small_recall"] >= base["small_recall"] - .01 - 1e-12}
    details = []
    for source in SOURCES:
        cr, br = [enrich_rows([r for r in values if r["dataset"] == source]) for values in (candidate_rows, base_rows)]
        cm, bm = metric_summary(cr), metric_summary(br)
        recall_delta = float(np.mean([r["recall"] for r in cr])-np.mean([r["recall"] for r in br]))
        fp_delta = float(np.mean([r["fp_image_fraction"] for r in cr])-np.mean([r["fp_image_fraction"] for r in br]))
        gates[source+"_dice"] = cm["dice"] >= bm["dice"] - .003 - 1e-12
        gates[source+"_mae"] = cm["mae"] <= bm["mae"] + .003 + 1e-12
        gates[source+"_recall"] = recall_delta >= -.01 - 1e-12
        gates[source+"_fp"] = fp_delta <= 1e-12
        for key in ("dice_lt05", "dice_lt01", "zero_dice"):
            gates[source+"_"+key] = cm[key] <= bm[key]
        details.append({"source": source, "dice_delta": cm["dice"]-bm["dice"], "mae_delta": cm["mae"]-bm["mae"],
                        "fp_image_fraction_delta": fp_delta, "recall_delta": recall_delta,
                        **{key+"_delta": cm[key]-bm[key] for key in ("dice_lt05", "dice_lt01", "zero_dice")}})
    return gates, details


def complete_decision(decisions):
    if len(decisions) != 3 or {r["seed"] for r in decisions} != set(SEEDS):
        raise ValueError("Need all three complete paired seeds, never best-seed selection")
    passed = all(all(r["gates"].values()) for r in decisions)
    return {"stage": "E11_BACKGROUND_DEVELOPMENT", "status": "DEVELOPMENT_PASS_NEEDS_INDEPENDENT_CONFIRMATION" if passed else "NO_PASS_RETAIN_B",
            "selected_recipe": ARM if passed else "B_expanded", "deployment_seed": 42, "per_seed": decisions,
            "formal_test_allowed": False, "C6_evaluations": 0, "FAR_evaluations": 0,
            "interpretation": "Reused validation, seed variability only; no new patient-level confirmation or negative-frame safety."}


def summarize():
    if OUTPUT.exists():
        verify_summary()
        return
    plan, bundle, manifests, protocol = check()
    results, contrasts, decisions, sources, inputs = [], [], [], [], {}
    for seed in SEEDS:
        path, _, rows, result = verified_run(seed, plan, bundle, manifests)
        bp, _, br, baseline = repeat.verified_run("B_expanded", seed, bundle, manifests, protocol)
        br = validate_rows(br, manifests["validation"])
        baseline.update(summarize_rows(br))
        results.extend((baseline, result))
        gates, details = development_gates(rows, br)
        decisions.append({"seed": seed, "passed": all(gates.values()), "gates": gates})
        sources.extend({"seed": seed, **r} for r in details)
        contrasts.append({"seed": seed, **{key+"_delta": result[key]-baseline[key] for key in METRICS}})
        for rp in (path, bp):
            for name in ("run_info.json", "checkpoints/best.pth", "validation_best_per_image.csv", "train_history.csv"):
                protect_file(rp / name, inputs)
    aggregates = []
    for arm in ("B_expanded", ARM):
        values = [r for r in results if r["arm"] == arm]
        record = {"arm": arm, "n_seeds": 3}
        for key in METRICS:
            record[key+"_mean"], record[key+"_sample_sd"] = repeat.mean_sd([r[key] for r in values])
        aggregates.append(record)
    paired = {"contrast": "B_background-minus-B_expanded", "n_seeds": 3}
    for key in METRICS:
        paired[key+"_delta_mean"], paired[key+"_delta_sample_sd"] = repeat.mean_sd([r[key+"_delta"] for r in contrasts])
    decision = complete_decision(decisions)
    OUTPUT.mkdir(exist_ok=False)
    for name, values in (("per_seed.csv", results), ("mean_sd.csv", aggregates), ("paired_contrasts.csv", contrasts),
                         ("paired_mean_sd.csv", [paired]), ("source_deltas.csv", sources)):
        write_csv(OUTPUT / name, values)
    write_json(OUTPUT / "decision.json", decision)
    lines = ["# E11 背景误报强化开发结果", "", "状态："+decision["status"], "",
             "只改有界背景像素BCE权重，训练曝光/增强/初始化/验证选择不变。三种子全部完成，旧B保持不变。", "",
             "| 方案 | 宏Dice均值±种子SD | 宏FP/图像面积 | 召回 | 小目标召回 | MAE |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in aggregates:
        lines.append(f"| {r['arm']} | {r['macro_dice_mean']:.6f}±{r['macro_dice_sample_sd']:.6f} | {r['macro_fp_image_fraction_mean']:.6f} | {r['recall_mean']:.6f} | {r['small_recall_mean']:.6f} | {r['mae_mean']:.6f} |")
    lines += ["", "所有门槛与来源失败数量见decision.json和source_deltas.csv。不按最好种子或安全指标重选检查点。",
              "", "通过仍需新独立确认；失败保留B，不在本批开发结果上追加lambda/阈值搜索。", "", decision["interpretation"]]
    (OUTPUT / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(OUTPUT / "audit.json", {"plan_sha256": file_hash(PLAN), "inputs": inputs,
               "outputs": {p.relative_to(PROJECT).as_posix(): file_hash(p) for p in OUTPUT.iterdir() if p.is_file()}})
    print("E11_SUMMARY_COMPLETE", decision["status"], flush=True)


def verify_summary():
    plan, bundle, manifests, protocol = check()
    audit = read_json(OUTPUT / "audit.json")
    if audit["plan_sha256"] != file_hash(PLAN):
        raise ValueError("Summary plan mismatch")
    for kind in ("inputs", "outputs"):
        for name, expected in audit[kind].items():
            if file_hash(PROJECT / name) != expected:
                raise ValueError("Summary input/output changed: " + name)
    saved = read_json(OUTPUT / "decision.json")
    replay = []
    for seed in SEEDS:
        _, _, rows, _ = verified_run(seed, plan, bundle, manifests)
        _, _, base_rows, _ = repeat.verified_run("B_expanded", seed, bundle, manifests, protocol)
        gates, _ = development_gates(rows, base_rows)
        replay.append({"seed": seed, "passed": all(gates.values()), "gates": gates})
    if complete_decision(replay) != saved:
        raise ValueError("Decision changed")
    print("E11_SUMMARY_HASHES_AND_THREE_SEED_DECISION_VERIFIED", flush=True)


def status():
    plan, bundle, manifests, _ = check()
    for seed in SEEDS:
        path = run_path(seed)
        if not path.exists():
            print(f"seed{seed}: NOT_STARTED", flush=True)
        else:
            info = read_json(path / "run_info.json")
            if info["status"] == "TRAINED":
                _, _, _, metrics = verified_run(seed, plan, bundle, manifests)
                print(f"seed{seed}: VERIFIED_COMPLETE macroDice={metrics['macro_dice']:.6f}", flush=True)
            else:
                print(f"seed{seed}: {info['status']}, rounds={info.get('completed_epochs',0)}", flush=True)


def main():
    os.chdir(PROJECT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "smoke", "train", "train-all", "status", "summarize", "verify"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 0 or args.command == "train" and args.seed is None:
        parser.error("Nonnegative workers and --seed for train required")
    if args.command == "train":
        train_one(args.seed, args.workers)
    elif args.command == "smoke":
        train_one(42, args.workers, smoke=True)
    elif args.command == "train-all":
        for seed in SEEDS:
            subprocess.run([sys.executable, str(Path(__file__)), "train", "--seed", str(seed), "--workers", str(args.workers)], cwd=PROJECT, check=True)
    else:
        {"prepare": prepare, "check": check, "status": status, "summarize": summarize, "verify": verify_summary}[args.command]()


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
