"""E9-Balance: one 70/30 sampling revision; user-run training, validation only."""
import argparse
import csv
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
sys.path.insert(0, str(PROJECT / "e8_repeat_windows"))
import run_e8_repeat as repeat
import train_e8 as trainer
from common import code_hashes as parent_hashes, read_json, write_json, write_csv, file_hash, signature
from common import metric_summary, validation_score, summarize_groups
from engine import require_cuda, environment

SEEDS = (42, 17, 2024)
PLAN = PROJECT / "runs/e9_balance_plan.json"
OUTPUT = PROJECT / "runs/e9_balance_summary"
SOURCES = repeat.SOURCES
METRICS = repeat.METRICS
# These are new E9 development criteria, not a retroactive E8 gate revision.
GATES = {"max_macro_dice_loss_vs_B": .003, "max_C5_dice_loss_vs_B": .01,
         "max_original_dice_loss_vs_A": .003, "max_source_near_zero_increase_vs_B": 0,
         "max_source_zero_increase_vs_B": 0, "require_all_unchanged_E8_gates_vs_A": True,
         "require_all_three_seeds": True}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def stage_hashes():
    return {**parent_hashes(), **{p.relative_to(PROJECT).as_posix(): file_hash(p)
                                for p in sorted(PACKAGE.glob("*.py"))}}


def make_config(seed, smoke=False):
    if seed not in SEEDS:
        raise ValueError("Only the three predeclared seeds are allowed")
    config = trainer.make_config("B_expanded", seed, smoke)
    config.update(stage="E9_BALANCE", original_sampling_mass=.7,
                  sampling="replacement; original mass 0.7; remaining 0.3 equally across C1-C4; no hard mining")
    return config


def sampling_probabilities(records, losses, config, epoch):
    if not config["expanded"] or config["hard"]:
        raise ValueError("E9 only supports expanded data without hard sampling")
    mass = config["original_sampling_mass"]
    if not 0 < mass < 1:
        raise ValueError("Sampling mass must be strictly inside (0,1)")
    strata = {}
    for i, row in enumerate(records):
        strata.setdefault(row["dataset"] if row["dataset"].startswith("PolypGen-") else "original", []).append(i)
    if "original" not in strata or set(strata) - {"original"} != {f"PolypGen-C{i}" for i in range(1, 5)}:
        raise ValueError("Expected original training plus exactly C1-C4; validation/holdout centres forbidden")
    probabilities = np.zeros(len(records), np.float64)
    for source, indices in strata.items():
        probabilities[indices] = (mass if source == "original" else (1 - mass) / 4) / len(indices)
    if not np.isclose(probabilities.sum(), 1) or np.any(probabilities <= 0):
        raise ValueError("Invalid source-balanced probabilities")
    return probabilities, 0


@contextmanager
def training_adapter():
    """Reuse the unedited E8 optimizer/model/loop, replacing only source sampler.

    Extend provenance snapshots and label run metadata for E9. This is process-local;
    no E8 source or protocol file changes. Hashes cover both base and extension.
    """
    old_sampler, old_hashes, old_writer = trainer.sampling_probabilities, trainer.code_hashes, trainer.write_json
    def stage_writer(path, value):
        if Path(path).name == "run_info.json":
            value.update(run_kind="E9_GPU_SMOKE" if value["config"]["smoke"] else "E9_DEVELOPMENT",
                         stage="E9_BALANCE", note="E8 training backend reused; only sampling mass changes. No C6/E0/fusion inference.")
        old_writer(path, value)
    trainer.sampling_probabilities, trainer.code_hashes, trainer.write_json = sampling_probabilities, stage_hashes, stage_writer
    try:
        yield
    finally:
        trainer.sampling_probabilities, trainer.code_hashes, trainer.write_json = old_sampler, old_hashes, old_writer


def run_path(seed, smoke=False):
    return PROJECT / "runs" / f"e9_{'smoke_' if smoke else ''}B_balance70_seed{seed}"


def check():
    plan = read_json(PLAN)
    if signature({k: v for k, v in plan.items() if k != "plan_signature"}) != plan["plan_signature"]:
        raise ValueError("E9 plan signature changed")
    if plan["source_hashes"] != stage_hashes() or plan["gates"] != GATES:
        raise ValueError("E9 code or criteria changed after freeze")
    for name, expected in plan["protected_artifacts"].items():
        if file_hash(PROJECT / name) != expected:
            raise ValueError("Protected artifact changed: " + name)
    bundle, manifests, protocol = repeat.e8.preflight()
    if protocol["protocol_signature"] != plan["parent_protocol_signature"] or bundle["bundle_signature"] != plan["bundle_signature"]:
        raise ValueError("Parent data/protocol changed")
    if plan["configs"] != {str(s): make_config(s) for s in SEEDS}:
        raise ValueError("E9 configs changed")
    print("E9_BALANCE_CHECK_PASS; E8/E0/C6_FROZEN; ONLY_70_30_REVISION", flush=True)
    return plan, bundle, manifests, protocol


def prepare():
    if PLAN.exists():
        check()
        return
    from diagnose_e8 import verify as verify_diagnosis
    verify_diagnosis()
    parent_plan, bundle, manifests, protocol = repeat.check()
    if any(run_path(seed).exists() or run_path(seed, True).exists() for seed in SEEDS):
        raise ValueError("An E9 run predates the plan; do not overwrite")
    audit = read_json(repeat.OUTPUT / "audit.json")
    protected = {**audit["inputs_sha256"], **parent_plan["protected_seed42_and_holdout_artifacts"]}
    for name in audit["outputs_sha256"]:
        path = repeat.OUTPUT / name
        protected[path.relative_to(PROJECT).as_posix()] = file_hash(path)
    for folder in (repeat.OUTPUT, PROJECT / "runs/e9_error_diagnosis"):
        for path in folder.rglob("*"):
            if path.is_file():
                protected[path.relative_to(PROJECT).as_posix()] = file_hash(path)
    protected[repeat.PLAN_PATH.relative_to(PROJECT).as_posix()] = file_hash(repeat.PLAN_PATH)
    for seed in SEEDS:
        for arm in ("A_baseline", "B_expanded"):
            repeat.verified_run(arm, seed, bundle, manifests, protocol)
    plan = {"stage": "E9_BALANCE_DEVELOPMENT", "frozen_at_singapore": repeat.now(),
            "parent_protocol_signature": protocol["protocol_signature"], "bundle_signature": bundle["bundle_signature"],
            "source_hashes": stage_hashes(), "protected_artifacts": protected,
            "seeds": list(SEEDS), "configs": {str(s): make_config(s) for s in SEEDS}, "gates": GATES,
            "only_training_change": "original/new source sampling mass 0.5/0.5 -> 0.7/0.3; new centres remain equally weighted",
            "primary_comparator": "same-seed frozen E8 B", "safety_comparator": "same-seed frozen E8 A",
            "checkpoint_selection": "Unchanged old/C5 equal-weight Dice; earliest maximum; no safety-selected checkpoint",
            "adoption": "All three seeds must pass all E9 criteria; otherwise retain B and report failure. No best-seed selection.",
            "fixed_deployment_seed": 42, "training_runs_planned": 3,
            "threshold_rationale": "Development tolerances: macro loss <=0.3 pp vs B, C5 loss <=1 pp vs B, original loss <=0.3 pp vs A; unchanged E8 failure/MAE gates still required. These are practical margins, not statistical noninferiority evidence.",
            "restrictions": "No C6 reevaluation, E0 inference, fusion, added data, hard mining, label changes or validation-to-training transfer.",
            "interpretation": "Adaptive follow-up after E8-R diagnosis; repeatedly used fixed validation, not new external confirmation."}
    plan["plan_signature"] = signature(plan)
    write_json(PLAN, plan)
    print("E9_BALANCE_PLAN_FROZEN; THREE_FORMAL_RUNS_NOT_STARTED", PLAN, flush=True)


def validate_rows(rows, validation):
    expected = {r["sample_id"]: r for r in validation}
    if len(rows) != len(expected) or {r["sample_id"] for r in rows} != set(expected):
        raise ValueError("Validation membership incomplete or duplicated")
    for row in rows:
        if row["dataset"] != expected[row["sample_id"]]["dataset"]:
            raise ValueError("Validation source changed")
        for key in ("dice", "iou", "mae"):
            if not np.isfinite(float(row[key])) or not 0 <= float(row[key]) <= 1:
                raise ValueError("Invalid validation metric")


def verified_run(seed, plan, bundle, manifests, smoke=False):
    path = run_path(seed, smoke)
    info = read_json(path / "run_info.json")
    config = make_config(seed, smoke)
    if info["status"] != "TRAINED" or info["run_kind"] != ("E9_GPU_SMOKE" if smoke else "E9_DEVELOPMENT"):
        raise ValueError("Require completed E9 run")
    if info["config"] != config or info["source_hashes"] != plan["source_hashes"] or info["bundle_signature"] != bundle["bundle_signature"]:
        raise ValueError("E9 provenance differs")
    for name, key in (("checkpoints/best.pth", "best_checkpoint_sha256"), ("validation_best_per_image.csv", "best_validation_csv_sha256")):
        if file_hash(path / name) != info[key]:
            raise ValueError("E9 result changed: " + name)
    cp = torch.load(path / "checkpoints/best.pth", map_location="cpu", weights_only=True)
    if any(cp[key] != info[key] for key in ("config", "source_hashes", "bundle_signature")) or cp["epoch"] != info["best_epoch"]:
        raise ValueError("E9 checkpoint metadata differs")
    del cp
    baseline = read_json(repeat.e8.run_path("B_expanded", seed) / "run_info.json")
    if info["environment"]["versions"] != baseline["environment"]["versions"] or info["environment"].get("gpu") != baseline["environment"].get("gpu"):
        raise ValueError("Training environment differs from paired B")
    if info["initial_encoder_sha256"] != baseline["initial_encoder_sha256"] or info["model_device"] != "cuda:0" or not info["amp_enabled"]:
        raise ValueError("Initial encoder/CUDA/AMP proof differs")
    if any(info["last_device_proof"][key] != "cuda:0" for key in ("model", "images", "masks", "logits")):
        raise ValueError("CUDA tensor proof missing")
    history = read_csv(path / "train_history.csv")
    training_records = manifests["original_train"] + manifests["new_train"]
    if (read_json(path / "train_manifest.json") != training_records or info["training_unique_images"] != len(training_records)
            or info["completed_epochs"] != config["epochs"] or info["formal_holdout_evaluations"] != 0):
        raise ValueError("Training membership/completion/holdout differs")
    probabilities, _ = sampling_probabilities(training_records, [], config, 1)
    expected_updates = config["samples_per_round"] // (config["batch_size"] * config["accumulation"])
    if len(history) != config["epochs"] or [int(r["round"]) for r in history] != list(range(1, config["epochs"] + 1)):
        raise ValueError("Incomplete training history")
    for row in history:
        if (int(row["train_draws"]) != config["samples_per_round"] or int(row["hard_boosted_images"]) != 0
                or int(row["optimizer_updates"]) + int(row["amp_skipped_updates"]) != expected_updates):
            raise ValueError("Training budget or hard-sampling differs")
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(row["round"]), 808]))
        draws = rng.choice(len(training_records), config["samples_per_round"], replace=True, p=probabilities)
        if trainer.hashlib_sha256_indices(draws) != row["sample_indices_sha256"]:
            raise ValueError("Actual draw sequence does not match frozen 70/30 sampler")
    for key, total in (("optimizer_updates", "optimizer_updates_total"), ("amp_skipped_updates", "amp_skipped_updates_total")):
        if sum(int(r[key]) for r in history) != info[total]:
            raise ValueError("Update summary differs")
    best = max(history, key=lambda r: float(r["validation_macro_dice"]))
    if int(best["round"]) != info["best_epoch"]:
        raise ValueError("Not earliest macro-Dice maximum")
    rows = read_csv(path / "validation_best_per_image.csv")
    validation = manifests["validation"]
    if smoke:
        validation = [r for source in SOURCES for r in [x for x in validation if x["dataset"] == source][:2]]
    validate_rows(rows, validation)
    if not np.isclose(validation_score(rows), info["best_validation_macro_dice"], atol=1e-12, rtol=0):
        raise ValueError("Validation score mismatch")
    result = {"arm": "B_balance70", "seed": seed, "best_round": info["best_epoch"],
              "macro_dice": validation_score(rows), **metric_summary(rows),
              "actual_optimizer_updates": info["optimizer_updates_total"], "amp_skipped_updates": info["amp_skipped_updates_total"]}
    return path, info, rows, result


def continuation_action(path, config, hashes, bundle_signature):
    if not path.exists():
        return "train"
    info = read_json(path / "run_info.json")
    if info["config"] != config or info["source_hashes"] != hashes or info["bundle_signature"] != bundle_signature:
        raise ValueError("Existing run differs; do not overwrite")
    if info["status"] == "TRAINED":
        return "skip"
    if info["status"] == "TRAINING":
        raise ValueError("May still be active; inspect terminal before recovery. Never start a second process.")
    if info["status"] not in ("FAILED", "INTERRUPTED") or not (path / "checkpoints/last.pth").is_file():
        raise ValueError("No safe committed-round recovery; preserve files and request guidance")
    return "resume"


def train_one(seed, workers=2, smoke=False):
    if smoke and seed != 42:
        raise ValueError("Smoke only uses seed42")
    with repeat.serial_training_lock():
        plan, bundle, manifests, _ = check()
        device = require_cuda()
        baseline = read_json(repeat.e8.run_path("B_expanded", seed) / "run_info.json")
        env = environment(device)
        if env["versions"] != baseline["environment"]["versions"] or env.get("gpu") != baseline["environment"].get("gpu"):
            raise ValueError("Environment mismatch before training")
        path, config = run_path(seed, smoke), make_config(seed, smoke)
        action = continuation_action(path, config, plan["source_hashes"], bundle["bundle_signature"])
        if action == "skip":
            verified_run(seed, plan, bundle, manifests, smoke)
            print("SKIP_VERIFIED_COMPLETE", path, flush=True)
            return
        if not smoke:
            verified_run(42, plan, bundle, manifests, smoke=True)
        print("USER_E9_TRAINING_START", seed, action, "SMOKE" if smoke else "100_ROUNDS", flush=True)
        with training_adapter():
            trainer.train(path, config, bundle, manifests, workers=workers, resume=action == "resume")
        verified_run(seed, plan, bundle, manifests, smoke)


def development_gates(candidate_rows, a_rows, b_rows):
    candidate = {"macro_dice": validation_score(candidate_rows)}
    a = {"macro_dice": validation_score(a_rows)}
    _, gates = repeat.e8.development_gates(candidate, a, candidate_rows, a_rows)
    gates = {"unchanged_E8_" + k: v for k, v in gates.items()}
    gates["macro_retained_vs_B"] = candidate["macro_dice"] >= validation_score(b_rows) - GATES["max_macro_dice_loss_vs_B"]
    details = []
    for source in SOURCES:
        c, old_a, old_b = [metric_summary([r for r in rows if r["dataset"] == source])
                          for rows in (candidate_rows, a_rows, b_rows)]
        if source == "original_validation":
            gates["original_dice_protected_vs_A"] = c["dice"] >= old_a["dice"] - GATES["max_original_dice_loss_vs_A"]
        else:
            gates["C5_dice_retained_vs_B"] = c["dice"] >= old_b["dice"] - GATES["max_C5_dice_loss_vs_B"]
        for metric, label in (("dice_lt01", "near_zero"), ("zero_dice", "zero")):
            gates[source + "_" + label + "_vs_B"] = c[metric] - old_b[metric] <= GATES["max_source_" + label + "_increase_vs_B"]
        details.append({"source": source, "dice_delta_vs_A": c["dice"] - old_a["dice"],
                        "dice_delta_vs_B": c["dice"] - old_b["dice"],
                        "low05_delta_vs_A": c["dice_lt05"] - old_a["dice_lt05"],
                        "low05_delta_vs_B": c["dice_lt05"] - old_b["dice_lt05"],
                        "near_zero_delta_vs_B": c["dice_lt01"] - old_b["dice_lt01"],
                        "zero_delta_vs_B": c["zero_dice"] - old_b["zero_dice"]})
    return gates, details


def verify_summary():
    check()
    audit = read_json(OUTPUT / "audit.json")
    if file_hash(PLAN) != audit["plan_sha256"]:
        raise ValueError("Summary plan changed")
    for key in ("inputs", "outputs"):
        for name, expected in audit[key].items():
            if file_hash(PROJECT / name) != expected:
                raise ValueError("Summary artifact changed: " + name)
    print("E9_BALANCE_SUMMARY_VERIFIED")


def summarize():
    if OUTPUT.exists():
        verify_summary()
        return
    plan, bundle, manifests, protocol = check()
    records, groups, decisions, source_details, contrasts, inputs = [], [], [], [], [], {}
    for seed in SEEDS:
        path, _, rows, result = verified_run(seed, plan, bundle, manifests)
        reference = {}
        for arm in ("A_baseline", "B_expanded"):
            ref_path, _, ref_rows, ref_result = repeat.verified_run(arm, seed, bundle, manifests, protocol)
            reference[arm] = ref_rows, ref_result
            records.append(ref_result)
            groups.extend({"arm": arm, "seed": seed, **r} for r in summarize_groups(ref_rows))
        records.append(result)
        groups.extend({"arm": result["arm"], "seed": seed, **r} for r in summarize_groups(rows))
        gates, details = development_gates(rows, reference["A_baseline"][0], reference["B_expanded"][0])
        decisions.append({"seed": seed, "passed": all(gates.values()), "gates": gates})
        source_details.extend({"seed": seed, **row} for row in details)
        for arm, (_, ref) in reference.items():
            contrasts.append({"seed": seed, "contrast": result["arm"] + "-" + arm,
                              **{key + "_delta": result[key] - ref[key] for key in METRICS}})
        for name in ("run_info.json", "checkpoints/best.pth", "validation_best_per_image.csv", "train_history.csv"):
            file = path / name
            inputs[file.relative_to(PROJECT).as_posix()] = file_hash(file)
    aggregates = []
    for arm in ("A_baseline", "B_expanded", "B_balance70"):
        values = [r for r in records if r["arm"] == arm]
        if len(values) != 3 or {r["seed"] for r in values} != set(SEEDS):
            raise ValueError("Require exactly three complete paired seeds")
        row = {"arm": arm, "n_seeds": 3}
        for key in METRICS:
            row[key + "_mean"], row[key + "_sample_sd"] = repeat.mean_sd([r[key] for r in values])
        aggregates.append(row)
    paired_aggregates = []
    for contrast in sorted({r["contrast"] for r in contrasts}):
        values = [r for r in contrasts if r["contrast"] == contrast]
        row = {"contrast": contrast, "n_seeds": 3}
        for key in METRICS:
            row[key + "_delta_mean"], row[key + "_delta_sample_sd"] = repeat.mean_sd([r[key + "_delta"] for r in values])
        paired_aggregates.append(row)
    passed = all(r["passed"] for r in decisions)
    decision = {"stage": "E9_BALANCE_DEVELOPMENT", "status": "DEVELOPMENT_PASS" if passed else "NO_PASS_RETAIN_B",
                "selected_recipe_for_next_protocol": "B_balance70" if passed else "B_expanded",
                "deployment_seed": 42, "per_seed": decisions, "gates": GATES,
                "source_details": source_details, "C6_evaluations": 0, "fusion_candidates": 0,
                "limitation": "Same repeatedly used validation; selection is developmental, not new external superiority/noninferiority evidence."}
    OUTPUT.mkdir(exist_ok=False)
    for name, values in (("per_seed.csv", records), ("mean_sd.csv", aggregates), ("groups_per_seed.csv", groups),
                         ("source_deltas.csv", source_details), ("paired_contrasts.csv", contrasts),
                         ("paired_contrast_mean_sd.csv", paired_aggregates)):
        write_csv(OUTPUT / name, values)
    write_json(OUTPUT / "decision.json", decision)
    lines = ["# E9-Balance 开发实验结果", "", "状态：" + decision["status"], "",
             "仅原/新增采样质量从50/50调整为70/30；不改数据划分、网络、损失、预算、检查点选择。", "",
             "| 方案 | 宏Dice均值±种子SD | 图像平均Dice | MAE | 低Dice平均张数 | 零Dice平均张数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        lines.append(f"| {row['arm']} | {row['macro_dice_mean']:.6f}±{row['macro_dice_sample_sd']:.6f} | {row['dice_mean']:.6f} | {row['mae_mean']:.6f} | {row['dice_lt05_mean']:.2f} | {row['zero_dice_mean']:.2f} |")
    lines += ["", "三个种子必须全部通过；不选最好种子。完整逐项门槛见decision.json，配对差值见paired_contrast_mean_sd.csv。",
              "", "后续协议候选：" + decision["selected_recipe_for_next_protocol"], "",
              "仅开发选择，不自动启动融合或外部测试；E8-R未通过状态保持原样。", "",
              decision["limitation"]]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {p.relative_to(PROJECT).as_posix(): file_hash(p) for p in OUTPUT.iterdir() if p.is_file()}
    write_json(OUTPUT / "audit.json", {"plan_sha256": file_hash(PLAN), "inputs": inputs, "outputs": outputs})
    print("E9_BALANCE_SUMMARY_COMPLETE", decision["status"], OUTPUT, flush=True)


def status():
    plan, bundle, manifests, _ = check()
    for seed in SEEDS:
        path = run_path(seed)
        if not path.exists():
            print(f"seed{seed}: NOT_STARTED")
            continue
        info = read_json(path / "run_info.json")
        if info["status"] == "TRAINED":
            _, _, _, result = verified_run(seed, plan, bundle, manifests)
            print(f"seed{seed}: VERIFIED_COMPLETE macro Dice={result['macro_dice']:.6f}")
        else:
            print(f"seed{seed}: {info['status']} completed rounds={info.get('completed_epochs', 0)}")


def main():
    os.chdir(PROJECT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "smoke", "train", "train-all", "status", "summarize", "verify"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 0 or args.command == "train" and args.seed is None:
        parser.error("Nonnegative workers and --seed for train are required")
    if args.command == "train":
        train_one(args.seed, args.workers)
    elif args.command == "smoke":
        train_one(42, args.workers, smoke=True)
    elif args.command == "train-all":
        # Fresh process per seed releases all GPU allocations. Completed runs are audited/skipped.
        for seed in SEEDS:
            subprocess.run([sys.executable, str(Path(__file__)), "train", "--seed", str(seed),
                            "--workers", str(args.workers)], cwd=PROJECT, check=True)
    else:
        {"prepare": prepare, "check": check, "status": status, "summarize": summarize, "verify": verify_summary}[args.command]()


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
