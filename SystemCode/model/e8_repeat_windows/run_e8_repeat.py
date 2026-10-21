"""E8-R: user-run seed17/2024 replication of the unchanged E8 protocol.

This wrapper never evaluates C6, performs fusion, retrains seed42, or changes
E8 hyperparameters. Commands that start training must be invoked by the user.
"""
import argparse
import csv
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
sys.path.insert(0, str(PROJECT / "e8_windows"))
import run_e8 as e8
from common import (ARMS, read_json, write_json, write_csv, file_hash, signature,
                    metric_summary, summarize_groups, validation_score)

NEW_SEEDS = (17, 2024)
ALL_SEEDS = (42, *NEW_SEEDS)
PLAN_PATH = PROJECT / "runs/e8_repeat_plan.json"
OUTPUT = PROJECT / "runs/e8_repeat_summary"
SOURCES = ("original_validation", "PolypGen-C5")
METRICS = ("macro_dice", "dice", "iou", "mae", "dice_lt05", "dice_lt01", "zero_dice", "dice_p10")


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite seed values")
    return float(values.mean()), float(values.std(ddof=1))


def aggregate_seed_results(records):
    expected = {(arm, seed) for arm in ARMS for seed in ALL_SEEDS}
    found = {(r["arm"], r["seed"]) for r in records}
    if len(records) != len(expected) or found != expected:
        raise ValueError("Require exactly four arms x three seeds; never summarize incomplete/duplicated runs")
    output = []
    for arm in ARMS:
        row = {"arm": arm, "n_seeds": 3}
        selected = [r for r in records if r["arm"] == arm]
        for metric in METRICS:
            average, sd = mean_sd([r[metric] for r in selected])
            row[metric + "_mean"], row[metric + "_sample_sd"] = average, sd
        output.append(row)
    return output


def paired_seed_contrasts(records):
    aggregate_seed_results(records)  # completeness is mandatory
    indexed = {(r["arm"], r["seed"]): r for r in records}
    rows = []
    for seed in ALL_SEEDS:
        for left, right in (("A_baseline", "B_expanded"), ("A_baseline", "C_hard"),
                            ("B_expanded", "D_expanded_hard"), ("C_hard", "D_expanded_hard")):
            a, b = indexed[left, seed], indexed[right, seed]
            rows.append({"seed": seed, "contrast": right + "-" + left,
                         **{metric + "_delta": b[metric]-a[metric] for metric in METRICS}})
    return rows


def continuation_action(path, config, hashes, bundle_signature):
    """No state deletion or overwrite; interrupted runs need a committed checkpoint."""
    path = Path(path)
    if not path.exists():
        return "train"
    info = read_json(path / "run_info.json")
    if info["config"] != config or info["source_hashes"] != hashes or info["bundle_signature"] != bundle_signature:
        raise ValueError("Existing run provenance differs; do not overwrite: " + str(path))
    if info["run_kind"] != "E8_DEVELOPMENT":
        raise ValueError("Not a formal E8 development run")
    if info["status"] == "TRAINED":
        return "skip_verified"
    if info["status"] == "TRAINING":
        raise ValueError("Run marked TRAINING: it may still be active. Do not start a second process; inspect the terminal first.")
    if info["status"] not in ("FAILED", "INTERRUPTED"):
        raise ValueError("Unexpected existing run status")
    if not (path / "checkpoints/last.pth").is_file():
        raise ValueError("Interrupted before first committed round; preserve directory and request recovery guidance")
    return "resume"


@contextmanager
def serial_training_lock():
    """OS lock releases on process exit; never delete a possibly live lock file."""
    path = PROJECT / "runs/e8_repeat_training.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0,2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK,1)
            except OSError as error:
                raise RuntimeError("Another E8-R training launcher is active; do not start parallel GPU runs") from error
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def verified_run(arm, seed, info, manifests, protocol):
    path, run = e8.completed_run(arm, seed, protocol, info)
    csv_path = path / "validation_best_per_image.csv"
    if file_hash(csv_path) != run["best_validation_csv_sha256"]:
        raise ValueError("Best per-image validation results changed")
    if run["initial_encoder_sha256"] != protocol["initial_encoder_sha256"]:
        raise ValueError("ImageNet initialization differs")
    if run["model_device"] != "cuda:0" or any(run["last_device_proof"][key] != "cuda:0"
                                               for key in ("model", "images", "masks", "logits")):
        raise ValueError("Formal run has no matching CUDA device proof")
    history = read_csv(path / "train_history.csv")
    if len(history) != 100 or [int(r["round"]) for r in history] != list(range(1,101)):
        raise ValueError("Formal run does not have all 100 rounds")
    if any(int(r["train_draws"]) != 1160 or int(r["optimizer_updates"])+int(r["amp_skipped_updates"]) != 145 for r in history):
        raise ValueError("Training exposure/attempted update budget differs")
    if sum(int(r["optimizer_updates"]) for r in history) != run["optimizer_updates_total"]:
        raise ValueError("Actual optimizer update summary mismatch")
    if sum(int(r["amp_skipped_updates"]) for r in history) != run["amp_skipped_updates_total"]:
        raise ValueError("AMP skip summary mismatch")
    best = max(history, key=lambda r: float(r["validation_macro_dice"]))
    if int(best["round"]) != run["best_epoch"]:
        raise ValueError("Best epoch does not match earliest validation maximum")
    rows = read_csv(csv_path)
    expected = {r["sample_id"]: r for r in manifests["validation"]}
    if len(rows) != len(expected) or {r["sample_id"] for r in rows} != set(expected):
        raise ValueError("Validation membership differs")
    for row in rows:
        original = expected[row["sample_id"]]
        if row["dataset"] != original["dataset"]:
            raise ValueError("Validation source label changed")
        for metric in ("dice", "iou", "mae"):
            value = float(row[metric])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid validation metric")
    score = validation_score(rows)
    if not np.isclose(score, run["best_validation_macro_dice"], rtol=0, atol=1e-12):
        raise ValueError("Best validation score mismatch")
    result = {"arm": arm, "seed": seed, "best_round": run["best_epoch"], "macro_dice": score,
              **metric_summary(rows), "round100_macro_dice": float(history[-1]["validation_macro_dice"]),
              "actual_optimizer_updates": run["optimizer_updates_total"], "amp_skipped_updates": run["amp_skipped_updates_total"]}
    return path, run, rows, result


def prepare():
    if PLAN_PATH.exists():
        raise FileExistsError("Repeat plan already frozen; use check")
    info, manifests, protocol = e8.preflight()
    for seed in NEW_SEEDS:
        for arm in ARMS:
            if e8.run_path(arm, seed).exists():
                raise ValueError("A new-seed run exists before repeat-plan freeze; inspect provenance first")
    protected = {}
    for arm in ARMS:
        path, run, _, _ = verified_run(arm, 42, info, manifests, protocol)
        for name in ("run_info.json", "checkpoints/best.pth", "validation_best_per_image.csv", "train_history.csv"):
            file = path / name
            protected[file.relative_to(PROJECT).as_posix()] = file_hash(file)
    for name in ("runs/e8_data_seed42/protocol.json", "runs/e8_validation_comparison_seed42/decision.json",
                 "runs/e8_holdout_lock_seed42.json", "runs/e8_holdout_seed42/run_info.json",
                 "runs/e8_holdout_seed42/summary.csv", "runs/e8_holdout_analysis_plan.json"):
        protected[name] = file_hash(PROJECT / name)
    plan = {"stage": "E8_R_REPLICATION_PREPARATION", "frozen_at_singapore": now(),
            "runner_sha256": file_hash(Path(__file__)), "parent_protocol_signature": protocol["protocol_signature"],
            "bundle_signature": info["bundle_signature"], "protected_seed42_and_holdout_artifacts": protected,
            "existing_seed": 42, "new_seeds": list(NEW_SEEDS), "aggregate_seeds": list(ALL_SEEDS),
            "arms": list(ARMS), "primary_replication_contrast": "B_expanded-A_baseline",
            "secondary_contrasts": ["C_hard-A_baseline", "D_expanded_hard-B_expanded", "D_expanded_hard-C_hard"],
            "hyperparameter_changes": False, "data_split_changes": False,
            "summary_rule": "All 12 runs required; arithmetic mean and sample SD(ddof=1) across seeds; same-seed paired contrasts. Additionally separate two newly run seeds from existing screening seed42.",
            "consistency_rule": "Report B's unchanged E8 development gates for every seed; all-seed reproduction only when all three pass. Also report each source's near-zero and zero failures without retroactively adding gates.",
            "selection_policy": "B remains the fixed recipe; never select best seed or replace seed42 checkpoint from these summaries.",
            "evaluation_policy": "Validation only; no C6 reevaluation, no E0 inference, no fusion or seed ensemble.",
            "limitations": ["Three seeds test training randomness on the same split, not independent patient or centre sampling.",
                            "Validation selected checkpoints; results are development reproducibility, not new external confirmation.",
                            "No p-value, patient CI, or blanket significance claim from three seeds.",
                            "Extra seeds were declared before E8 seed42 training; B-focused replication interpretation follows observed screening and C6."]}
    plan["plan_signature"] = signature(plan)
    write_json(PLAN_PATH, plan)
    print("REPEAT_PLAN_FROZEN", PLAN_PATH, "8_NEW_RUNS_NOT_STARTED", flush=True)


def check(verify_data=True):
    plan = read_json(PLAN_PATH)
    if signature({k:v for k,v in plan.items() if k != "plan_signature"}) != plan["plan_signature"]:
        raise ValueError("Repeat plan signature changed")
    if file_hash(Path(__file__)) != plan["runner_sha256"]:
        raise ValueError("Repeat runner changed after plan freeze")
    for name, expected in plan["protected_seed42_and_holdout_artifacts"].items():
        if file_hash(PROJECT / name) != expected:
            raise ValueError("Protected existing artifact changed: " + name)
    info, manifests, protocol = e8.preflight(verify_content=verify_data)
    if protocol["protocol_signature"] != plan["parent_protocol_signature"] or info["bundle_signature"] != plan["bundle_signature"]:
        raise ValueError("Parent protocol/data changed")
    print("E8_R_CHECK_PASS; E0_AND_SEED42_FROZEN; C6_NOT_USED_FOR_SELECTION", flush=True)
    return plan, info, manifests, protocol


def status():
    _, info, manifests, protocol = check()
    for seed in NEW_SEEDS:
        for arm in ARMS:
            path = e8.run_path(arm, seed)
            if not path.exists():
                print(f"seed{seed} {arm}: NOT_STARTED", flush=True)
            else:
                run = read_json(path / "run_info.json")
                if run["status"] == "TRAINED":
                    _, _, _, result = verified_run(arm, seed, info, manifests, protocol)
                    print(f"seed{seed} {arm}: VERIFIED_COMPLETE; best={result['best_round']}; macro Dice={result['macro_dice']:.6f}", flush=True)
                else:
                    print(f"seed{seed} {arm}: {run['status']}; completed round={run.get('completed_epochs',0)}", flush=True)


def start_one_locked(arm, seed, workers):
    if seed not in NEW_SEEDS:
        raise ValueError("Seed42 retraining is blocked in E8-R")
    _, info, manifests, protocol = check()
    path = e8.run_path(arm, seed)
    action = continuation_action(path, protocol["configs"][f"{arm}/seed{seed}"], protocol["source_hashes"], info["bundle_signature"])
    if action == "skip_verified":
        verified_run(arm, seed, info, manifests, protocol)
        print("SKIP_VERIFIED_COMPLETED_RUN", arm, seed, flush=True)
        return
    print("USER_TRAINING_START", arm, seed, action, "C6_NOT_EVALUATED", flush=True)
    subprocess.run([sys.executable, str(PROJECT / "e8_windows/run_e8.py"), action,
                    "--arm", arm, "--seed", str(seed), "--workers", str(workers)], cwd=PROJECT, check=True)
    # Checkpoint and validation audit after subprocess exit; no model inference.
    verified_run(arm, seed, info, manifests, protocol)


def start_one(arm, seed, workers):
    if seed not in NEW_SEEDS:
        raise ValueError("Seed42 retraining is blocked in E8-R")
    with serial_training_lock():
        start_one_locked(arm,seed,workers)


def compare_seed(seed):
    if seed not in NEW_SEEDS:
        raise ValueError("Seed42 comparison is protected, not regenerated")
    _, info, manifests, protocol = check()
    for arm in ARMS:
        verified_run(arm, seed, info, manifests, protocol)
    folder = PROJECT / "runs" / f"e8_validation_comparison_seed{seed}"
    if folder.exists():
        decision = read_json(folder / "decision.json")
        if decision["seed"] != seed or decision["protocol_signature"] != protocol["protocol_signature"] or decision["bundle_signature"] != info["bundle_signature"]:
            raise ValueError("Existing comparison provenance mismatch")
        print("COMPARISON_EXISTS_PRESERVED", folder, flush=True)
        return
    subprocess.run([sys.executable, str(PROJECT / "e8_windows/run_e8.py"), "compare", "--seed", str(seed)], cwd=PROJECT, check=True)


def summarize():
    if OUTPUT.exists():
        raise FileExistsError("Repeat summary exists; use verify, do not overwrite")
    plan, info, manifests, protocol = check()
    records, groups, per_image, inputs = [], [], {}, {}
    for seed in ALL_SEEDS:
        for arm in ARMS:
            path, run, rows, result = verified_run(arm, seed, info, manifests, protocol)
            records.append(result)
            per_image[arm, seed] = rows
            groups.extend({"seed": seed, "arm": arm, **r} for r in summarize_groups(rows))
            for name in ("run_info.json", "checkpoints/best.pth", "validation_best_per_image.csv", "train_history.csv"):
                file = path / name
                inputs[file.relative_to(PROJECT).as_posix()] = file_hash(file)
    aggregates = aggregate_seed_results(records)
    contrasts = paired_seed_contrasts(records)
    b_gates, tail_details, indexed = [], [], {(r["arm"],r["seed"]):r for r in records}
    for seed in ALL_SEEDS:
        gain, gates = e8.development_gates(indexed["B_expanded",seed], indexed["A_baseline",seed],
                                         per_image["B_expanded",seed], per_image["A_baseline",seed])
        b_gates.append({"seed": seed, "macro_dice_gain_B_minus_A": gain, "unchanged_development_pass": all(gates.values()), "gates": gates})
        for source in SOURCES:
            a = metric_summary([r for r in per_image["A_baseline",seed] if r["dataset"] == source])
            b = metric_summary([r for r in per_image["B_expanded",seed] if r["dataset"] == source])
            tail_details.append({"seed":seed,"source":source,"count":a["count"],
                                 "dice_delta_B_minus_A":b["dice"]-a["dice"],
                                 "low05_delta_B_minus_A":b["dice_lt05"]-a["dice_lt05"],
                                 "near_zero01_delta_B_minus_A":b["dice_lt01"]-a["dice_lt01"],
                                 "zero_delta_B_minus_A":b["zero_dice"]-a["zero_dice"]})
    all_pass = all(r["unchanged_development_pass"] for r in b_gates)
    new_pass = all(r["unchanged_development_pass"] for r in b_gates if r["seed"] in NEW_SEEDS)
    contrast_aggregates = []
    for label in sorted({r["contrast"] for r in contrasts}):
        selected = [r for r in contrasts if r["contrast"] == label]
        record = {"contrast": label, "n_seeds": 3}
        for metric in METRICS:
            record[metric+"_delta_mean"], record[metric+"_delta_sample_sd"] = mean_sd([r[metric+"_delta"] for r in selected])
        new = [r for r in selected if r["seed"] in NEW_SEEDS]
        record["new_seeds_macro_dice_delta_mean"], record["new_seeds_macro_dice_delta_sample_sd"] = mean_sd([r["macro_dice_delta"] for r in new])
        contrast_aggregates.append(record)
    result = {"stage":"E8_R_VALIDATION_REPLICATION", "completed_at_singapore":now(), "seeds":list(ALL_SEEDS),
              "status":"B_GATES_REPRODUCED_ALL_SEEDS" if all_pass else "B_GATES_NOT_REPRODUCED_ALL_SEEDS",
              "B_gates_all_three_pass":all_pass, "B_gates_new_seeds_pass":new_pass, "B_per_seed":b_gates,
              "B_source_tail_details":tail_details,
              "B_any_source_near_zero_or_zero_increase":any(r["near_zero01_delta_B_minus_A"]>0 or r["zero_delta_B_minus_A"]>0 for r in tail_details),
              "paired_contrast_aggregates":contrast_aggregates, "recipe_remains":"B_expanded",
              "best_seed_selection":False, "C6_evaluations":0, "fusion_candidates":0,
              "interpretation":"Development reproducibility only; same validation split and validation-selected checkpoints; no new external superiority claim over E0."}
    OUTPUT.mkdir()
    write_csv(OUTPUT / "per_seed.csv", records)
    write_csv(OUTPUT / "mean_sd.csv", aggregates)
    write_csv(OUTPUT / "groups_per_seed.csv", groups)
    write_csv(OUTPUT / "paired_contrasts.csv", contrasts)
    write_csv(OUTPUT / "paired_contrast_mean_sd.csv", contrast_aggregates)
    write_csv(OUTPUT / "B_source_tail_details.csv", tail_details)
    write_json(OUTPUT / "stability.json", result)
    lines = ["# E8-R：多种子开发重复实验结果", "", f"日期：{now()}。状态：{result['status']}。", "",
             "同一冻结数据划分，seed42/17/2024，四组各100 round。下面±为种子间样本标准差，不是患者置信区间。", "",
             "| 方案 | 宏Dice均值±SD | 图像均Dice均值±SD | MAE均值±SD | 低Dice数量均值±SD | 零Dice数量均值±SD |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in aggregates:
        cells = [f"{r[k+'_mean']:.6f}±{r[k+'_sample_sd']:.6f}" for k in ("macro_dice","dice","mae","dice_lt05","zero_dice")]
        lines.append("| "+r["arm"]+" | "+" | ".join(cells)+" |")
    lines += ["", "## 同种子 B−A 的方向与原门槛", "", "| seed | 宏Dice增量 | 原开发门槛全部通过 |", "|---|---:|---|"]
    for r in b_gates:
        lines.append(f"| {r['seed']} | {r['macro_dice_gain_B_minus_A']:+.6f} | {r['unchanged_development_pass']} |")
    lines += ["", "## B−A 的来源严重失败变化（描述性，不改原门槛）", "", "| seed | 验证来源 | Dice<0.1 数增量 | 零Dice数增量 |", "|---|---|---:|---:|"]
    for r in tail_details:
        lines.append(f"| {r['seed']} | {r['source']} | {r['near_zero01_delta_B_minus_A']:+d} | {r['zero_delta_B_minus_A']:+d} |")
    lines += ["", "## 解释边界", "", "B为先前固定的主方案，不从这些种子中挑最好检查点、不按新结果重新选择C6模型。",
              "这是同数据划分上的训练随机性重复；验证参与checkpoint选择。不能证明新患者/新中心的泛化，不能通过三种子均值宣布显著优于E0。",
              "已完成C6只保留seed42原结果，本轮没有C6推理、融合或seed集成。未来融合或更大范围外部确认需要另立协议。",
              "零Dice/近零Dice未被隐藏：详见groups_per_seed.csv和per_seed.csv。若原门槛不一致，不改变门槛强行宣称重复成功。"]
    (OUTPUT / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    audit = {"plan_sha256":file_hash(PLAN_PATH), "runner_sha256":file_hash(Path(__file__)), "inputs_sha256":inputs,
             "outputs_sha256":{p.name:file_hash(p) for p in OUTPUT.iterdir() if p.is_file()}}
    write_json(OUTPUT / "audit.json", audit)
    print("E8_R_SUMMARY_COMPLETE", result["status"], OUTPUT, flush=True)


def verify():
    audit = read_json(OUTPUT / "audit.json")
    if file_hash(PLAN_PATH) != audit["plan_sha256"] or file_hash(Path(__file__)) != audit["runner_sha256"]:
        raise ValueError("Repeat plan/runner changed")
    for name, expected in audit["inputs_sha256"].items():
        if file_hash(PROJECT / name) != expected:
            raise ValueError("Summary input changed: "+name)
    for name, expected in audit["outputs_sha256"].items():
        if file_hash(OUTPUT / name) != expected:
            raise ValueError("Summary output changed: "+name)
    check()
    print("E8_R_SUMMARY_INPUTS_AND_OUTPUTS_UNCHANGED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare","check","status","train","train-seed","compare","summarize","verify"))
    parser.add_argument("--seed", type=int, choices=NEW_SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    if args.command in ("train","train-seed","compare") and args.seed is None:
        parser.error("--seed 17 or 2024 is required")
    if args.command == "train" and args.arm is None:
        parser.error("--arm is required")
    if args.command == "train":
        start_one(args.arm,args.seed,args.workers)
    elif args.command == "train-seed":
        for arm in ARMS:
            start_one(arm,args.seed,args.workers)
        compare_seed(args.seed)
    elif args.command == "compare":
        compare_seed(args.seed)
    else:
        {"prepare":prepare,"check":check,"status":status,"summarize":summarize,"verify":verify}[args.command]()


if __name__ == "__main__":
    main()
