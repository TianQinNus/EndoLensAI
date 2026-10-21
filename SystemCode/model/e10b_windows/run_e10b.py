"""E10-B: three cached probability fusions, calibration-only choice, locked selection."""
import argparse
import os
import sys
from pathlib import Path
import numpy as np

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
sys.path.insert(0, str(PROJECT / "e10a_windows"))
import run_e10a as a
from common import read_json, write_json, file_hash, signature, read_mask, write_csv, metric_summary, summarize_groups
from e0_protocol import binary_metrics

PLAN = PROJECT / "runs/e10b_fusion_plan.json"
OUTPUT = PROJECT / "runs/e10b_fusion_seed42"
B = "B_expanded"
E0 = "E0_frozen_reference"
ALPHAS = (.25, .5, .75)  # E0 mass, not B mass
GATES = {"min_dice_gain_vs_each_single": .003, "max_mae_increase_vs_each_single": .003,
         "max_low05_increase_vs_each_single": 0, "max_near_zero_increase_vs_each_single": 0,
         "max_zero_increase_vs_each_single": 0}
TOLERANCE = 1e-12


def candidate_name(alpha):
    if alpha not in ALPHAS:
        raise ValueError("Only three predeclared candidate weights are allowed")
    return f"fusion_E0w{int(alpha * 100):03d}"


def source_hashes():
    return {**a.code_hashes(), **{p.relative_to(PROJECT).as_posix(): file_hash(p) for p in sorted(PACKAGE.glob("*.py"))}}


def fuse(p_b, p0, alpha):
    candidate_name(alpha)
    if (p_b.shape != p0.shape or p_b.dtype != np.float32 or p0.dtype != np.float32
            or not np.isfinite(p_b).all() or not np.isfinite(p0).all()
            or min(p_b.min(), p0.min()) < 0 or max(p_b.max(), p0.max()) > 1):
        raise ValueError("Require aligned finite float32 probability inputs in [0,1]")
    return ((1 - alpha) * p_b + alpha * p0).astype(np.float32)


def performance_gates(fusion, baselines):
    if set(baselines) != {B, E0} or any(fusion["count"] != r["count"] for r in baselines.values()):
        raise ValueError("Require both equally sized single-model comparators")
    gates, deltas = {}, []
    for label in (B, E0):
        reference = baselines[label]
        difference = {key: fusion[key] - reference[key] for key in ("dice", "mae", "dice_lt05", "dice_lt01", "zero_dice")}
        gates[label + "_dice"] = difference["dice"] >= GATES["min_dice_gain_vs_each_single"] - TOLERANCE
        gates[label + "_mae"] = difference["mae"] <= GATES["max_mae_increase_vs_each_single"] + TOLERANCE
        gates[label + "_low05"] = difference["dice_lt05"] <= GATES["max_low05_increase_vs_each_single"]
        gates[label + "_near_zero"] = difference["dice_lt01"] <= GATES["max_near_zero_increase_vs_each_single"]
        gates[label + "_zero"] = difference["zero_dice"] <= GATES["max_zero_increase_vs_each_single"]
        deltas.append({"reference": label, **{key + "_delta": value for key, value in difference.items()}})
    return gates, deltas


def choose_candidate(candidates):
    if len(candidates) != 3 or {r["alpha_e0"] for r in candidates} != set(ALPHAS):
        raise ValueError("Require all three distinct predeclared candidates")
    eligible = [r for r in candidates if all(r["gates"].values())]
    if not eligible:
        return None
    highest = max(r["dice"] for r in eligible)
    return min((r for r in eligible if highest - r["dice"] <= TOLERANCE), key=lambda r: r["alpha_e0"])


def freeze():
    if PLAN.exists():
        check()
        return
    if OUTPUT.exists():
        raise ValueError("E10-B outputs predate freeze; preserve and inspect")
    a.verify()
    prior = read_json(a.OUTPUT / "decision.json")
    if prior["status"] != "COMPLETED" or not prior["two_way_threshold_rescue_observed"]:
        raise ValueError("Require verified E10-A two-way descriptive rescue")
    prior_plan, records = a.check()
    audit = read_json(a.OUTPUT / "audit.json")
    protected = {**prior_plan["protected_inputs"], **audit["output_hashes"]}
    for path in (a.PLAN, a.OUTPUT / "audit.json"):
        protected[path.relative_to(PROJECT).as_posix()] = file_hash(path)
    plan = {"stage": "E10_B_FIXED_WEIGHT_CACHED_FUSION", "frozen_at_singapore": a.e9.repeat.now(),
            "source_hashes": source_hashes(), "protected_inputs": protected, "models": prior_plan["models"],
            "membership": a.audit_membership(records), "alpha_e0_candidates": list(ALPHAS),
            "formula": "p_fusion=(1-alpha_e0)*p_B + alpha_e0*p_E0; float32 probability average, not logit average",
            "threshold": .5, "gates": GATES, "numeric_tolerance": TOLERANCE,
            "primary_unit": "image mean; source/GT size/patient mean diagnostic only",
            "choice": "On calibration only: all gates vs both singles; greatest Dice; ties within1e-12 -> smaller E0 weight. If none passes, selection blocked.",
            "selection": "One locked candidate only; same gates vs both selection singles; failure never switches to runner-up.",
            "bootstrap": "20,000 paired patient-cluster replicates seed42, image-weighted Dice delta vs each single; descriptive, not a gate",
            "data_role": "Adaptive reuse of FAR. E10-A already inspected all single-model outcomes and earlier E6/E7 reused these partitions. Not new unseen or blind confirmation.",
            "margin_rationale": "New E10-B gates frozen before fusion: +0.3 pp Dice vs BOTH singles, MAE increase<=0.003 vs BOTH, all three failure counts nonincreasing vs BOTH. No retrospective change to E7/E8/E9.",
            "restrictions": {"training_runs": 0, "model_inferences": 0, "C6_evaluations": 0,
                             "calibration_fit": False, "threshold_search": False, "dynamic_routing": False,
                             "candidate_count": 3, "automatic_external_test": False},
            "failure_policy": "No expanded weight/threshold search, no exclusion of failed cases, no partition reshuffle, retain B as existing primary recipe with its known limits. E0 remains a separate baseline."}
    plan["plan_signature"] = signature(plan)
    write_json(PLAN, plan)
    print("E10_B_PROTOCOL_FROZEN; THREE_CANDIDATES_NOT_COMPUTED", PLAN, flush=True)


def check():
    plan = read_json(PLAN)
    if (signature({k: v for k, v in plan.items() if k != "plan_signature"}) != plan["plan_signature"]
            or plan["source_hashes"] != source_hashes() or plan["gates"] != GATES
            or plan["alpha_e0_candidates"] != list(ALPHAS) or plan["threshold"] != .5):
        raise ValueError("E10-B plan/code/config changed")
    _, records = a.check()
    for name, expected in plan["protected_inputs"].items():
        if file_hash(a.safe_path(PROJECT, name)) != expected:
            raise ValueError("Protected old artifact changed: " + name)
    print("E10_B_CHECK_PASS; CACHED_ONLY; NO_TRAINING_OR_INFERENCE", flush=True)
    return plan, records


def load_pair(record, index):
    # Only the caller's partition is decoded; other-partition descriptors may
    # be hashed for integrity, never used in calibration calculations.
    saved = index[record["sample_id"]]
    path = a.safe_path(a.CACHE, saved["probability_path"])
    if file_hash(path) != saved["probability_sha256"]:
        raise ValueError("E0 probability changed")
    p0 = a.load_probability(path, (record["height"], record["width"]))
    a.replay_row(record, p0, saved)
    _, p_b = a.replay_b_item(record, a.OUTPUT / "B_expanded/items" / (record["stem"] + ".json"))
    gt = read_mask(a.safe_path(PROJECT, record["mask"]))
    return p_b, p0, gt


def compute_rows(records, partition, alphas, output=None, quiet=False):
    if partition not in ("calibration", "selection") or any(alpha not in ALPHAS for alpha in alphas):
        raise ValueError("Unexpected partition/weight")
    if len(set(alphas)) != len(alphas) or (output is not None and (partition != "selection" or len(alphas) != 1)):
        raise ValueError("Duplicate weights or prediction persistence outside one locked selection candidate")
    subset = [r for r in records if r["split"] == partition]
    if not subset:
        raise ValueError("Empty requested partition")
    index_rows = [r for r in a.read_csv(a.CACHE / "per_image.csv") if r["model"] == "e0"]
    index = {r["sample_id"]: r for r in index_rows}
    if len(index) != len(index_rows) or not {r["sample_id"] for r in subset} <= set(index):
        raise ValueError("E0 index duplicate/incomplete")
    rows = []
    for pos, record in enumerate(subset, 1):
        p_b, p0, gt = load_pair(record, index)
        models = [(B, 0., p_b), (E0, 1., p0)]
        models += [(candidate_name(alpha), alpha, fuse(p_b, p0, alpha)) for alpha in alphas]
        for label, alpha, p in models:
            row = {"model": label, "alpha_e0": alpha, "sample_id": record["sample_id"],
                   "dataset": record["source"], "split": partition, "group_id": record["group_id"],
                   "declared_group_id": record["declared_group_id"], "height": gt.shape[0], "width": gt.shape[1],
                   "gt_area_fraction": float(gt.mean()), **binary_metrics(p >= .5, gt),
                   "mae": float(np.abs(p - gt.astype(np.float32)).mean())}
            rows.append(row)
            if output is not None and label not in (B, E0):
                # Only the locked selection candidate is persisted as a prediction.
                relative = "probabilities/" + record["stem"] + ".npz"
                target = a.safe_path(output, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.with_suffix(".npz.tmp").open("wb") as handle:
                    np.savez_compressed(handle, probability=p)
                target.with_suffix(".npz.tmp").replace(target)
        if not quiet and (pos % 40 == 0 or pos == len(subset)):
            print(f"[E10-B {partition} {pos}/{len(subset)}] weights={list(alphas)}; no model inference", flush=True)
    return rows


def model_summaries(rows):
    return [{"model": label, **metric_summary([r for r in rows if r["model"] == label])}
            for label in dict.fromkeys(r["model"] for r in rows)]


def dump_audit(folder):
    hashes = {p.relative_to(PROJECT).as_posix(): file_hash(p) for p in folder.rglob("*")
              if p.is_file() and p.name != "audit.json"}
    write_json(folder / "audit.json", {"plan_sha256": file_hash(PLAN), "outputs": hashes})


def verify_audit(folder):
    audit = read_json(folder / "audit.json")
    if audit["plan_sha256"] != file_hash(PLAN):
        raise ValueError("Phase plan changed")
    for name, expected in audit["outputs"].items():
        if file_hash(a.safe_path(PROJECT, name)) != expected:
            raise ValueError("Phase artifact changed: " + name)


def calibrate():
    with a.e9.repeat.serial_training_lock():
        plan, records = check()
        if OUTPUT.exists():
            verify_calibration()
            print("CALIBRATION_LOCK_EXISTS_PRESERVED; NO_NEW_CANDIDATE_SEARCH")
            return
        OUTPUT.mkdir(exist_ok=False)
        import shutil
        for name in plan["source_hashes"]:
            destination = OUTPUT / "source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, destination)
        folder = OUTPUT / "calibration"
        folder.mkdir()
        write_json(folder / "run_info.json", {"status": "RUNNING", "plan_sha256": file_hash(PLAN),
                   "partition": "calibration", "selection_candidates_computed": 0})
        try:
            rows = compute_rows(records, "calibration", ALPHAS)
            summaries = model_summaries(rows)
            indexed = {r["model"]: r for r in summaries}
            baselines = {label: indexed[label] for label in (B, E0)}
            candidates = []
            for alpha in ALPHAS:
                result = indexed[candidate_name(alpha)]
                gates, deltas = performance_gates(result, baselines)
                candidates.append({"alpha_e0": alpha, "model": result["model"], "dice": result["dice"],
                                   "gates": gates, "passed": all(gates.values()), "deltas": deltas})
            selected = choose_candidate(candidates)
            lock = {"stage": "E10_B_CALIBRATION_LOCK", "locked_at_singapore": a.e9.repeat.now(),
                    "status": "READY_FOR_SELECTION" if selected else "NO_CALIBRATION_PASS_SELECTION_BLOCKED",
                    "selected_alpha_e0": selected["alpha_e0"] if selected else None,
                    "selected_model": selected["model"] if selected else None, "threshold": .5,
                    "plan_sha256": file_hash(PLAN), "gates": GATES, "candidates": candidates,
                    "formal_test_allowed": False, "data_role": plan["data_role"]}
            lock["lock_signature"] = signature(lock)
            write_csv(folder / "per_image.csv", rows)
            write_csv(folder / "summary.csv", summaries)
            write_json(folder / "candidate_gates.json", candidates)
            write_json(OUTPUT / "locked_config.json", lock)
            write_json(folder / "run_info.json", {"status": "COMPLETED", "plan_sha256": file_hash(PLAN),
                       "partition": "calibration", "candidate_weights": list(ALPHAS),
                       "images": len(rows) // 5, "selection_candidates_computed": 0,
                       "locked_config_sha256": file_hash(OUTPUT / "locked_config.json")})
            (folder / "report.md").write_text(report_table("E10-B calibration", summaries) +
                    "\n\n状态：" + lock["status"] + "\n\n冻结E0权重：" + str(lock["selected_alpha_e0"]) +
                    "\n\n只有calibration用于选择；完整条件见candidate_gates.json。无候选通过时，不计算selection融合。\n", encoding="utf-8")
            dump_audit(folder)
            write_json(OUTPUT / "freeze_audit.json", {"plan_sha256": file_hash(PLAN),
                       "locked_config_sha256": file_hash(OUTPUT / "locked_config.json"),
                       "calibration_audit_sha256": file_hash(folder / "audit.json"),
                       "source_snapshot_hashes": {p.relative_to(PROJECT).as_posix(): file_hash(p)
                                                  for p in (OUTPUT / "source_snapshot").rglob("*") if p.is_file()}})
            print("E10_B_CALIBRATION_LOCKED", lock["status"], "E0_WEIGHT", lock["selected_alpha_e0"], flush=True)
        except BaseException as error:
            write_json(folder / "run_info.json", {"status": "FAILED", "error": str(error),
                       "note": "Partial attempt preserved; no deletion/overwrite or unlogged repeat. Request recovery guidance."})
            raise


def read_lock():
    lock = read_json(OUTPUT / "locked_config.json")
    audit = read_json(OUTPUT / "freeze_audit.json")
    if (signature({k: v for k, v in lock.items() if k != "lock_signature"}) != lock["lock_signature"]
            or file_hash(OUTPUT / "locked_config.json") != audit["locked_config_sha256"]
            or lock["plan_sha256"] != file_hash(PLAN) or lock["gates"] != GATES
            or file_hash(OUTPUT / "calibration/audit.json") != audit["calibration_audit_sha256"]):
        raise ValueError("Calibration lock/provenance changed")
    for path, expected in audit["source_snapshot_hashes"].items():
        if file_hash(a.safe_path(PROJECT, path)) != expected:
            raise ValueError("Frozen source snapshot changed")
    return lock


def report_table(title, summaries):
    lines = ["# " + title, "", "| 模型 | Dice | IoU | MAE | Dice<0.5 | Dice<0.1 | 零Dice |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['model']} | {r['dice']:.6f} | {r['iou']:.6f} | {r['mae']:.6f} | {r['dice_lt05']} | {r['dice_lt01']} | {r['zero_dice']} |")
    return "\n".join(lines)


def paired_diagnostics(rows, selected_model):
    indexed = {(r["model"], r["sample_id"]): r for r in rows}
    fused_rows = [r for r in rows if r["model"] == selected_model]
    paired, counts, intervals, patients = [], [], {}, []
    for label in (B, E0):
        pairs = []
        for f in fused_rows:
            baseline = indexed[label, f["sample_id"]]
            pair = {"reference": label, "sample_id": f["sample_id"], "group_id": f["group_id"],
                    "gt_area_fraction": f["gt_area_fraction"], "reference_dice": baseline["dice"], "fusion_dice": f["dice"],
                    "dice_delta": f["dice"] - baseline["dice"], "mae_delta": f["mae"] - baseline["mae"]}
            pairs.append(pair)
        paired.extend(pairs)
        counts.append({"reference": label, "new_low05": sum(r["fusion_dice"] < .5 <= r["reference_dice"] for r in pairs),
                       "rescued_low05": sum(r["reference_dice"] < .5 <= r["fusion_dice"] for r in pairs),
                       "new_near_zero": sum(r["fusion_dice"] < .1 <= r["reference_dice"] for r in pairs),
                       "rescued_near_zero": sum(r["reference_dice"] < .1 <= r["fusion_dice"] for r in pairs),
                       "new_zero": sum(r["fusion_dice"] == 0 < r["reference_dice"] for r in pairs),
                       "rescued_zero": sum(r["reference_dice"] == 0 < r["fusion_dice"] for r in pairs)})
        intervals[label] = a.bootstrap([{"group_id": r["group_id"], "dice_delta_B_minus_E0": r["dice_delta"]} for r in pairs])
        for group in sorted({r["group_id"] for r in pairs}):
            subset = [r for r in pairs if r["group_id"] == group]
            patients.append({"reference": label, "group_id": group, "count": len(subset),
                             "dice_delta_patient_mean": float(np.mean([r["dice_delta"] for r in subset])),
                             "mae_delta_patient_mean": float(np.mean([r["mae_delta"] for r in subset]))})
    return paired, counts, intervals, patients


def evaluate():
    with a.e9.repeat.serial_training_lock():
        plan, records = check()
        verify_calibration(records)
        lock = read_lock()
        if lock["status"] == "NO_CALIBRATION_PASS_SELECTION_BLOCKED":
            print("SELECTION_BLOCKED; NO_CALIBRATION_CANDIDATE_PASS; RETAIN_B", flush=True)
            return
        if lock["status"] != "READY_FOR_SELECTION" or lock["selected_alpha_e0"] not in ALPHAS:
            raise ValueError("Unexpected lock state")
        folder = OUTPUT / "selection"
        if folder.exists():
            # Existing completed output can only be verified, never recomputed
            # into a new evaluation or replaced by a different candidate.
            if read_json(folder / "run_info.json")["status"] == "COMPLETED":
                verify_selection(records)
                print("COMPLETED_SELECTION_PRESERVED; NO_NEW_EVALUATION")
                return
            raise ValueError("Selection attempt already exists; preserve incomplete outputs and request recovery guidance")
        folder.mkdir(exist_ok=False)
        lock_hash = file_hash(OUTPUT / "locked_config.json")
        write_json(folder / "run_info.json", {"status": "RUNNING", "plan_sha256": file_hash(PLAN),
                   "locked_config_sha256": lock_hash, "selected_alpha_e0": lock["selected_alpha_e0"]})
        try:
            rows = compute_rows(records, "selection", (lock["selected_alpha_e0"],), folder)
            summaries = model_summaries(rows)
            indexed = {r["model"]: r for r in summaries}
            gates, deltas = performance_gates(indexed[lock["selected_model"]], {label: indexed[label] for label in (B, E0)})
            passed = all(gates.values())
            paired, counts, intervals, patients = paired_diagnostics(rows, lock["selected_model"])
            decision = {"stage": "E10_B_LOCKED_SELECTION_DEVELOPMENT", "status": "DEVELOPMENT_PASS" if passed else "NO_PASS_RETAIN_B",
                        "selected_alpha_e0": lock["selected_alpha_e0"], "gates": gates, "deltas": deltas,
                        "locked_config_sha256": lock_hash, "formal_test_allowed": False,
                        "next_step": "Independent-data confirmation under a new protocol; no automatic test" if passed else "Retain original B; stop this weight search; do not switch runner-up",
                        "data_role": plan["data_role"], "training_runs": 0, "model_inferences": 0, "C6_evaluations": 0}
            write_csv(folder / "per_image.csv", rows)
            write_csv(folder / "summary.csv", summaries)
            write_csv(folder / "groups.csv", [{"model": label, **r} for label in indexed
                      for r in summarize_groups([x for x in rows if x["model"] == label])])
            write_csv(folder / "paired_deltas.csv", paired)
            write_csv(folder / "by_patient.csv", patients)
            write_json(folder / "failure_transitions.json", counts)
            write_json(folder / "patient_bootstrap.json", intervals)
            write_json(folder / "decision.json", decision)
            (folder / "report.md").write_text(report_table("E10-B 冻结selection开发结果", summaries) +
                    "\n\n状态：" + decision["status"] + "\n\n冻结E0权重：" + str(lock["selected_alpha_e0"]) +
                    "\n\n逐项门槛见decision.json；双向失败转移、患者描述性bootstrap及面积分层见同目录。\n\n" +
                    plan["data_role"] + "\n\n" + decision["next_step"] + "\n", encoding="utf-8")
            if file_hash(OUTPUT / "locked_config.json") != lock_hash:
                raise ValueError("Lock changed during selection")
            write_json(folder / "run_info.json", {"status": "COMPLETED", "plan_sha256": file_hash(PLAN),
                       "locked_config_sha256": lock_hash, "selected_alpha_e0": lock["selected_alpha_e0"],
                       "images": len(rows) // 3, "fusion_candidates_evaluated": 1,
                       "training_runs": 0, "model_inferences": 0, "C6_evaluations": 0})
            dump_audit(folder)
            print("E10_B_SELECTION_COMPLETE", decision["status"], deltas, flush=True)
        except BaseException as error:
            write_json(folder / "run_info.json", {"status": "FAILED", "error": str(error),
                       "locked_config_sha256": lock_hash, "note": "Preserve attempt; do not change alpha or remove the marker."})
            raise


def compare_saved(saved, replay):
    if len(saved) != len(replay):
        raise ValueError("Saved/replayed row count differs")
    indexed = {(r["model"], r["sample_id"]): r for r in saved}
    if len(indexed) != len(saved) or set(indexed) != {(r["model"], r["sample_id"]) for r in replay}:
        raise ValueError("Missing/duplicated/different result membership")
    for r in replay:
        old = indexed[r["model"], r["sample_id"]]
        for key in ("alpha_e0", "dice", "iou", "mae", "tp", "fp", "fn", "gt_area_fraction"):
            if not np.isclose(float(old[key]), r[key], rtol=0, atol=TOLERANCE):
                raise ValueError("Saved metric differs: " + key)
        for key in ("dataset", "split", "group_id", "declared_group_id"):
            if old[key] != r[key]:
                raise ValueError("Saved metadata differs")


def verify_calibration(records=None):
    if records is None:
        _, records = check()
    folder = OUTPUT / "calibration"
    verify_audit(folder)
    lock = read_lock()
    if read_json(folder / "run_info.json")["status"] != "COMPLETED":
        raise ValueError("Incomplete calibration")
    rows = compute_rows(records, "calibration", ALPHAS, quiet=True)
    compare_saved(a.read_csv(folder / "per_image.csv"), rows)
    indexed = {r["model"]: r for r in model_summaries(rows)}
    candidates = []
    for alpha in ALPHAS:
        result = indexed[candidate_name(alpha)]
        gates, deltas = performance_gates(result, {label: indexed[label] for label in (B, E0)})
        candidates.append({"alpha_e0": alpha, "model": result["model"], "dice": result["dice"],
                           "gates": gates, "passed": all(gates.values()), "deltas": deltas})
    if candidates != lock["candidates"]:
        raise ValueError("Calibration candidate choice changed")
    chosen = choose_candidate(candidates)
    expected_status = "READY_FOR_SELECTION" if chosen else "NO_CALIBRATION_PASS_SELECTION_BLOCKED"
    if lock["status"] != expected_status or lock["selected_alpha_e0"] != (chosen["alpha_e0"] if chosen else None):
        raise ValueError("Lock does not follow predeclared calibration-only choice")
    print("E10_B_CALIBRATION_REPLAY_VERIFIED; LOCK_UNCHANGED", flush=True)


def verify_selection(records=None):
    if records is None:
        _, records = check()
    folder = OUTPUT / "selection"
    verify_audit(folder)
    lock = read_lock()
    if lock["status"] != "READY_FOR_SELECTION":
        raise ValueError("Selection should have been blocked")
    rows = compute_rows(records, "selection", (lock["selected_alpha_e0"],), quiet=True)
    compare_saved(a.read_csv(folder / "per_image.csv"), rows)
    indexed = {r["model"]: r for r in model_summaries(rows)}
    gates, deltas = performance_gates(indexed[lock["selected_model"]], {label: indexed[label] for label in (B, E0)})
    decision = read_json(folder / "decision.json")
    if (decision["gates"] != gates or decision["deltas"] != deltas
            or decision["status"] != ("DEVELOPMENT_PASS" if all(gates.values()) else "NO_PASS_RETAIN_B")
            or decision["locked_config_sha256"] != file_hash(OUTPUT / "locked_config.json")):
        raise ValueError("Selection decision differs")
    index = {r["sample_id"]: r for r in a.read_csv(a.CACHE / "per_image.csv") if r["model"] == "e0"}
    for record in [r for r in records if r["split"] == "selection"]:
        p_b, p0, _ = load_pair(record, index)
        target = a.safe_path(folder, "probabilities/" + record["stem"] + ".npz")
        cached = a.load_probability(target, p_b.shape)
        if not np.array_equal(cached, fuse(p_b, p0, lock["selected_alpha_e0"])):
            raise ValueError("Saved fused probability differs")
    print("E10_B_SELECTION_REPLAY_VERIFIED; NO_NEW_CANDIDATE", flush=True)


def verify():
    _, records = check()
    verify_calibration(records)
    lock = read_lock()
    if lock["status"] == "NO_CALIBRATION_PASS_SELECTION_BLOCKED":
        if (OUTPUT / "selection").exists():
            raise ValueError("Selection output exists despite calibration block")
        print("E10_B_VERIFIED_CALIBRATION_NO_PASS; SELECTION_NOT_EVALUATED")
    else:
        verify_selection(records)


if __name__ == "__main__":
    os.chdir(PROJECT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check", "calibrate", "evaluate", "verify"))
    args = parser.parse_args()
    {"freeze": freeze, "check": check, "calibrate": calibrate, "evaluate": evaluate, "verify": verify}[args.command]()
