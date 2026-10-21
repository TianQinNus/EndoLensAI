"""E10-C: one bounded two-temperature equal-fusion policy, patient-grouped OOF gate."""
import argparse
import os
import sys
from pathlib import Path
import numpy as np

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
sys.path.insert(0, str(PROJECT / "e10b_windows"))
import run_e10b as b
from common import read_json, write_json, file_hash, signature, write_csv, metric_summary, summarize_groups
from e0_protocol import binary_metrics
from temperature import apply_temperature, assign_folds, sample_pair, fit_temperature, logits, BOUNDS, EPSILON, PIXELS_PER_IMAGE

PLAN = PROJECT / "runs/e10c_temperature_plan.json"
OUTPUT = PROJECT / "runs/e10c_temperature_seed42"
RAW = "raw_equal"
CAL = "temperature_equal"
CB, C0 = "B_temperature_control", "E0_temperature_control"
LABELS = (b.B, b.E0, RAW, CB, C0, CAL)
FOLDS = 5
RETENTION = {"max_dice_loss_vs_raw_equal": .003, "max_mae_increase_vs_raw_equal": .003,
             "max_each_failure_count_increase_vs_raw_equal": 0, "min_folds_retaining_raw_dice": 4,
             "max_each_fold_near_zero_or_zero_increase_vs_raw_equal": 0}


def source_hashes():
    return {**b.source_hashes(), **{p.relative_to(PROJECT).as_posix(): file_hash(p) for p in sorted(PACKAGE.glob("*.py"))}}


def freeze():
    if PLAN.exists():
        check()
        return
    if OUTPUT.exists():
        raise ValueError("E10-C outputs predate freeze; preserve and inspect")
    b.verify()
    prior_plan, records = b.check()
    if b.read_lock()["status"] != "NO_CALIBRATION_PASS_SELECTION_BLOCKED":
        raise ValueError("Require unchanged E10-B blocked outcome")
    folded = assign_folds([r for r in records if r["split"] == "calibration"], FOLDS, 42)
    protected = dict(prior_plan["protected_inputs"])
    protected.update(read_json(b.OUTPUT / "calibration/audit.json")["outputs"])
    for path in (b.PLAN, b.OUTPUT / "locked_config.json", b.OUTPUT / "freeze_audit.json", b.OUTPUT / "calibration/audit.json"):
        protected[path.relative_to(PROJECT).as_posix()] = file_hash(path)
    plan = {"stage": "E10_C_BOUNDED_TEMPERATURE_EQUAL_FUSION", "frozen_at_singapore": b.a.e9.repeat.now(),
            "source_hashes": source_hashes(), "protected_inputs": protected, "models": prior_plan["models"],
            "membership": prior_plan["membership"], "fold_assignment": [{"sample_id": r["sample_id"],
                     "group_id": r["group_id"], "cv_fold": r["cv_fold"]} for r in folded],
            "method": {"temperature_bounds": list(BOUNDS), "probability_logit_clip": EPSILON,
                       "pixels_per_image": PIXELS_PER_IMAGE, "seed": 42, "folds": FOLDS,
                       "fit": "independent B/E0 bounded image-equal pixel NLL; convex inverse-temperature bisection64",
                       "sampling": "same deterministic uniform pixels for both models; natural prevalence, no class balancing",
                       "fusion": "0.5*sigmoid(logit(pB)/TB)+0.5*sigmoid(logit(pE0)/T0)", "threshold": .5,
                       "rounding": "single calibrated probabilities preserve original >=0.5 masks with one-ULP float32 boundary guard, no GT"},
            "gates_vs_both_singles": b.GATES, "retention_gates_vs_raw_equal": RETENTION,
            "workflow": "One policy only. Five patient folds fit on four, evaluate held patient fold. Aggregate OOF gates + fold guards must pass before any full-calibration refit. Then same aggregate safety/retention gates on full calibration; lock temperatures only if pass. Selection once, no tuning.",
            "controls": list(LABELS), "primary_unit": "image mean; patient mean/GT area/NLL/Brier descriptive only",
            "data_role": prior_plan["data_role"],
            "independent_confirmation": "Not performed on FAR/C5/C6. Requires a new audited never-used development/test role under a separate protocol, only after a development pass.",
            "restrictions": {"network_training": 0, "model_inferences": 0, "C6_evaluations": 0,
                             "temperature_policies": 1, "alpha_search": False, "threshold_search": False,
                             "bias_fit": False, "dynamic_routing": False, "automatic_external_test": False},
            "reference": "Guo et al.2017, https://proceedings.mlr.press/v70/guo17a.html; classifier calibration motivates this segmentation experiment, does not establish segmentation/fusion safety.",
            "failure_policy": "Preserve B/E0 and all historical NO_PASS; stop this temperature policy if OOF/full fit/selection fails. No expanded bounds, fallback fit, case removal or altered fold partition."}
    plan["plan_signature"] = signature(plan)
    write_json(PLAN, plan)
    print("E10_C_PROTOCOL_FROZEN; FIVE_FOLD_FITS_NOT_STARTED", flush=True)


def check():
    plan = read_json(PLAN)
    if (signature({k: v for k, v in plan.items() if k != "plan_signature"}) != plan["plan_signature"]
            or plan["source_hashes"] != source_hashes() or plan["gates_vs_both_singles"] != b.GATES
            or plan["retention_gates_vs_raw_equal"] != RETENTION
            or plan["method"]["temperature_bounds"] != list(BOUNDS)):
        raise ValueError("E10-C code/plan/config changed")
    _, records = b.check()
    for path, expected in plan["protected_inputs"].items():
        if file_hash(b.a.safe_path(PROJECT, path)) != expected:
            raise ValueError("Protected artifact changed: " + path)
    folded = assign_folds([r for r in records if r["split"] == "calibration"], FOLDS, 42)
    if [{"sample_id": r["sample_id"], "group_id": r["group_id"], "cv_fold": r["cv_fold"]} for r in folded] != plan["fold_assignment"]:
        raise ValueError("Patient fold allocation changed")
    print("E10_C_CHECK_PASS; ONE_POLICY; FROZEN_B/E0; NO_NEW_INFERENCE", flush=True)
    return plan, records, folded


def index_e0():
    rows = [r for r in b.a.read_csv(b.a.CACHE / "per_image.csv") if r["model"] == "e0"]
    index = {r["sample_id"]: r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate E0 index")
    return index


def collect_samples(records):
    if not records or any(r["split"] != "calibration" for r in records):
        raise ValueError("Temperature fitting accepts calibration only")
    index, samples = index_e0(), {}
    for r in records:
        pb, p0, gt = b.load_pair(r, index)
        samples[r["sample_id"]] = sample_pair(pb, p0, gt, r["sample_id"])
    return samples


def fit_pair(training, samples):
    if not training or any(r["split"] != "calibration" for r in training):
        raise ValueError("Only calibration training records can fit temperatures")
    values = [samples[r["sample_id"]] for r in training]
    return {"B": fit_temperature(values, "b"), "E0": fit_temperature(values, "e0"),
            "training_ids": [r["sample_id"] for r in training],
            "training_groups": sorted({r["group_id"] for r in training}),
            "sample_indices_hashes": {r["sample_id"]: samples[r["sample_id"]]["sample_indices_sha256"] for r in training}}


def temperatures(parameters):
    return parameters["B"]["temperature"], parameters["E0"]["temperature"]


def evaluate_records(records, parameters, output=None, fold=None, quiet=False):
    tb, t0 = temperatures(parameters)
    index, rows = index_e0(), []
    for pos, r in enumerate(records, 1):
        pb, p0, gt = b.load_pair(r, index)
        qb, q0 = apply_temperature(pb, tb), apply_temperature(p0, t0)
        if not np.array_equal(pb >= .5, qb >= .5) or not np.array_equal(p0 >= .5, q0 >= .5):
            raise ValueError("Positive-temperature single-model masks changed")
        arrays = (pb, p0, b.fuse(pb, p0, .5), qb, q0, b.fuse(qb, q0, .5))
        for label, probability in zip(LABELS, arrays):
            target = gt.astype(np.float32)
            z = logits(probability)
            row = {"model": label, "sample_id": r["sample_id"], "dataset": r["source"], "split": r["split"],
                   "group_id": r["group_id"], "declared_group_id": r["declared_group_id"],
                   "cv_fold": -1 if fold is None else fold, "height": gt.shape[0], "width": gt.shape[1],
                   "gt_area_fraction": float(gt.mean()), **binary_metrics(probability >= .5, gt),
                   "mae": float(np.abs(probability - target).mean()),
                   "full_image_nll": float(np.mean(np.logaddexp(0, z) - gt * z)),
                   "full_image_brier": float(np.mean((probability.astype(np.float64) - gt) ** 2))}
            rows.append(row)
            if output is not None and label == CAL:
                destination = b.a.safe_path(output, "probabilities/" + r["stem"] + ".npz")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.with_suffix(".npz.tmp").open("wb") as handle:
                    np.savez_compressed(handle, probability=probability)
                destination.with_suffix(".npz.tmp").replace(destination)
        if not quiet and (pos % 32 == 0 or pos == len(records)):
            print(f"[E10-C evaluate {pos}/{len(records)}] fold={fold}; TB={tb:.5f}, T0={t0:.5f}", flush=True)
    return rows


def summaries(rows):
    return [{"model": label, **metric_summary([r for r in rows if r["model"] == label]),
             "full_image_nll": float(np.mean([r["full_image_nll"] for r in rows if r["model"] == label])),
             "full_image_brier": float(np.mean([r["full_image_brier"] for r in rows if r["model"] == label]))}
            for label in LABELS]


def assess(rows):
    results = {r["model"]: r for r in summaries(rows)}
    gates, deltas = b.performance_gates(results[CAL], {label: results[label] for label in (b.B, b.E0)})
    raw, calibrated = results[RAW], results[CAL]
    gates["raw_equal_dice_retained"] = calibrated["dice"] >= raw["dice"] - RETENTION["max_dice_loss_vs_raw_equal"] - b.TOLERANCE
    gates["raw_equal_mae_protected"] = calibrated["mae"] <= raw["mae"] + RETENTION["max_mae_increase_vs_raw_equal"] + b.TOLERANCE
    for key in ("dice_lt05", "dice_lt01", "zero_dice"):
        gates["raw_equal_" + key + "_protected"] = calibrated[key] <= raw[key]
    deltas.append({"reference": RAW, **{key + "_delta": calibrated[key] - raw[key]
                                     for key in ("dice", "mae", "dice_lt05", "dice_lt01", "zero_dice")}})
    return gates, deltas


def assess_oof(rows):
    gates, deltas = assess(rows)
    fold_metrics, retained = [], 0
    for fold in range(FOLDS):
        values = {r["model"]: r for r in summaries([r for r in rows if r["cv_fold"] == fold])}
        raw, cal = values[RAW], values[CAL]
        retained += int(cal["dice"] >= raw["dice"] - RETENTION["max_dice_loss_vs_raw_equal"] - b.TOLERANCE)
        for metric in ("dice_lt01", "zero_dice"):
            gates[f"fold{fold}_{metric}_vs_raw"] = cal[metric] <= raw[metric]
        fold_metrics.extend({"cv_fold": fold, **r} for r in values.values())
    gates["at_least_four_folds_retain_raw_dice"] = retained >= RETENTION["min_folds_retaining_raw_dice"]
    return gates, deltas, fold_metrics


def write_audit(folder):
    write_json(folder / "audit.json", {"plan_sha256": file_hash(PLAN),
               "outputs": {p.relative_to(PROJECT).as_posix(): file_hash(p) for p in folder.rglob("*")
                           if p.is_file() and p.name != "audit.json"}})


def verify_audit(folder):
    audit = read_json(folder / "audit.json")
    if audit["plan_sha256"] != file_hash(PLAN):
        raise ValueError("Phase plan changed")
    for path, expected in audit["outputs"].items():
        if file_hash(b.a.safe_path(PROJECT, path)) != expected:
            raise ValueError("Phase artifact changed: " + path)


def save_diagnostics(folder, rows, decision):
    metrics = summaries(rows)
    write_csv(folder / "per_image.csv", rows)
    write_csv(folder / "summary.csv", metrics)
    write_csv(folder / "groups.csv", [{"model": label, **r} for label in LABELS
              for r in summarize_groups([x for x in rows if x["model"] == label])])
    paired, counts, intervals, patients = b.paired_diagnostics(rows, CAL)
    write_csv(folder / "paired_deltas.csv", paired)
    write_csv(folder / "by_patient.csv", patients)
    write_json(folder / "failure_transitions.json", counts)
    write_json(folder / "patient_bootstrap.json", intervals)
    write_json(folder / "decision.json", decision)
    (folder / "report.md").write_text(b.report_table("E10-C " + folder.name, metrics) +
             "\n\n状态：" + decision["status"] + "\n\n温度独立按训练折NLL拟合，不按Dice搜索；相同患者不跨折。\n\n" +
             "旧数据复用、自适应开发，不是新的独立确认。失败时保留原B，不扩展温度/权重/阈值范围。\n", encoding="utf-8")


def cv():
    with b.a.e9.repeat.serial_training_lock():
        plan, _, folded = check()
        folder = OUTPUT / "cross_validation"
        if folder.exists():
            verify_cv(folded)
            print("EXISTING_OOF_PRESERVED; NO_NEW_FIT_POLICY")
            return
        OUTPUT.mkdir(exist_ok=False)
        folder.mkdir()
        import shutil
        for path in plan["source_hashes"]:
            destination = OUTPUT / "source_snapshot" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / path, destination)
        write_json(folder / "run_info.json", {"status": "RUNNING", "network_training": 0, "selection_evaluations": 0})
        try:
            samples, rows, fits = collect_samples(folded), [], []
            for fold in range(FOLDS):
                training = [r for r in folded if r["cv_fold"] != fold]
                held = [r for r in folded if r["cv_fold"] == fold]
                if {r["group_id"] for r in training} & {r["group_id"] for r in held}:
                    raise ValueError("Patient fold leakage")
                fit = fit_pair(training, samples)
                fit.update(cv_fold=fold, held_ids=[r["sample_id"] for r in held], held_groups=sorted({r["group_id"] for r in held}))
                fits.append(fit)
                print("FOLD_TEMPERATURES", fold, temperatures(fit), flush=True)
                rows.extend(evaluate_records(held, fit, fold=fold))
            gates, deltas, fold_metrics = assess_oof(rows)
            decision = {"stage": "E10_C_PATIENT_OOF", "status": "OOF_PASS_READY_FOR_FULL_REFIT" if all(gates.values()) else "NO_OOF_PASS_RETAIN_B",
                        "gates": gates, "deltas": deltas, "plan_sha256": file_hash(PLAN), "selection_evaluations": 0,
                        "full_calibration_fit_allowed": all(gates.values()), "independent_confirmation": False}
            write_json(folder / "fold_fits.json", fits)
            write_csv(folder / "fold_assignment.csv", [{k: r[k] for k in ("sample_id", "group_id", "cv_fold")} for r in folded])
            write_csv(folder / "fold_metrics.csv", fold_metrics)
            save_diagnostics(folder, rows, decision)
            write_json(folder / "run_info.json", {"status": "COMPLETED", "patient_groups": len({r["group_id"] for r in folded}),
                       "images": len(folded), "folds": FOLDS, "temperature_fits": 10, "network_training": 0,
                       "model_inferences": 0, "selection_evaluations": 0})
            write_audit(folder)
            print("E10_C_OOF_COMPLETE", decision["status"], deltas, flush=True)
        except BaseException as error:
            write_json(folder / "run_info.json", {"status": "FAILED", "error": str(error), "note": "Preserve partial attempt; request recovery guidance."})
            raise


def refit():
    with b.a.e9.repeat.serial_training_lock():
        _, _, folded = check()
        verify_cv(folded)
        prior = read_json(OUTPUT / "cross_validation/decision.json")
        if prior["status"] != "OOF_PASS_READY_FOR_FULL_REFIT":
            print("FULL_REFIT_AND_SELECTION_BLOCKED; NO_OOF_PASS", flush=True)
            return
        folder = OUTPUT / "full_calibration"
        if folder.exists():
            verify_audit(folder)
            read_lock()
            print("EXISTING_FULL_REFIT_PRESERVED")
            return
        folder.mkdir()
        write_json(folder / "run_info.json", {"status": "RUNNING"})
        try:
            fit = fit_pair(folded, collect_samples(folded))
            rows = evaluate_records(folded, fit)
            gates, deltas = assess(rows)
            passed = all(gates.values())
            decision = {"stage": "E10_C_FULL_CALIBRATION", "status": "FULL_CALIBRATION_PASS" if passed else "NO_FULL_CALIBRATION_PASS_RETAIN_B",
                        "gates": gates, "deltas": deltas}
            save_diagnostics(folder, rows, decision)
            write_json(folder / "fit.json", fit)
            lock = {"stage": "E10_C_TEMPERATURE_LOCK", "status": "READY_FOR_SELECTION" if passed else "SELECTION_BLOCKED_FULL_CALIBRATION_FAIL",
                    "plan_sha256": file_hash(PLAN), "temperatures": {"B": temperatures(fit)[0], "E0": temperatures(fit)[1]},
                    "alpha_e0": .5, "threshold": .5, "formal_test_allowed": False,
                    "oof_audit_sha256": file_hash(OUTPUT / "cross_validation/audit.json")}
            lock["lock_signature"] = signature(lock)
            write_json(OUTPUT / "locked_config.json", lock)
            write_json(folder / "run_info.json", {"status": "COMPLETED", "locked_config_sha256": file_hash(OUTPUT / "locked_config.json")})
            write_audit(folder)
            print("E10_C_REFIT_LOCKED", lock["status"], lock["temperatures"], flush=True)
        except BaseException as error:
            write_json(folder / "run_info.json", {"status": "FAILED", "error": str(error)})
            raise


def read_lock():
    lock = read_json(OUTPUT / "locked_config.json")
    if (signature({k: v for k, v in lock.items() if k != "lock_signature"}) != lock["lock_signature"]
            or lock["plan_sha256"] != file_hash(PLAN)
            or lock["oof_audit_sha256"] != file_hash(OUTPUT / "cross_validation/audit.json")
            or file_hash(OUTPUT / "locked_config.json") != read_json(OUTPUT / "full_calibration/run_info.json")["locked_config_sha256"]):
        raise ValueError("Frozen temperature lock changed")
    return lock


def evaluate():
    with b.a.e9.repeat.serial_training_lock():
        plan, records, folded = check()
        verify_cv(folded)
        if not read_json(OUTPUT / "cross_validation/decision.json")["full_calibration_fit_allowed"]:
            print("SELECTION_BLOCKED_BY_OOF; NO_SELECTION_CALCULATION")
            return
        verify_audit(OUTPUT / "full_calibration")
        lock = read_lock()
        if lock["status"] != "READY_FOR_SELECTION":
            print("SELECTION_BLOCKED_BY_FULL_CALIBRATION")
            return
        folder = OUTPUT / "selection"
        if folder.exists():
            verify_audit(folder)
            print("EXISTING_SELECTION_PRESERVED; NO_NEW_EVALUATION")
            return
        folder.mkdir()
        lock_hash = file_hash(OUTPUT / "locked_config.json")
        write_json(folder / "run_info.json", {"status": "RUNNING", "locked_config_sha256": lock_hash})
        try:
            fit = {"B": {"temperature": lock["temperatures"]["B"]}, "E0": {"temperature": lock["temperatures"]["E0"]}}
            rows = evaluate_records([r for r in records if r["split"] == "selection"], fit, folder)
            gates, deltas = assess(rows)
            decision = {"stage": "E10_C_LOCKED_SELECTION_DEVELOPMENT", "status": "DEVELOPMENT_PASS" if all(gates.values()) else "NO_PASS_RETAIN_B",
                        "gates": gates, "deltas": deltas, "locked_config_sha256": lock_hash,
                        "data_role": plan["data_role"], "formal_test_allowed": False,
                        "next_step": "New independent audited data under a separate protocol; no automatic test" if all(gates.values()) else "Stop temperature policy; retain B; no expanded parameter/routing search"}
            save_diagnostics(folder, rows, decision)
            if file_hash(OUTPUT / "locked_config.json") != lock_hash:
                raise ValueError("Temperature lock changed during selection")
            write_json(folder / "run_info.json", {"status": "COMPLETED", "locked_config_sha256": lock_hash,
                       "network_training": 0, "model_inferences": 0, "C6_evaluations": 0})
            write_audit(folder)
            print("E10_C_SELECTION_COMPLETE", decision["status"], deltas, flush=True)
        except BaseException as error:
            write_json(folder / "run_info.json", {"status": "FAILED", "error": str(error)})
            raise


def compare_rows(saved, replay):
    if len(saved) != len(replay):
        raise ValueError("Replay membership differs")
    index = {(r["model"], r["sample_id"]): r for r in saved}
    if len(index) != len(saved) or set(index) != {(r["model"], r["sample_id"]) for r in replay}:
        raise ValueError("Missing/duplicate replay rows")
    for r in replay:
        old = index[r["model"], r["sample_id"]]
        for key in ("dice", "iou", "mae", "tp", "fp", "fn", "full_image_nll", "full_image_brier"):
            if not np.isclose(float(old[key]), r[key], rtol=0, atol=1e-12):
                raise ValueError("Replay metric differs: " + key)


def verify_cv(folded=None):
    if folded is None:
        _, _, folded = check()
    folder = OUTPUT / "cross_validation"
    verify_audit(folder)
    fits = read_json(folder / "fold_fits.json")
    if len(fits) != FOLDS:
        raise ValueError("Require all five OOF fits")
    rows = []
    for fit in fits:
        fold = fit["cv_fold"]
        training = [r for r in folded if r["cv_fold"] != fold]
        held = [r for r in folded if r["cv_fold"] == fold]
        if (fit["training_ids"] != [r["sample_id"] for r in training]
                or fit["held_ids"] != [r["sample_id"] for r in held]
                or set(fit["training_groups"]) & set(fit["held_groups"])):
            raise ValueError("OOF training/heldout patient membership changed")
        rows.extend(evaluate_records(held, fit, fold=fold, quiet=True))
    compare_rows(b.a.read_csv(folder / "per_image.csv"), rows)
    gates, deltas, _ = assess_oof(rows)
    decision = read_json(folder / "decision.json")
    if decision["gates"] != gates or decision["deltas"] != deltas or decision["full_calibration_fit_allowed"] != all(gates.values()):
        raise ValueError("OOF decision changed")
    print("E10_C_OOF_REPLAY_VERIFIED; SAME_FIVE_FROZEN_FITS", flush=True)


def verify():
    _, records, folded = check()
    verify_cv(folded)
    prior = read_json(OUTPUT / "cross_validation/decision.json")
    if not prior["full_calibration_fit_allowed"]:
        if (OUTPUT / "full_calibration").exists() or (OUTPUT / "selection").exists() or (OUTPUT / "locked_config.json").exists():
            raise ValueError("Downstream outputs exist despite OOF block")
        print("E10_C_VERIFIED_NO_OOF_PASS; FULL_FIT/SELECTION/INDEPENDENT_CONFIRMATION_NOT_EXECUTED")
        return
    verify_audit(OUTPUT / "full_calibration")
    lock = read_lock()
    fit = read_json(OUTPUT / "full_calibration/fit.json")
    rows = evaluate_records(folded, fit, quiet=True)
    compare_rows(b.a.read_csv(OUTPUT / "full_calibration/per_image.csv"), rows)
    if lock["status"] != "READY_FOR_SELECTION":
        if (OUTPUT / "selection").exists():
            raise ValueError("Selection exists despite full-fit block")
        print("E10_C_VERIFIED_FULL_CALIBRATION_NO_PASS")
        return
    folder = OUTPUT / "selection"
    verify_audit(folder)
    rows = evaluate_records([r for r in records if r["split"] == "selection"], fit, quiet=True)
    compare_rows(b.a.read_csv(folder / "per_image.csv"), rows)
    gates, deltas = assess(rows)
    if read_json(folder / "decision.json")["gates"] != gates or read_json(folder / "decision.json")["deltas"] != deltas:
        raise ValueError("Selection decision changed")
    print("E10_C_ALL_PHASES_VERIFIED; NO_NEW_MODEL_INFERENCE_OR_POLICY")


if __name__ == "__main__":
    os.chdir(PROJECT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check", "cv", "refit", "evaluate", "verify"))
    args = parser.parse_args()
    {"freeze": freeze, "check": check, "cv": cv, "refit": refit, "evaluate": evaluate, "verify": verify}[args.command]()
