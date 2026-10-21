"""E10-D: calibration-only local rescue diagnosis; no deployment/selection."""
import argparse
import os
import shutil
import sys
from pathlib import Path
import numpy as np

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
sys.path.insert(0, str(PROJECT / "e10c_windows"))
import run_e10c as c
from rescue import (FEATURES, BASE_FEATURES, ACCEPT, RIDGE, candidate_regions, region_features,
                    supplement, isolated_label, fit_gate, score_gate, discrimination)
from common import read_json, write_json, file_hash, signature, write_csv, metric_summary
from e0_protocol import binary_metrics

ARCHIVED_PLAN = PROJECT / "runs/e10d_rescue_plan.json"
ARCHIVED_OUTPUT = PROJECT / "runs/e10d_rescue_seed42"
PLAN = PROJECT / "runs/e10d_rescue_compat_plan.json"
OUTPUT = PROJECT / "runs/e10d_rescue_compat_seed42"
VIEWS = ("hflip", "vflip")
LABELS = (c.b.B, c.b.E0, "raw_equal", "whole_max_control", "all_components_control",
          "no_tta_gate_ablation", "tta_rescue_oof")
POLICY = {"features": list(FEATURES), "ablation_features": list(BASE_FEATURES),
          "accept_score": ACCEPT, "ridge_mean_nll": RIDGE, "min_component_pixels": 1,
          "connectivity": 8, "folds": 5, "seed": 42, "mask_threshold": .5,
          "min_accepted_components": 10, "min_accepted_patients": 5,
          "min_accepted_component_precision": .8, "min_weighted_auc": .6,
          "min_auc_gain_vs_no_tta": .02, "max_dice_loss_vs_raw_equal": .003,
          "min_nonworse_dice_folds_vs_B": 4}


def configure_backend(torch, label):
    """Keep each identity-cache producer's precision, not one shared setting."""
    if label not in (c.b.B, c.b.E0):
        raise ValueError("Unknown frozen model")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = label == c.b.E0
    return {"cudnn_benchmark": False, "cudnn_deterministic": True,
            "matmul_allow_tf32": False, "cudnn_allow_tf32": label == c.b.E0}


def source_hashes():
    return {**c.source_hashes(), **{p.relative_to(PROJECT).as_posix(): file_hash(p) for p in sorted(PACKAGE.glob("*.py"))}}


def freeze():
    if PLAN.exists():
        check()
        return
    if OUTPUT.exists():
        raise ValueError("Outputs predate freeze; preserve and inspect")
    c.verify()
    prior, _, folded = c.check()
    if read_json(c.OUTPUT / "cross_validation/decision.json")["status"] != "NO_OOF_PASS_RETAIN_B":
        raise ValueError("Require preserved E10-C failure")
    protected = {**prior["protected_inputs"], **read_json(c.OUTPUT / "cross_validation/audit.json")["outputs"]}
    for path in (c.PLAN, c.OUTPUT / "cross_validation/audit.json", PROJECT / "pretrained_pth/pvt_v2_b2.pth"):
        protected[path.relative_to(PROJECT).as_posix()] = file_hash(path)
    archived = read_json(ARCHIVED_PLAN)
    if (archived["policy"] != POLICY or read_json(ARCHIVED_OUTPUT / "tta/run_info.json")["status"] != "FAILED"
            or (ARCHIVED_OUTPUT / "diagnosis").exists()):
        raise ValueError("Compatibility repair requires preserved pre-diagnosis failed attempt")
    for name, expected in archived["source_hashes"].items():
        if file_hash(ARCHIVED_OUTPUT / "source_snapshot" / name) != expected:
            raise ValueError("Original frozen source snapshot changed")
    protected[ARCHIVED_PLAN.relative_to(PROJECT).as_posix()] = file_hash(ARCHIVED_PLAN)
    for path in ARCHIVED_OUTPUT.rglob("*"):
        if path.is_file():
            protected[path.relative_to(PROJECT).as_posix()] = file_hash(path)
    plan = {"stage": "E10_D_LOCAL_RESCUE_FEASIBILITY_ONLY", "frozen_at_singapore": c.b.a.e9.repeat.now(),
            "source_hashes": source_hashes(), "protected_inputs": protected, "models": prior["models"],
            "records": folded, "policy": POLICY, "gates_vs_singles": c.b.GATES,
            "method": "8-connected regions of (pE0>=.5)&(pB<.5); accepted regions use max(pB,pE0), elsewhere pB. hflip/vflip aligned to original coordinates; cached identity maps stay authoritative.",
            "gate": "Fixed ridge logistic useful-component classifier. Patient/image/component-equal weights; train-only weighted standardization. No class balancing; fixed admission0.8. Five patient folds; no full-data fitted deployment gate.",
            "label": "Isolated addition relative to B improves Dice>1e-6 AND probability MAE<=1e-12 increase. GT allowed only in training folds and held-out outcome assessment, never prediction features or admission.",
            "ablation": "Same fixed logistic procedure without the two TTA features. Diagnostic comparator, never a fallback selected after seeing results.",
            "inference": {"calibration_images": 160, "patients": 75, "flipped_forwards": 640,
                          "identity_audit_forwards": 320, "fp32": True, "input_size": 352,
                          "backend": "B cuDNN TF32 off; E0 cuDNN TF32 on as historical E6 producer; matmul TF32 off, benchmark off, deterministic on for both. FP32 tensors/storage, E0 convolution internal TF32 allowed.",
                          "identity_tolerances": {"mean_abs": 2e-5, "max_abs": .005}},
            "execution_repair": {"original_plan_sha256": file_hash(ARCHIVED_PLAN),
                                 "reason": "TF32-off E0 identity failed before any fusion/gate outcomes. Restoring historical E6 cuDNN TF32 gives exact identity on the failing image; tolerances/policy/folds unchanged. Original attempt preserved, new output, no mixed old/new views.",
                                 "failed_image_probe": {"stem": "Patient125__1403-03-051141916-6343",
                                     "tf32_off_mean_abs": 3.584110447503181e-5, "tf32_off_max_abs": .002900242805480957,
                                     "legacy_tf32_on_mean_abs": 0., "legacy_tf32_on_max_abs": 0.}},
            "scope": "Only original calibration160. No new selection fusion, C5/C6, independent dataset, base-network training, checkpoint selection or negative-frame claim.",
            "data_role": "Adaptive reused FAR development. Patient CV prevents current gate-fit leakage, not historical design overfitting. Neither an untouched test nor confirmation.",
            "stop": "Any failure retains B; no threshold/feature/ridge/fold/view search. Even feasibility pass requires separately authorized frozen next protocol; never auto-refit, auto-selection or deploy."}
    plan["plan_signature"] = signature(plan)
    write_json(PLAN, plan)
    print("E10_D_FROZEN; CALIBRATION_ONLY; NO_OUTCOMES_COMPUTED", flush=True)


def check():
    plan = read_json(PLAN)
    if (signature({k: v for k, v in plan.items() if k != "plan_signature"}) != plan["plan_signature"]
            or plan["source_hashes"] != source_hashes() or plan["policy"] != POLICY):
        raise ValueError("E10-D frozen source/config changed")
    _, _, folded = c.check()
    if plan["records"] != folded or len(folded) != 160 or any(r["split"] != "calibration" for r in folded):
        raise ValueError("Calibration/fold membership changed")
    for name, expected in plan["protected_inputs"].items():
        if file_hash(c.b.a.safe_path(PROJECT, name)) != expected:
            raise ValueError("Protected input changed: " + name)
    print("E10_D_CHECK_PASS; FROZEN_B/E0; CALIBRATION160_ONLY", flush=True)
    return plan, folded


def item_path(model, r):
    return OUTPUT / "tta" / model / "items" / (r["stem"] + ".json")


def load_views(model, r):
    item = read_json(item_path(model, r))
    expected = read_json(PLAN)["models"][model]["sha256"]
    if (item["sample_id"] != r["sample_id"] or item["checkpoint_sha256"] != expected
            or item["plan_sha256"] != file_hash(PLAN) or set(item["views"]) != set(VIEWS)):
        raise ValueError("TTA item identity/provenance changed")
    expected_backend = {"cudnn_benchmark": False, "cudnn_deterministic": True,
                        "matmul_allow_tf32": False, "cudnn_allow_tf32": model == c.b.E0}
    if item["backend"] != expected_backend:
        raise ValueError("TTA backend differs from its identity-cache producer")
    audit = item["identity_audit"]
    if audit["mean_abs"] > 2e-5 or audit["max_abs"] > .005:
        raise ValueError("Identity inference protocol mismatch")
    arrays = []
    for view in VIEWS:
        meta = item["views"][view]
        path = c.b.a.safe_path(OUTPUT, meta["path"])
        if file_hash(path) != meta["sha256"]:
            raise ValueError("Committed TTA cache changed")
        arrays.append(c.b.a.load_probability(path, (r["height"], r["width"])))
    return arrays


def infer():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    with c.b.a.e9.repeat.serial_training_lock():
        plan, records = check()
        state = OUTPUT / "tta/run_info.json"
        if state.exists():
            info = read_json(state)
            if info["plan_sha256"] != file_hash(PLAN):
                raise ValueError("Run belongs to another plan")
            if info["status"] == "COMPLETED":
                verify_tta(records)
                print("EXISTING_TTA_PRESERVED; NO_NEW_FORWARDS")
                return
            if info["status"] not in ("FAILED", "INTERRUPTED"):
                raise ValueError("May be active; inspect before recovery, do not launch another GPU run")
        else:
            OUTPUT.mkdir(exist_ok=False)
            (OUTPUT / "tta").mkdir()
            for name in plan["source_hashes"]:
                target = OUTPUT / "source_snapshot" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PROJECT / name, target)
            info = {"status": "INITIALIZING", "plan_sha256": file_hash(PLAN), "started_at": c.b.a.e9.repeat.now()}
        device = c.b.a.require_cuda()
        c.b.a.set_seed(42, device)
        env = c.b.a.environment(device)
        reference = read_json(c.b.a.OUTPUT / "run_info.json")["environment"]
        if env["versions"] != reference["versions"] or env["gpu"] != reference["gpu"]:
            raise ValueError("Environment differs from frozen inference")
        info.update(status="RUNNING", environment=env)
        write_json(state, info)
        index = c.index_e0()
        try:
            for label in (c.b.B, c.b.E0):
                backend = configure_backend(torch, label)
                if label == c.b.B:
                    cp = torch.load(c.b.a.B_CHECKPOINT, map_location="cpu", weights_only=True)
                    model = c.b.a.make_model(cp["config"], initialization=False)
                    model.load_state_dict(cp["model_state"], strict=True)
                    del cp
                else:
                    sys.path.insert(0, str(PROJECT / "e6_windows"))
                    from pvt_timm_compat import install_if_missing
                    install_if_missing()
                    from lib.pvt import PolypPVT
                    model = PolypPVT()
                    model.load_state_dict(torch.load(PROJECT / "model_pth/PolypPVT.pth", map_location="cpu", weights_only=True), strict=True)
                model.to(device).eval()
                with torch.inference_mode():
                    for pos, r in enumerate(records, 1):
                        if source_hashes() != plan["source_hashes"]:
                            raise ValueError("Source changed during inference")
                        item = item_path(label, r)
                        if item.exists():
                            load_views(label, r)
                            continue
                        with Image.open(c.b.a.safe_path(PROJECT, r["image"])) as im:
                            x = c.b.a.normalize_rgb(im.convert("RGB"), 352).unsqueeze(0).to(device)
                        def forward(value, axis=None):
                            raw = model(value)
                            z = raw[0] + raw[1] if label == c.b.E0 else raw
                            if value.device != z.device or z.device != next(model.parameters()).device:
                                raise ValueError("Device mismatch")
                            q = torch.sigmoid(F.interpolate(z.float(), size=(r["height"], r["width"]), mode="bilinear", align_corners=False))[0, 0]
                            if axis is not None:
                                q = torch.flip(q, (axis,))
                            return q.cpu().numpy().astype(np.float32)
                        identity = forward(x)
                        pb, p0, _ = c.b.load_pair(r, index)
                        cached = pb if label == c.b.B else p0
                        difference = np.abs(identity.astype(np.float64) - cached)
                        audit = {"mean_abs": float(difference.mean()), "max_abs": float(difference.max()),
                                 "mask_disagreement_pixels": int(((identity >= .5) != (cached >= .5)).sum())}
                        if audit["mean_abs"] > 2e-5 or audit["max_abs"] > .005:
                            raise ValueError("Identity differs from authoritative cache: " + label + "/" + r["sample_id"])
                        views = {}
                        for view, input_axis, output_axis in (("hflip", 3, 1), ("vflip", 2, 0)):
                            q = forward(torch.flip(x, (input_axis,)), output_axis)
                            relative = f"tta/{label}/{view}/{r['stem']}.npz"
                            target = c.b.a.safe_path(OUTPUT, relative)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            tmp = target.with_suffix(".npz.tmp")
                            with tmp.open("wb") as handle:
                                np.savez_compressed(handle, probability=q)
                            tmp.replace(target)
                            views[view] = {"path": relative, "sha256": file_hash(target)}
                        item.parent.mkdir(parents=True, exist_ok=True)
                        write_json(item, {"sample_id": r["sample_id"], "checkpoint_sha256": plan["models"][label]["sha256"],
                                          "plan_sha256": file_hash(PLAN), "backend": backend, "identity_audit": audit, "views": views})
                        if pos % 16 == 0 or pos == len(records):
                            print(f"[E10-D {label} {pos}/160] identity audit+hflip+vflip; no training", flush=True)
                del model
                torch.cuda.empty_cache()
            info.update(status="COMPLETED", finished_at=c.b.a.e9.repeat.now(), unique_flipped_forwards=640,
                        unique_identity_audit_forwards=320, base_network_training=0, selection_images=0)
            write_json(state, info)
            write_audit(OUTPUT / "tta")
        except BaseException as error:
            info.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
            write_json(state, info)
            raise
    verify_tta(records)


def write_audit(folder):
    write_json(folder / "audit.json", {"plan_sha256": file_hash(PLAN), "outputs": {
        p.relative_to(PROJECT).as_posix(): file_hash(p) for p in folder.rglob("*") if p.is_file() and p.name != "audit.json"}})


def verify_audit(folder):
    audit = read_json(folder / "audit.json")
    if audit["plan_sha256"] != file_hash(PLAN):
        raise ValueError("Audit belongs to changed plan")
    actual = {p.relative_to(PROJECT).as_posix() for p in folder.rglob("*") if p.is_file() and p.name != "audit.json"}
    if actual != set(audit["outputs"]):
        raise ValueError("Missing or unregistered artifact")
    for name, expected in audit["outputs"].items():
        if file_hash(c.b.a.safe_path(PROJECT, name)) != expected:
            raise ValueError("Artifact changed: " + name)


def verify_tta(records):
    if read_json(OUTPUT / "tta/run_info.json")["status"] != "COMPLETED":
        raise ValueError("TTA incomplete")
    verify_audit(OUTPUT / "tta")
    if len(list((OUTPUT / "tta").rglob("*.npz"))) != 640:
        raise ValueError("Unexpected view membership")
    for label in (c.b.B, c.b.E0):
        if len(list((OUTPUT / "tta" / label / "items").glob("*.json"))) != 160:
            raise ValueError("Unexpected patient/image item membership")
        for r in records:
            load_views(label, r)
    print("E10_D_TTA_VERIFIED; 640_FLIPPED_CACHES; CALIBRATION_ONLY", flush=True)


def collect(records):
    rows, index = [], c.index_e0()
    for pos, r in enumerate(records, 1):
        pb, p0, gt = c.b.load_pair(r, index)
        bv, ev = load_views(c.b.B, r), load_views(c.b.E0, r)
        for number, indices in enumerate(candidate_regions(pb, p0), 1):
            rows.append({"sample_id": r["sample_id"], "group_id": r["group_id"], "cv_fold": r["cv_fold"],
                         "component_id": number, "pixels": len(indices),
                         **region_features(pb, p0, bv, ev, indices), **isolated_label(pb, p0, gt, indices)})
        if pos % 32 == 0:
            print(f"[E10-D regions {pos}/160] prediction-only features; GT diagnostic labels separated", flush=True)
    return rows


def crossfit(components, records):
    result, fits = [dict(r) for r in components], []
    for fold in range(5):
        training = [r for r in result if r["cv_fold"] != fold]
        held = [r for r in result if r["cv_fold"] == fold]
        train_records, held_records = [r for r in records if r["cv_fold"] != fold], [r for r in records if r["cv_fold"] == fold]
        train_groups, held_groups = {r["group_id"] for r in train_records}, {r["group_id"] for r in held_records}
        if train_groups & held_groups:
            raise ValueError("Patient fit leakage")
        full, ablated = fit_gate(training), fit_gate(training, BASE_FEATURES)
        for r, score, control in zip(held, score_gate(held, full), score_gate(held, ablated)):
            r.update(tta_score=float(score), no_tta_score=float(control))
        fits.append({"fold": fold, "training_ids": [r["sample_id"] for r in train_records],
                     "heldout_ids": [r["sample_id"] for r in held_records], "training_groups": sorted(train_groups),
                     "heldout_groups": sorted(held_groups), "tta": full, "no_tta": ablated})
    return result, fits


def evaluate(records, components):
    lookup, rows, diagnosis = {}, [], []
    for row in components:
        lookup.setdefault(row["sample_id"], []).append(row)
    index = c.index_e0()
    for r in records:
        pb, p0, gt = c.b.load_pair(r, index)
        regions, meta = candidate_regions(pb, p0), lookup.get(r["sample_id"], [])
        if len(regions) != len(meta) or any(m["component_id"] != j + 1 or m["pixels"] != len(a) for j, (a, m) in enumerate(zip(regions, meta))):
            raise ValueError("Component replay identity mismatch")
        arrays = (pb, p0, c.b.fuse(pb, p0, .5), np.maximum(pb, p0), supplement(pb, p0, regions),
                  supplement(pb, p0, [a for a, m in zip(regions, meta) if m["no_tta_score"] >= ACCEPT]),
                  supplement(pb, p0, [a for a, m in zip(regions, meta) if m["tta_score"] >= ACCEPT]))
        base = binary_metrics(pb >= .5, gt)
        for label, p in zip(LABELS, arrays):
            row = {"model": label, "sample_id": r["sample_id"], "dataset": r["source"], "group_id": r["group_id"],
                   "cv_fold": r["cv_fold"], "split": "calibration", "gt_area_fraction": float(gt.mean()),
                   **binary_metrics(p >= .5, gt), "mae": float(np.abs(p - gt.astype(np.float32)).mean())}
            if label in ("whole_max_control", "all_components_control", "no_tta_gate_ablation", "tta_rescue_oof") and row["tp"] < base["tp"]:
                raise AssertionError("Non-erasing TP invariant violated")
            rows.append(row)
        ppv = base["tp"] / max(1, base["tp"] + base["fp"])
        recall = base["tp"] / int(gt.sum())
        failure = "not_low" if base["dice"] >= .5 else ("zero_overlap" if base["tp"] == 0 else
            "miss_dominant" if recall < ppv else "oversegmentation_dominant_or_mixed")
        diagnosis.append({"sample_id": r["sample_id"], "group_id": r["group_id"], "b_dice": base["dice"],
                          "b_precision": ppv, "b_recall": recall, "b_failure_category_descriptive": failure,
                          "candidate_components": len(meta), "helpful_isolated_components": sum(m["helpful"] for m in meta),
                          "tta_accepted": sum(m["tta_score"] >= ACCEPT for m in meta),
                          "candidate_added_tp": sum(m["added_tp"] for m in meta), "candidate_added_fp": sum(m["added_fp"] for m in meta)})
    return rows, diagnosis


def assess(rows, components):
    summaries = {label: metric_summary([r for r in rows if r["model"] == label]) for label in LABELS}
    gates, deltas = c.b.performance_gates(summaries["tta_rescue_oof"], {label: summaries[label] for label in (c.b.B, c.b.E0)})
    raw, fused = summaries["raw_equal"], summaries["tta_rescue_oof"]
    gates["raw_equal_dice_retained"] = fused["dice"] >= raw["dice"] - .003 - 1e-12
    full, ablated = discrimination(components, "tta_score"), discrimination(components, "no_tta_score")
    gates["at_least_ten_accepted_components"] = full["admitted"] >= 10
    gates["at_least_five_accepted_patients"] = full["admitted_groups"] >= 5
    gates["accepted_precision_at_least_080"] = full["precision"] is not None and full["precision"] >= .8
    gates["accepted_weighted_precision_at_least_080"] = full.get("weighted_precision") is not None and full["weighted_precision"] >= .8
    gates["weighted_auc_at_least_060"] = full["weighted_auc"] is not None and full["weighted_auc"] >= .6
    gates["tta_auc_gain_at_least_002"] = (full["weighted_auc"] is not None and ablated["weighted_auc"] is not None
                                         and full["weighted_auc"] >= ablated["weighted_auc"] + .02)
    folds, nonworse = [], 0
    for fold in range(5):
        sub = {label: metric_summary([r for r in rows if r["model"] == label and r["cv_fold"] == fold]) for label in LABELS}
        nonworse += int(sub["tta_rescue_oof"]["dice"] >= sub[c.b.B]["dice"] - 1e-12)
        for key in ("dice_lt01", "zero_dice"):
            gates[f"fold{fold}_{key}_protected_vs_B"] = sub["tta_rescue_oof"][key] <= sub[c.b.B][key]
        folds.extend({"cv_fold": fold, "model": label, **value} for label, value in sub.items())
    gates["four_folds_nonworse_dice_vs_B"] = nonworse >= 4
    return summaries, folds, {"status": "FEASIBILITY_PASS_NOT_DEPLOYABLE" if all(gates.values()) else "NO_FEASIBILITY_PASS_RETAIN_B",
                             "gates": gates, "deltas": deltas, "tta_discrimination": full, "no_tta_discrimination": ablated,
                             "base_network_training": 0, "gate_fits": 10, "selection_allowed": False,
                             "full_data_refit_allowed": False, "independent_confirmation_performed": False}


def compare_rows(saved, replay):
    index = {(r["model"], r["sample_id"]): r for r in saved}
    if len(index) != len(saved) or len(saved) != len(replay) or set(index) != {(r["model"], r["sample_id"]) for r in replay}:
        raise ValueError("Missing/duplicate replay metric row")
    for row in replay:
        old = index[row["model"], row["sample_id"]]
        for key, value in row.items():
            if isinstance(value, (float, int)):
                if not np.isclose(float(old[key]), value, rtol=0, atol=1e-12):
                    raise ValueError("Metric/identity replay differs: " + key)
            elif old[key] != value:
                raise ValueError("Metric row identity differs: " + key)


def compute(records):
    comp, fits = crossfit(collect(records), records)
    rows, diagnosis = evaluate(records, comp)
    summaries, folds, decision = assess(rows, comp)
    return comp, fits, rows, diagnosis, summaries, folds, decision


def diagnose():
    with c.b.a.e9.repeat.serial_training_lock():
        _, records = check()
        verify_tta(records)
        folder = OUTPUT / "diagnosis"
        if folder.exists():
            verify()
            print("EXISTING_DIAGNOSIS_PRESERVED")
            return
        folder.mkdir()
        write_json(folder / "run_info.json", {"status": "RUNNING", "plan_sha256": file_hash(PLAN)})
        try:
            comp, fits, rows, diagnosis, summaries, folds, decision = compute(records)
            write_csv(folder / "components_oof.csv", comp)
            write_csv(folder / "per_image.csv", rows)
            write_csv(folder / "image_diagnosis.csv", diagnosis)
            write_csv(folder / "summary.csv", [{"model": label, **r} for label, r in summaries.items()])
            write_csv(folder / "fold_metrics.csv", folds)
            write_json(folder / "fold_fits.json", fits)
            write_json(folder / "decision.json", decision)
            transitions = {}
            base = {r["sample_id"]: r for r in rows if r["model"] == c.b.B}
            for label in LABELS[2:]:
                values = [r for r in rows if r["model"] == label]
                transitions[label] = {key: {"new": sum(r["dice"] < threshold and base[r["sample_id"]]["dice"] >= threshold for r in values),
                                             "rescued": sum(r["dice"] >= threshold and base[r["sample_id"]]["dice"] < threshold for r in values)}
                                      for key, threshold in (("low", .5), ("near_zero", .1))}
                transitions[label]["zero"] = {"new": sum(r["dice"] == 0 and base[r["sample_id"]]["dice"] > 0 for r in values),
                                                "rescued": sum(r["dice"] > 0 and base[r["sample_id"]]["dice"] == 0 for r in values)}
            write_json(folder / "failure_transitions.json", transitions)
            lines = ["# E10-D 局部补救可行性诊断", "", "日期：2026-10-14（Asia/Singapore）", "",
                     "状态：" + decision["status"], "", "FAR calibration160张/75患者，5折；B/E0冻结。仅探索，非独立确认，无部署参数。", "",
                     c.b.report_table("OOF / diagnostic controls", [{"model": label, **r} for label, r in summaries.items()]), "",
                     "接纳规则固定0.8；5个预测特征、ridge0.01，无阈值搜索。no_tta是固定消融，不作为失败后的备选。",
                     "GT仅用于训练折helpful标签和留出诊断。图像失败类型由precision/recall粗分，不等于医学病因诊断。",
                     "局部补充不删除B区域，因此阳性样本已有TP不能被抹掉；但FP、Dice、近零、MAE仍可能恶化，不能保证阴性帧安全。", "",
                     "## 接纳与稳定性判别", "", str(decision["tta_discrimination"]), "", str(decision["no_tta_discrimination"]), "",
                     "失败门槛：" + ", ".join(k for k, v in decision["gates"].items() if not v), "",
                     "无全量重拟合、selection、C5/C6或独立确认。失败保留B，不更改旧E10-B/C结论。"]
            (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            write_json(folder / "run_info.json", {"status": "COMPLETED", "plan_sha256": file_hash(PLAN),
                       "base_network_training": 0, "gate_fits": 10, "selection_images": 0})
            write_audit(folder)
            print("E10_D_DIAGNOSIS_COMPLETE", decision["status"], decision["tta_discrimination"], flush=True)
        except BaseException as error:
            write_json(folder / "run_info.json", {"status": "FAILED", "plan_sha256": file_hash(PLAN), "error": str(error)})
            raise


def verify():
    plan, records = check()
    for forbidden in ("selection", "full_calibration", "locked_config.json"):
        if (OUTPUT / forbidden).exists():
            raise ValueError("Downstream artifact forbidden in diagnosis-only stage")
    for name, expected in plan["source_hashes"].items():
        if file_hash(OUTPUT / "source_snapshot" / name) != expected:
            raise ValueError("Source snapshot changed")
    verify_tta(records)
    folder = OUTPUT / "diagnosis"
    verify_audit(folder)
    if read_json(folder / "run_info.json")["status"] != "COMPLETED":
        raise ValueError("Diagnosis incomplete; preserve partial attempt")
    comp, fits, rows, diagnosis, summaries, folds, decision = compute(records)
    if fits != read_json(folder / "fold_fits.json") or decision != read_json(folder / "decision.json"):
        raise ValueError("Exact gate refit/decision replay differs")
    compare_rows(c.b.a.read_csv(folder / "per_image.csv"), rows)
    saved = c.b.a.read_csv(folder / "components_oof.csv")
    if len(saved) != len(comp):
        raise ValueError("Component membership changed")
    for old, new in zip(saved, comp):
        for key, value in new.items():
            if isinstance(value, (float, int)):
                if not np.isclose(float(old[key]), value, rtol=0, atol=1e-12):
                    raise ValueError("Component feature/label/score replay differs")
            elif old[key] != value:
                raise ValueError("Component identity replay differs")
    print("E10_D_VERIFIED; EXACT_TEN_GATE_REFITS; NO_NEW_MODEL_FORWARDS; NO_DEPLOYMENT", flush=True)


if __name__ == "__main__":
    os.chdir(PROJECT)
    c.b.a.torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check", "infer", "diagnose", "verify"))
    args = parser.parse_args()
    {"freeze": freeze, "check": check, "infer": infer, "diagnose": diagnose, "verify": verify}[args.command]()
