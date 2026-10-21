"""E10-A: frozen B/E0 comparison on reused FAR development data. No fusion."""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
sys.path.insert(0, str(PROJECT / "e9_balance_windows"))
import run_e9_balance as e9
import analyze_holdout as descriptive
from common import read_json, write_json, file_hash, signature, read_mask, metric_summary, summarize_groups, write_csv
from models import make_model
from data import normalize_rgb
from engine import require_cuda, set_seed, environment
from e0_protocol import binary_metrics
from prepare_data import fingerprint, near_matches

PLAN = PROJECT / "runs/e10a_far_plan.json"
OUTPUT = PROJECT / "runs/e10a_far_seed42"
MANIFEST = PROJECT / "runs/e6_development_seed42/development_manifest.json"
CACHE = PROJECT / "runs/e6_development_inference_seed42"
B_CHECKPOINT = PROJECT / "runs/e8_B_expanded_seed42/checkpoints/best.pth"
THRESHOLD = .5
BOOTSTRAP_REPETITIONS = 20000


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_path(root, value):
    if not isinstance(value, str) or "\\" in value or ":" in value:
        raise ValueError("Expected a safe relative POSIX path")
    root = Path(root).resolve()
    path = (root / value).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Path escapes expected root")
    return path


def code_hashes():
    extras = [PROJECT / "e8_repeat_windows/run_e8_repeat.py", PROJECT / "e8_windows/analyze_holdout.py"]
    extras += sorted(PACKAGE.glob("*.py"))
    return {**e9.stage_hashes(), **{p.relative_to(PROJECT).as_posix(): file_hash(p) for p in extras}}


def legacy_e6_signature(value):
    # E6 used compact JSON separators; E8/E9 signatures use default separators.
    # Preserve the historical convention rather than rewriting its manifest.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def audit_membership(records):
    if len(records) != 432 or len({r["sample_id"] for r in records}) != 432:
        raise ValueError("Require all 432 distinct FAR images")
    if set(r["source"] for r in records) != {"FAR-POLYP-SEG"}:
        raise ValueError("Unexpected source")
    counts = Counter(r["split"] for r in records)
    if dict(counts) != {"calibration": 160, "selection": 272}:
        raise ValueError("Original FAR partitions changed")
    groups = {split: {r["group_id"] for r in records if r["split"] == split} for split in counts}
    declared = {split: {r["declared_group_id"] for r in records if r["split"] == split} for split in counts}
    if (groups["calibration"] & groups["selection"] or declared["calibration"] & declared["selection"]
            or len(groups["calibration"]) != 75 or len(groups["selection"]) != 112
            or len(declared["calibration"]) != 75 or len(declared["selection"]) != 112):
        raise ValueError("Patient group counts or cross-partition separation changed")
    # group_id is a representative image ID of a connected patient group, not
    # necessarily a bare PatientNNN string. Verify both identifiers explicitly.
    mapping = defaultdict(set)
    for r in records:
        if r["stem"].split("__")[0] != r["declared_group_id"]:
            raise ValueError("Declared patient does not match author filename")
        mapping[r["declared_group_id"]].add(r["group_id"])
    if any(len(values) != 1 for values in mapping.values()):
        raise ValueError("One patient split across connected groups")
    return {"images": 432, "patient_groups": 187, "calibration_images": 160, "calibration_groups": 75,
            "selection_images": 272, "selection_groups": 112, "cross_partition_patient_groups": 0}


def far_records():
    info = read_json(MANIFEST.parent / "run_info.json")
    if info["status"] != "COMPLETED" or info["run_kind"] != "E6_PREPARE" or file_hash(MANIFEST) != info["manifest_sha256"]:
        raise ValueError("Require intact frozen E6 manifest")
    records = read_json(MANIFEST)
    if legacy_e6_signature(records) != info["manifest_signature"]:
        raise ValueError("FAR manifest signature changed")
    audit_membership(records)
    for r in records:
        for key in ("image", "mask"):
            if file_hash(safe_path(PROJECT, r[key])) != r[key + "_sha256"]:
                raise ValueError("FAR content changed: " + r["sample_id"])
    return records


def audit_training_overlap(records, training):
    exact = {r["decoded_rgb_sha256"] for r in records} & {r["decoded_rgb_sha256"] for r in training}
    if exact or {r["sample_id"] for r in records} & {r["sample_id"] for r in training}:
        raise ValueError("FAR overlaps registered B training data")
    matches, train_thumbs = [], {}
    for r in records:
        rgb, dhash, thumb = fingerprint(safe_path(PROJECT, r["image"]))
        if hashlib.sha256(rgb.tobytes()).hexdigest() != r["decoded_rgb_sha256"]:
            raise ValueError("FAR decoded fingerprint changed")
        gt = read_mask(safe_path(PROJECT, r["mask"]))
        if gt.shape != rgb.shape[:2] or not gt.any() or int(gt.sum()) != r["gt_foreground_pixels"]:
            raise ValueError("FAR label shape/content mismatch")
        candidates = [x for x in training if (int(dhash, 16) ^ int(x["dhash64"], 16)).bit_count() <= 2]
        thumbs = []
        for candidate in candidates:
            key = candidate["sample_id"]
            if key not in train_thumbs:
                train_thumbs[key] = fingerprint(PROJECT / candidate["image"])[2]
            thumbs.append(train_thumbs[key])
        near = near_matches({**r, "dhash64": dhash}, thumb, candidates, thumbs)
        matches.extend({"far_sample_id": r["sample_id"], "train_sample_id": key, "rgb16_mae": distance}
                       for key, distance in near)
    if matches:
        raise ValueError("FAR/B training near-duplicate candidate; stop, do not remove cases after seeing outcomes: " + str(matches))
    return {"registered_B_train_images": len(training), "exact_decoded_overlap": 0, "near_duplicate_candidates": 0,
            "heuristic": "E8 dHash Hamming<=2 AND RGB16 MAE<=0.01 with aspect ratio difference<=0.01",
            "limitation": "Image-level heuristic, not proof of all related frames or public-model-author training history."}


def load_probability(path, shape):
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["probability"]:
            raise ValueError("Unexpected NPZ fields")
        p = archive["probability"]
    if p.dtype != np.float32 or p.shape != shape or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("Invalid probability cache")
    return p


def replay_row(record, p, saved=None):
    gt = read_mask(safe_path(PROJECT, record["mask"]))
    metrics = {**binary_metrics(p >= THRESHOLD, gt), "mae": float(np.abs(p - gt.astype(np.float32)).mean()),
               "gt_area_fraction": float(gt.mean())}
    if saved is not None:
        if (saved["sample_id"] != record["sample_id"] or int(saved["height"]) != gt.shape[0]
                or int(saved["width"]) != gt.shape[1]):
            raise ValueError("Cached row identity/size mismatch")
        for key in ("dice", "iou", "mae", "tp", "fp", "fn"):
            if not np.isclose(float(saved[key]), metrics[key], rtol=0, atol=1e-12):
                raise ValueError("Cached metric mismatch: " + record["sample_id"] + "/" + key)
    return {"sample_id": record["sample_id"], "dataset": record["source"], "split": record["split"],
            "group_id": record["group_id"], "declared_group_id": record["declared_group_id"],
            "height": gt.shape[0], "width": gt.shape[1], **metrics}


def e0_cache(records):
    info = read_json(CACHE / "run_info.json")
    manifest_info = read_json(MANIFEST.parent / "run_info.json")
    if (info["status"] != "COMPLETED" or info["run_kind"] != "FULL_DEVELOPMENT" or info["processed_count"] != 432
            or info["manifest_signature"] != manifest_info["manifest_signature"]
            or info["manifest_sha256"] != file_hash(MANIFEST)
            or info["e0_checkpoint_sha256"] != file_hash(PROJECT / "model_pth/PolypPVT.pth")):
        raise ValueError("E0 cache provenance differs")
    required = {"input": "RGB, Pillow bilinear 352x352, ImageNet normalization",
                "e0_output": "sigmoid(resize(P1+P2 logits to original size))", "per_image_minmax": False,
                "storage": "float32 NPZ at original resolution"}
    if any(info["protocol"].get(k) != v for k, v in required.items()):
        raise ValueError("E0 cached probability protocol differs")
    rows = [r for r in read_csv(CACHE / "per_image.csv") if r["model"] == "e0"]
    index = {r["sample_id"]: r for r in rows}
    if len(rows) != 432 or len(index) != 432 or set(index) != {r["sample_id"] for r in records}:
        raise ValueError("E0 cache incomplete/duplicated")
    hashes = {p.relative_to(PROJECT).as_posix(): file_hash(p) for p in (CACHE / "run_info.json", CACHE / "per_image.csv")}
    replay = []
    for r in records:
        saved = index[r["sample_id"]]
        if saved["split"] != r["split"] or saved["source"] != r["source"]:
            raise ValueError("E0 cached partition differs")
        path = safe_path(CACHE, saved["probability_path"])
        if file_hash(path) != saved["probability_sha256"]:
            raise ValueError("E0 probability changed")
        p = load_probability(path, (r["height"], r["width"]))
        replay.append(replay_row(r, p, saved))
        hashes[path.relative_to(PROJECT).as_posix()] = saved["probability_sha256"]
    return index, replay, hashes


def freeze():
    if PLAN.exists():
        check()
        return
    if OUTPUT.exists():
        raise ValueError("E10 outputs predate freeze; preserve and inspect")
    e9.verify_summary()
    decision = read_json(e9.OUTPUT / "decision.json")
    if decision["status"] != "NO_PASS_RETAIN_B" or decision["selected_recipe_for_next_protocol"] != "B_expanded":
        raise ValueError("Expected retained original B after E9")
    _, bundle, manifests, protocol = e9.check()
    _, b_info, _, _ = e9.repeat.verified_run("B_expanded", 42, bundle, manifests, protocol)
    records = far_records()
    overlap = audit_training_overlap(records, manifests["original_train"] + manifests["new_train"])
    _, _, protected = e0_cache(records)
    protected[MANIFEST.relative_to(PROJECT).as_posix()] = file_hash(MANIFEST)
    protected[(MANIFEST.parent / "run_info.json").relative_to(PROJECT).as_posix()] = file_hash(MANIFEST.parent / "run_info.json")
    for path in (e9.PLAN, e9.OUTPUT / "audit.json", e9.OUTPUT / "decision.json"):
        protected[path.relative_to(PROJECT).as_posix()] = file_hash(path)
    plan = {"stage": "E10_A_FAR_FROZEN_MODEL_COMPLEMENTARITY", "frozen_at_singapore": e9.repeat.now(),
            "source_hashes": code_hashes(), "protected_inputs": protected,
            "models": {"B_expanded": {"checkpoint": B_CHECKPOINT.relative_to(PROJECT).as_posix(),
                                        "sha256": b_info["best_checkpoint_sha256"], "seed": 42, "round": 98},
                       "E0_frozen_reference": {"checkpoint": "model_pth/PolypPVT.pth",
                                                "sha256": file_hash(PROJECT / "model_pth/PolypPVT.pth"), "mode": "replay verified E6 cache"}},
            "membership": audit_membership(records), "overlap_audit": overlap,
            "input_size": 352, "threshold": .5, "low_dice": .5, "near_zero_dice": .1,
            "inference": "B FP32 on CUDA, original-size bilinear logits resize then sigmoid; E0 verified cache follows same mathematical preprocessing/output protocol; historical cache GPU numerical settings not refitted.",
            "statistics": {"primary_unit": "image mean", "secondary_unit": "patient mean",
                           "bootstrap": "paired patient-cluster percentile, image-weighted delta, descriptive only",
                           "bootstrap_repetitions": BOOTSTRAP_REPETITIONS, "bootstrap_seed": 42},
            "data_role": "Reused FAR development data; full 432 and both old partitions inspected in E10-A. Any future use of selection is adaptive, never fresh blind confirmation.",
            "diagnostics": ["B/E0 per-image and per-patient Dice/IoU/MAE", "low/near-zero/zero rescue in both directions",
                            "both-low cases, error correlation, mask agreement", "GT-size strata, descriptive bootstrap"],
            "next_stage_indicator": "Two-way threshold rescue is a descriptive reason to consider small fixed-weight development, not a fusion success gate. No fusion auto-launch.",
            "restrictions": {"training_runs": 0, "E0_new_inferences": 0, "C6_evaluations": 0,
                             "fusion_candidates": 0, "threshold_search": False, "calibration_fit": False,
                             "checkpoint_reselection": False},
            "limitations": "One reused positive-only dataset, one B seed; cannot prove new-centre generalization, negative-frame specificity or universal superiority."}
    plan["plan_signature"] = signature(plan)
    write_json(PLAN, plan)
    print("E10_A_PLAN_FROZEN; NO_B_FAR_INFERENCE_YET", PLAN, flush=True)


def check():
    plan = read_json(PLAN)
    if signature({k: v for k, v in plan.items() if k != "plan_signature"}) != plan["plan_signature"] or plan["source_hashes"] != code_hashes():
        raise ValueError("E10 source/plan changed")
    e9.check()
    for name, expected in plan["protected_inputs"].items():
        if file_hash(safe_path(PROJECT, name)) != expected:
            raise ValueError("E10 protected input changed: " + name)
    for model in plan["models"].values():
        if file_hash(safe_path(PROJECT, model["checkpoint"])) != model["sha256"]:
            raise ValueError("Frozen checkpoint changed")
    records = far_records()
    print("E10_A_CHECK_PASS; REUSED_DEVELOPMENT; NO_FUSION", flush=True)
    return plan, records


def infer():
    with e9.repeat.serial_training_lock():
        plan, records = check()
        if OUTPUT.exists():
            info = read_json(OUTPUT / "run_info.json")
            if info["plan_sha256"] != file_hash(PLAN):
                raise ValueError("Existing E10 run belongs to another plan")
            if info["status"] == "COMPLETED":
                verify()
                print("E10_COMPLETED_PRESERVED; NO_NEW_INFERENCE")
                return
            if info["status"] not in ("FAILED", "INTERRUPTED"):
                raise ValueError("May still be active. Inspect process/OS lock before recovering stale state.")
        else:
            OUTPUT.mkdir(exist_ok=False)
            (OUTPUT / "source_snapshot").mkdir()
            import shutil
            for name in plan["source_hashes"]:
                destination = OUTPUT / "source_snapshot" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PROJECT / name, destination)
            write_json(OUTPUT / "manifest_snapshot.json", records)
            info = {"status": "INITIALIZING", "plan_sha256": file_hash(PLAN), "run_kind": "E10_A_DEVELOPMENT_ONLY",
                    "started_at_singapore": e9.repeat.now(), "B_completed_images": 0,
                    "E0_new_inferences": 0, "fusion_candidates": 0, "C6_evaluations": 0, "training_runs": 0}
        device = require_cuda()
        set_seed(42, device)
        env = environment(device)
        reference = read_json(PROJECT / "runs/e8_B_expanded_seed42/run_info.json")["environment"]
        if env["versions"] != reference["versions"] or env.get("gpu") != reference.get("gpu"):
            raise ValueError("Environment differs from frozen B")
        if "environment" in info and info["environment"]["versions"] != env["versions"]:
            raise ValueError("Resume environment changed")
        info.update(status="RUNNING", environment=env)
        write_json(OUTPUT / "run_info.json", info)
        try:
            cp = torch.load(B_CHECKPOINT, map_location="cpu", weights_only=True)
            if cp["epoch"] != 98 or cp["config"] != e9.trainer.make_config("B_expanded", 42):
                raise ValueError("Frozen B checkpoint metadata differs")
            model = make_model(cp["config"], initialization=False)
            model.load_state_dict(cp["model_state"], strict=True)
            model.to(device).eval()
            del cp
            with torch.inference_mode():
                for pos, record in enumerate(records, 1):
                    if code_hashes() != plan["source_hashes"]:
                        raise ValueError("E10 code changed during inference")
                    item = OUTPUT / "B_expanded/items" / (record["stem"] + ".json")
                    if item.exists():
                        replay_b_item(record, item)
                    else:
                        with Image.open(safe_path(PROJECT, record["image"])) as im:
                            x = normalize_rgb(im.convert("RGB"), 352).unsqueeze(0).to(device)
                        logits = model(x)
                        if not x.device == logits.device == next(model.parameters()).device == device:
                            raise ValueError("CUDA device proof differs")
                        p = torch.sigmoid(F.interpolate(logits.float(), size=(record["height"], record["width"]),
                                                       mode="bilinear", align_corners=False))[0, 0].cpu().numpy().astype(np.float32)
                        relative = "B_expanded/probabilities/" + record["stem"] + ".npz"
                        target = safe_path(OUTPUT, relative)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        # The JSON item is the commit marker. Uncommitted NPZ bytes
                        # can be replaced on recovery, never a committed prediction.
                        temporary = target.with_suffix(".npz.tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(handle, probability=p)
                        temporary.replace(target)
                        row = replay_row(record, load_probability(target, (record["height"], record["width"])))
                        item.parent.mkdir(parents=True, exist_ok=True)
                        write_json(item, {**row, "probability_path": relative, "probability_sha256": file_hash(target),
                                          "checkpoint_sha256": plan["models"]["B_expanded"]["sha256"]})
                        info["device_proof"] = {"model": str(next(model.parameters()).device), "input": str(x.device),
                                                "output": str(logits.device), "dtype": str(logits.dtype)}
                    info.update(B_completed_images=pos, updated_at_singapore=e9.repeat.now())
                    write_json(OUTPUT / "run_info.json", info)
                    if pos % 25 == 0 or pos == len(records):
                        print(f"[E10-A B {pos}/{len(records)}] COMMITTED | CUDA FP32 | E0 cache unchanged", flush=True)
            del model
            torch.cuda.empty_cache()
            info.update(status="INFERRED", finished_inference_at_singapore=e9.repeat.now())
            write_json(OUTPUT / "run_info.json", info)
        except BaseException as error:
            info.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
            write_json(OUTPUT / "run_info.json", info)
            raise
    analyze()


def replay_b_item(record, path):
    saved = read_json(path)
    expected = read_json(PLAN)["models"]["B_expanded"]["sha256"]
    if saved["checkpoint_sha256"] != expected:
        raise ValueError("B prediction checkpoint differs")
    probability = safe_path(OUTPUT, saved["probability_path"])
    if file_hash(probability) != saved["probability_sha256"]:
        raise ValueError("Committed B prediction changed")
    p = load_probability(probability, (record["height"], record["width"]))
    return replay_row(record, p, saved), p


def bootstrap(rows, repetitions=BOOTSTRAP_REPETITIONS, seed=42):
    if not rows or repetitions < 1:
        raise ValueError("Bootstrap requires rows and positive repetitions")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row["dice_delta_B_minus_E0"])
    keys = sorted(grouped)
    sums = np.array([sum(grouped[k]) for k in keys])
    counts = np.array([len(grouped[k]) for k in keys])
    rng, values = np.random.default_rng(seed), []
    for start in range(0, repetitions, 1000):
        indices = rng.integers(0, len(keys), size=(min(1000, repetitions - start), len(keys)))
        values.extend((sums[indices].sum(axis=1) / counts[indices].sum(axis=1)).tolist())
    low, high = np.percentile(values, [2.5, 97.5])
    return {"group_count": len(keys), "image_count": len(rows), "mean_delta": float(np.mean([r["dice_delta_B_minus_E0"] for r in rows])),
            "ci95_low": float(low), "ci95_high": float(high), "repetitions": repetitions, "seed": seed,
            "interpretation": "Descriptive patient-cluster interval; no refitting or correction for repeated development/selection."}


def build_analysis(records, e0_rows, b_rows, agreements):
    expected = {r["sample_id"] for r in records}
    if len(expected) != len(records):
        raise ValueError("Duplicate manifest ID")
    indexed = [{r["sample_id"]: r for r in rows} for rows in (e0_rows, b_rows)]
    if any(len(rows) != len(records) or set(index) != expected for rows, index in zip((e0_rows, b_rows), indexed)):
        raise ValueError("Unpaired/incomplete/duplicate model results")
    paired = []
    for record in records:
        left, right = (index[record["sample_id"]] for index in indexed)
        paired.append({"sample_id": record["sample_id"], "group_id": record["group_id"],
                       "declared_group_id": record["declared_group_id"], "split": record["split"],
                       "gt_area_fraction": left["gt_area_fraction"], "e0_dice": left["dice"], "b_dice": right["dice"],
                       "dice_delta_B_minus_E0": right["dice"] - left["dice"], "e0_mae": left["mae"], "b_mae": right["mae"],
                       "mae_delta_B_minus_E0": right["mae"] - left["mae"], "mask_agreement_iou": agreements[record["sample_id"]],
                       "failure_category": descriptive.failure_category(left["dice"], right["dice"])})
    summaries, groups, pairs, intervals = [], [], {}, {}
    for split in ("ALL", "calibration", "selection"):
        selected_pairs = [r for r in paired if split == "ALL" or r["split"] == split]
        pairs[split] = descriptive.pair_summary(selected_pairs)
        intervals[split] = bootstrap(selected_pairs)
        for label, rows in (("E0_frozen_reference", e0_rows), ("B_expanded", b_rows)):
            subset = [r for r in rows if split == "ALL" or r["split"] == split]
            summaries.append({"model": label, "partition": split, **metric_summary(subset)})
            groups.extend({"model": label, "partition": split, **r} for r in summarize_groups(subset))
    patients = []
    for group in sorted({r["group_id"] for r in paired}):
        subset = [r for r in paired if r["group_id"] == group]
        patients.append({"group_id": group, "declared_group_id": subset[0]["declared_group_id"],
                         "split": subset[0]["split"], "count": len(subset),
                         **{k: float(np.mean([r[k] for r in subset])) for k in
                            ("e0_dice", "b_dice", "dice_delta_B_minus_E0", "e0_mae", "b_mae", "mae_delta_B_minus_E0")}})
    patient_summary = []
    for split in ("ALL", "calibration", "selection"):
        subset = [r for r in patients if split == "ALL" or r["split"] == split]
        patient_summary.append({"partition": split, "patient_groups": len(subset),
                                **{k: float(np.mean([r[k] for r in subset])) for k in
                                   ("e0_dice", "b_dice", "dice_delta_B_minus_E0", "e0_mae", "b_mae", "mae_delta_B_minus_E0")}})
    return paired, summaries, groups, pairs, intervals, patients, patient_summary


def analyze():
    plan, records = check()
    info = read_json(OUTPUT / "run_info.json")
    if info["status"] == "COMPLETED":
        verify()
        return
    if info["status"] != "INFERRED" or info["B_completed_images"] != 432 or info["plan_sha256"] != file_hash(PLAN):
        raise ValueError("Require all B predictions before analysis")
    e0_index, e0_rows, _ = e0_cache(records)
    b_rows, agreements = [], {}
    for record in records:
        row, b = replay_b_item(record, OUTPUT / "B_expanded/items" / (record["stem"] + ".json"))
        p0 = load_probability(safe_path(CACHE, e0_index[record["sample_id"]]["probability_path"]), b.shape)
        m0, mb = p0 >= .5, b >= .5
        union = int((m0 | mb).sum())
        agreements[record["sample_id"]] = float((m0 & mb).sum() / union) if union else 1.0
        b_rows.append(row)
    paired, summaries, groups, pairs, intervals, patients, patient_summary = build_analysis(records, e0_rows, b_rows, agreements)
    files = (("E0_replayed_per_image.csv", e0_rows), ("B_per_image.csv", b_rows), ("model_summary.csv", summaries),
             ("groups.csv", groups), ("paired_B_vs_E0.csv", paired), ("by_patient.csv", patients), ("patient_summary.csv", patient_summary))
    for name, values in files:
        write_csv(OUTPUT / name, values)
    write_json(OUTPUT / "complementarity.json", pairs)
    write_json(OUTPUT / "patient_bootstrap.json", intervals)
    pair = pairs["ALL"]
    b_rescues = pair["B_rescues_E0_low"] + pair["near_zero_rescued_by_B"] + pair["zero_rescued_by_B"]
    e0_rescues = pair["E0_rescues_B_low"] + pair["near_zero_new_in_B"] + pair["zero_new_in_B"]
    two_way = bool(b_rescues and e0_rescues)
    # These overlapping rescue counters are only tested for nonzero presence,
    # never summed/interpreted as a distinct rescued-case count.
    decision = {"stage": "E10_A_DEVELOPMENT_DIAGNOSIS", "status": "COMPLETED",
                "two_way_threshold_rescue_observed": two_way,
                "next_stage": "CONSIDER_SMALL_FIXED_WEIGHT_FUSION_DEVELOPMENT" if two_way else "LIMITED_THRESHOLD_COMPLEMENTARITY_REVIEW_BEFORE_FUSION",
                "B_checkpoint_remains_frozen": True, "fusion_candidates_evaluated": 0, "fusion_success_claim": False,
                "selection_partition_already_inspected_in_this_stage": True,
                "limitation": "Reused positive-only FAR; descriptive indicator, not a performance PASS or external confirmation."}
    write_json(OUTPUT / "decision.json", decision)
    lines = ["# E10-A：冻结B/E0在FAR上的比较与互补性", "", "日期：" + e9.repeat.now(), "",
             "原B seed42 round98与冻结E0；B新增FP32 CUDA推理，E0复用经逐图概率/指标重放核验的原E6缓存。",
             "没有训练、融合、校准拟合、权重/阈值搜索、检查点重选或C6重评估。", "",
             "FAR共432张阳性图像、187患者组；calibration160张/75组，selection272张/112组，患者交集0。",
             "全部FAR及两分区单模型结果在本阶段查看；它们均是复用开发数据，后续selection不能称为未见或盲测。", "",
             "## 图像等权主结果", "", "| 分区 | 模型 | Dice | IoU | MAE | Dice<0.5 | Dice<0.1 | 零Dice |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['partition']} | {r['model']} | {r['dice']:.6f} | {r['iou']:.6f} | {r['mae']:.6f} | {r['dice_lt05']} | {r['dice_lt01']} | {r['zero_dice']} |")
    lines += ["", "## 双向互补（GT事后诊断，不是可部署路由）", "",
              f"全量：B补救E0低Dice {pair['B_rescues_E0_low']}张；E0补救B低Dice {pair['E0_rescues_B_low']}张；两者共同低Dice {pair['both_lt05']}张；两者均>=0.5 {pair['both_ge05']}张。",
              f"B补救E0零Dice {pair['zero_rescued_by_B']}张；E0补救B零Dice {pair['zero_new_in_B']}张。",
              f"B相对E0的Dice胜/平/负：{pair['dice_wins_B']}/{pair['dice_ties']}/{pair['dice_losses_B']}。错误相关性仅描述：{pair['error_pearson_correlation_descriptive']}。", "",
              "完整分区互补、面积分层、患者均值与描述性患者成组bootstrap见同目录文件。bootstrap不重新拟合模型，不校正历史选择，不能据此宣称普遍显著优越。", "",
              "## 下一阶段指示", "", decision["next_stage"], "",
              "存在双向补救仅说明可以考虑小规模固定权重开发，不证明概率平均有效；没有自动启动融合。若继续需独立冻结新协议，B/E0均作为对照。",
              "单一复用来源、阳性病例、单B种子；不能推断无息肉帧误报率、新患者/中心普遍泛化。训练重叠审计只覆盖本项目登记数据与固定图像近重复启发式，不证明公开模型作者从未见过相关数据。"]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    info.update(status="COMPLETED", completed_at_singapore=e9.repeat.now(), probability_metric_replay="PASS",
                patient_groups=187, E0_cache_replayed_images=432)
    write_json(OUTPUT / "run_info.json", info)
    hashes = {p.relative_to(PROJECT).as_posix(): file_hash(p) for p in OUTPUT.rglob("*") if p.is_file()}
    write_json(OUTPUT / "audit.json", {"plan_sha256": file_hash(PLAN), "output_hashes": hashes})
    print("E10_A_COMPLETE", pair, decision, flush=True)


def verify():
    plan, records = check()
    audit = read_json(OUTPUT / "audit.json")
    if audit["plan_sha256"] != file_hash(PLAN) or read_json(OUTPUT / "run_info.json")["status"] != "COMPLETED":
        raise ValueError("E10 summary provenance differs")
    for name, expected in audit["output_hashes"].items():
        if file_hash(safe_path(PROJECT, name)) != expected:
            raise ValueError("E10 artifact changed: " + name)
    _, e0_rows, _ = e0_cache(records)
    b_rows = [replay_b_item(r, OUTPUT / "B_expanded/items" / (r["stem"] + ".json"))[0] for r in records]
    saved = {(r["model"], r["partition"]): r for r in read_csv(OUTPUT / "model_summary.csv")}
    for label, rows in (("E0_frozen_reference", e0_rows), ("B_expanded", b_rows)):
        for split in ("ALL", "calibration", "selection"):
            metrics = metric_summary([r for r in rows if split == "ALL" or r["split"] == split])
            for key, value in metrics.items():
                if not np.isclose(float(saved[label, split][key]), value, rtol=0, atol=1e-12):
                    raise ValueError("E10 recomputed summary differs")
    print("E10_A_VERIFIED; ALL_864_CACHED_PREDICTIONS_REPLAYED; NO_NEW_INFERENCE", flush=True)


if __name__ == "__main__":
    os.chdir(PROJECT)
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check", "infer", "analyze", "verify"))
    args = parser.parse_args()
    {"freeze": freeze, "check": check, "infer": infer, "analyze": analyze, "verify": verify}[args.command]()
