"""E8-H: verify cached locked evaluation and describe B/E0 complementarity.

No training, new inference, fusion weights, threshold search, or model selection.
Freeze this supplemental descriptive-analysis plan BEFORE C6 inference.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image
from common import (PROJECT, ARMS, read_json, write_json, file_hash, write_csv,
                    signature, read_mask, metric_summary, summarize_groups)
from e0_protocol import binary_metrics
from run_e8 import preflight

PLAN = PROJECT / "runs/e8_holdout_analysis_plan.json"
RESULTS = PROJECT / "runs/e8_holdout_seed42"
OUTPUT = PROJECT / "runs/e8_holdout_analysis_seed42"
E0 = "E0_frozen_reference"
B = "B_expanded"
MODEL_NAMES = (*ARMS, E0)


def timestamp():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def freeze():
    if PLAN.exists():
        raise FileExistsError("Analysis plan exists; preserve it")
    if RESULTS.exists():
        raise ValueError("Descriptive plan must be frozen before any holdout inference outputs exist")
    lock_path = PROJECT / "runs/e8_holdout_lock_seed42.json"
    lock = read_json(lock_path)
    if signature({k: v for k, v in lock.items() if k != "lock_signature"}) != lock["lock_signature"]:
        raise ValueError("Invalid model lock")
    plan = {"stage": "E8_HOLDOUT_DESCRIPTIVE_COMPLEMENTARITY", "frozen_at_singapore": timestamp(),
            "analysis_script_sha256": file_hash(Path(__file__)), "lock_sha256": file_hash(lock_path),
            "binary_mask_threshold": .5, "low_dice_cutoff": .5, "near_zero_dice_cutoff": .1,
            "practical_dice_difference": .01, "tie_tolerance": 1e-12,
            "pair": [E0, B], "selected_model_remains": lock["selected_arm_from_validation"],
            "policy": "Replay cached probabilities and masks; descriptive paired comparison of B/E0. No fusion evaluated, no holdout model reselection, no patient-level inference/CI.",
            "diagnostics": ["All five model metrics and GT size strata", "B-E0 per-image Dice/MAE and mask agreement IoU",
                            "B-E0 low/near-zero/zero failure transitions", "Both good / only B good / only E0 good / both low",
                            "Error Pearson correlation, descriptive only", "Data expansion and hard sampling prespecified contrasts"]}
    plan["plan_signature"] = signature(plan)
    write_json(PLAN, plan)
    print("DESCRIPTIVE_ANALYSIS_PLAN_FROZEN", PLAN)


def failure_category(e0_dice, b_dice, cutoff=.5):
    if e0_dice >= cutoff and b_dice >= cutoff:
        return "both_ge05"
    if e0_dice < cutoff <= b_dice:
        return "B_rescues_E0_low"
    if b_dice < cutoff <= e0_dice:
        return "E0_rescues_B_low"
    return "both_lt05"


def pair_summary(rows):
    if not rows:
        raise ValueError("Empty paired comparison")
    e0 = np.array([r["e0_dice"] for r in rows], dtype=np.float64)
    b = np.array([r["b_dice"] for r in rows], dtype=np.float64)
    delta = b - e0
    mae_delta = np.array([r["b_mae"] - r["e0_mae"] for r in rows])
    correlation = float(np.corrcoef(1-e0, 1-b)[0, 1]) if e0.std() > 0 and b.std() > 0 else None
    return {"count": len(rows), "dice_delta_B_minus_E0": float(delta.mean()),
            "mae_delta_B_minus_E0": float(mae_delta.mean()),
            "dice_wins_B": int((delta > 1e-12).sum()), "dice_ties": int((np.abs(delta) <= 1e-12).sum()),
            "dice_losses_B": int((delta < -1e-12).sum()),
            "practical_dice_wins_B_gt001": int((delta > .01).sum()),
            "practical_dice_losses_B_lt_minus001": int((delta < -.01).sum()),
            "both_ge05": int(((e0 >= .5) & (b >= .5)).sum()),
            "B_rescues_E0_low": int(((e0 < .5) & (b >= .5)).sum()),
            "E0_rescues_B_low": int(((b < .5) & (e0 >= .5)).sum()),
            "both_lt05": int(((e0 < .5) & (b < .5)).sum()),
            "near_zero_rescued_by_B": int(((e0 < .1) & (b >= .1)).sum()),
            "near_zero_new_in_B": int(((b < .1) & (e0 >= .1)).sum()),
            "zero_rescued_by_B": int(((e0 == 0) & (b > 0)).sum()),
            "zero_new_in_B": int(((b == 0) & (e0 > 0)).sum()),
            "B_dice_up_mae_not_worse": int(((delta > 1e-12) & (mae_delta <= 0)).sum()),
            "B_dice_up_mae_worse": int(((delta > 1e-12) & (mae_delta > 0)).sum()),
            "error_pearson_correlation_descriptive": correlation}


def paired_contrast(left_rows, right_rows, label):
    left = {r["sample_id"]: r for r in left_rows}
    right = {r["sample_id"]: r for r in right_rows}
    if set(left) != set(right) or len(left) != len(left_rows) or len(right) != len(right_rows):
        raise ValueError("Paired membership differs")
    return {"contrast": label, "count": len(left),
            "dice_delta": float(np.mean([right[k]["dice"]-left[k]["dice"] for k in left])),
            "mae_delta": float(np.mean([right[k]["mae"]-left[k]["mae"] for k in left])),
            "low05_delta": sum(r["dice"] < .5 for r in right_rows)-sum(r["dice"] < .5 for r in left_rows),
            "zero_delta": sum(r["dice"] == 0 for r in right_rows)-sum(r["dice"] == 0 for r in left_rows)}


def cached_probability(path, shape):
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["probability"]:
            raise ValueError("Unexpected cached probability fields")
        p = archive["probability"]
    if p.dtype != np.float32 or p.shape != shape or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("Invalid cached probability")
    return p


def verify_and_replay(manifests, lock):
    status = read_json(RESULTS / "run_info.json")
    if status["status"] != "COMPLETED" or status["completed_models"] != list(lock["models"]):
        raise ValueError("Require completed five-model locked evaluation")
    if status["lock_sha256"] != file_hash(PROJECT / "runs/e8_holdout_lock_seed42.json"):
        raise ValueError("Evaluation lock mismatch")
    records = {r["sample_id"]: r for r in manifests["holdout"]}
    all_rows, summaries, artifact_hashes = {}, [], {}
    for name in MODEL_NAMES:
        folder = RESULTS / name
        with (folder / "per_image.csv").open(encoding="utf-8-sig", newline="") as handle:
            saved_rows = list(csv.DictReader(handle))
        if len(saved_rows) != len(records) or {r["sample_id"] for r in saved_rows} != set(records):
            raise ValueError("Incomplete or duplicate cached rows: " + name)
        saved_metrics = read_json(folder / "metrics.json")
        if saved_metrics["checkpoint_sha256"] != lock["models"][name]["sha256"]:
            raise ValueError("Model metrics checkpoint mismatch")
        replay = []
        for saved in saved_rows:
            record = records[saved["sample_id"]]
            gt = read_mask(PROJECT / record["mask"])
            probability_path = (folder / saved["probability_path"]).resolve()
            mask_path = (folder / saved["mask_path"]).resolve()
            if not probability_path.is_relative_to(folder.resolve()) or not mask_path.is_relative_to(folder.resolve()):
                raise ValueError("Cached path escapes model result directory")
            p = cached_probability(probability_path, gt.shape)
            prediction = p >= .5
            if not np.array_equal(read_mask(mask_path), prediction):
                raise ValueError("Saved binary mask differs from thresholded probability")
            metrics = binary_metrics(prediction, gt)
            mae = float(np.abs(p-gt.astype(np.float32)).mean())
            if int(saved["height"]) != gt.shape[0] or int(saved["width"]) != gt.shape[1] or saved["dataset"] != record["dataset"]:
                raise ValueError("Saved image metadata mismatch")
            for key, value in {**metrics, "mae": mae, "gt_area_fraction": float(gt.mean())}.items():
                if not np.isclose(float(saved[key]), value, rtol=0, atol=1e-12):
                    raise ValueError("Cached metric replay mismatch: " + name + "/" + key)
            replay.append({"sample_id": record["sample_id"], "dataset": record["dataset"], **metrics,
                           "mae": mae, "gt_area_fraction": float(gt.mean()),
                           "probability_path": saved["probability_path"], "mask_path": saved["mask_path"]})
        summary = metric_summary(replay)
        for key, value in summary.items():
            if not np.isclose(value, saved_metrics["summary"][key], rtol=0, atol=1e-12):
                raise ValueError("Model summary replay mismatch")
        all_rows[name] = replay
        summaries.append({"model": name, **summary})
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                artifact_hashes[path.relative_to(PROJECT).as_posix()] = file_hash(path)
        print("CACHED_REPLAY_VERIFIED", name, len(replay), flush=True)
    for file in (RESULTS / "run_info.json", RESULTS / "summary.csv", RESULTS / "contrasts_vs_A.csv"):
        artifact_hashes[file.relative_to(PROJECT).as_posix()] = file_hash(file)
    with (RESULTS / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        saved = {r["model"]: r for r in csv.DictReader(handle)}
    if set(saved) != set(MODEL_NAMES):
        raise ValueError("Five-model result summary membership mismatch")
    for row in summaries:
        for key, value in row.items():
            if key != "model" and not np.isclose(float(saved[row["model"]][key]), value, rtol=0, atol=1e-12):
                raise ValueError("Top-level summary mismatch")
    return all_rows, summaries, artifact_hashes


def analyze():
    if OUTPUT.exists():
        raise FileExistsError("Analysis exists; run verify instead of overwriting")
    plan = read_json(PLAN)
    if signature({k: v for k, v in plan.items() if k != "plan_signature"}) != plan["plan_signature"]:
        raise ValueError("Analysis plan changed")
    if file_hash(Path(__file__)) != plan["analysis_script_sha256"]:
        raise ValueError("Analysis source changed after descriptive plan freeze")
    lock_path = PROJECT / "runs/e8_holdout_lock_seed42.json"
    if file_hash(lock_path) != plan["lock_sha256"]:
        raise ValueError("Model lock changed")
    info, manifests, protocol = preflight()
    lock = read_json(lock_path)
    if lock["protocol_signature"] != protocol["protocol_signature"] or lock["bundle_signature"] != info["bundle_signature"]:
        raise ValueError("Model lock protocol/data mismatch")
    for descriptor in lock["models"].values():
        if file_hash(PROJECT / descriptor["checkpoint"]) != descriptor["sha256"]:
            raise ValueError("Frozen checkpoint changed")
    all_rows, summaries, artifact_hashes = verify_and_replay(manifests, lock)
    e0, b = ({r["sample_id"]: r for r in all_rows[name]} for name in (E0, B))
    paired = []
    for key in sorted(e0):
        left, right = e0[key], b[key]
        left_mask = read_mask(RESULTS / E0 / left["mask_path"])
        right_mask = read_mask(RESULTS / B / right["mask_path"])
        union = int((left_mask | right_mask).sum())
        agreement = float((left_mask & right_mask).sum()/union) if union else 1.0
        paired.append({"sample_id": key, "gt_area_fraction": left["gt_area_fraction"],
                       "e0_dice": left["dice"], "b_dice": right["dice"], "dice_delta_B_minus_E0": right["dice"]-left["dice"],
                       "e0_mae": left["mae"], "b_mae": right["mae"], "mae_delta_B_minus_E0": right["mae"]-left["mae"],
                       "e0_predicted_pixels": left["tp"]+left["fp"], "b_predicted_pixels": right["tp"]+right["fp"],
                       "mask_agreement_iou": agreement, "failure_category": failure_category(left["dice"], right["dice"])})
    pair = pair_summary(paired)
    contrasts = [paired_contrast(all_rows[left], all_rows[right], f"{right}-{left}") for left, right in
                 (("A_baseline", B), ("A_baseline", "C_hard"), (B, "D_expanded_hard"), ("C_hard", "D_expanded_hard"), (E0, B))]
    OUTPUT.mkdir()
    write_csv(OUTPUT / "model_summary.csv", summaries)
    write_csv(OUTPUT / "paired_B_vs_E0.csv", paired)
    write_csv(OUTPUT / "contrasts.csv", contrasts)
    write_csv(OUTPUT / "groups.csv", [{"model": name, **group} for name in MODEL_NAMES for group in summarize_groups(all_rows[name])])
    verification = {"status": "PASS", "verified_at_singapore": timestamp(), "models": list(MODEL_NAMES),
                    "samples_each": len(manifests["holdout"]), "plan_sha256": file_hash(PLAN),
                    "lock_sha256": file_hash(lock_path), "source_sha256": file_hash(Path(__file__)),
                    "result_artifacts_sha256": artifact_hashes, "probability_metric_replay": "all five models, all samples, atol=1e-12",
                    "new_model_inferences": 0, "fusion_candidates_evaluated": 0, "model_reselection": False}
    write_json(OUTPUT / "verification.json", verification)
    write_json(OUTPUT / "complementarity.json", pair)
    date = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d")
    report = [f"# E8-H：冻结 C6 留出评估与 B/E0 互补性诊断\n\n日期：{date}（Asia/Singapore）。",
              "## 1. 范围与协议\n\n沿用 E8 冻结协议，在 C6 的83张单帧阳性图像上评估四组 seed42 开发 best 与原始 E0。B 仍为验证集选出的主方案，不根据 C6 重选模型。没有训练或融合权重搜索。",
              f"模型配置锁定于 {lock['created_utc']}；补充描述性分析计划冻结于 {plan['frozen_at_singapore']}，早于首次 C6 推理。",
              "二值阈值0.5、原尺寸 logits 插值后 sigmoid、ImageNet RGB归一化，与既有评估协议一致。原始模型、冻结配置与图像/掩膜通过校验；五模型全部83张缓存概率、PNG与逐图指标重放一致。",
              "## 2. 同数据模型结果\n\n| 模型 | Dice | IoU | MAE | Dice<0.5 | Dice<0.1 | 零Dice |\n|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        report.append(f"| {r['model']} | {r['dice']:.6f} | {r['iou']:.6f} | {r['mae']:.6f} | {r['dice_lt05']} | {r['dice_lt01']} | {r['zero_dice']} |")
    report += ["\n## 3. 预声明对照\n\n| 对照 | Dice增量 | MAE增量 | 低Dice数量增量 | 零Dice数量增量 |\n|---|---:|---:|---:|---:|"]
    for r in contrasts:
        report.append(f"| {r['contrast']} | {r['dice_delta']:+.6f} | {r['mae_delta']:+.6f} | {r['low05_delta']:+d} | {r['zero_delta']:+d} |")
    report += ["\n## 4. B与E0的逐图互补性\n\n| E0状态 / B状态 | B Dice≥0.5 | B Dice<0.5 |\n|---|---:|---:|",
               f"| E0 Dice≥0.5 | {pair['both_ge05']} | {pair['E0_rescues_B_low']} |",
               f"| E0 Dice<0.5 | {pair['B_rescues_E0_low']} | {pair['both_lt05']} |",
               f"\nB 的 Dice 胜/平/负为 {pair['dice_wins_B']}/{pair['dice_ties']}/{pair['dice_losses_B']}；绝对差超过0.01的明显胜/负为 {pair['practical_dice_wins_B_gt001']}/{pair['practical_dice_losses_B_lt_minus001']}。",
               f"B补救E0零Dice {pair['zero_rescued_by_B']}张，同时新增零Dice {pair['zero_new_in_B']}张。B在Dice提高且MAE不增加的图像有 {pair['B_dice_up_mae_not_worse']}张；Dice提高但MAE增加的有 {pair['B_dice_up_mae_worse']}张。",
               "这是使用真实掩膜的事后错误诊断，不是可部署路由规则；不能据此给推理阶段提供真实Dice或GT面积。模型错误相关性仅描述这83张图像，不是患者级统计推断。",
               "## 5. 小目标与其他面积分层\n\n| 模型 | 面积分层 | 数量 | Dice | MAE | Dice<0.5 | 零Dice |\n|---|---|---:|---:|---:|---:|---:|"]
    for name in (E0, B):
        for r in summarize_groups(all_rows[name]):
            if r["group"] in ("small_lt1pct", "medium_1to10pct", "large_ge10pct"):
                report.append(f"| {name} | {r['group']} | {r['count']} | {r['dice']:.6f} | {r['mae']:.6f} | {r['dice_lt05']} | {r['zero_dice']} |")
    report += ["\n## 6. 解释边界与下一阶段\n\n这是单个新中心、83张阳性图像、seed42的有限范围证据；没有可靠患者ID，不提供患者级置信区间或显著性宣称。没有阴性帧，不能据此判断完整临床误报率。",
               "C6已完成模型评估，从此不能称为新融合实验的未见盲测集。C6上的互补病例只提供开发假设，未证明固定权重融合有效，也不能据此选择权重、阈值或自动路由。",
               "后续应先依据本轮结果选择研究主线；若研究融合，另行声明开发数据与未用于调参的评估数据；若研究U-Net稳定性，按预声明种子作匹配对照。不修改已完成的E8模型选择、协议或留出结果。",
               "详细逐图差值与类别见 paired_B_vs_E0.csv；模型概率文件保留在原评估目录。验证记录见 verification.json。"]
    (OUTPUT / "report.md").write_text("\n\n".join(report)+"\n", encoding="utf-8")
    hashes = {p.name: file_hash(p) for p in OUTPUT.iterdir() if p.is_file()}
    write_json(OUTPUT / "analysis_artifacts.json", hashes)
    print("COMPLEMENTARITY", pair, flush=True)
    print("HOLDOUT_ANALYSIS_COMPLETE", OUTPUT)


def verify():
    info = read_json(OUTPUT / "verification.json")
    for name, expected in info["result_artifacts_sha256"].items():
        if file_hash(PROJECT / name) != expected:
            raise ValueError("Cached evaluation artifact changed: " + name)
    for name, expected in read_json(OUTPUT / "analysis_artifacts.json").items():
        if file_hash(OUTPUT / name) != expected:
            raise ValueError("Analysis artifact changed: " + name)
    if file_hash(PLAN) != info["plan_sha256"] or file_hash(PROJECT / "runs/e8_holdout_lock_seed42.json") != info["lock_sha256"]:
        raise ValueError("Frozen plan/lock changed")
    if file_hash(Path(__file__)) != info["source_sha256"]:
        raise ValueError("Analysis source changed")
    preflight()
    print("HOLDOUT_AND_ANALYSIS_ARTIFACTS_UNCHANGED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "analyze", "verify"))
    args = parser.parse_args()
    {"freeze": freeze, "analyze": analyze, "verify": verify}[args.command]()
