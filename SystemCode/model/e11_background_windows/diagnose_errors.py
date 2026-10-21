"""Cached FAR error plates, for visual review only. No inference or training data export."""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "e10d_windows"))
import run_e10d as d
from common import read_json, write_json, file_hash, write_csv, read_mask

OUTPUT = PROJECT / "runs/e11_cached_error_review"


def band_diagnosis(pred, gt):
    def resized(a):
        return np.asarray(Image.fromarray(a.astype(np.uint8)*255).resize((352, 352), Image.Resampling.NEAREST)) > 127
    p, g = resized(pred), resized(gt)
    dilated = F.max_pool2d(torch.from_numpy(g.astype(np.float32))[None, None], 11, stride=1, padding=5)[0, 0].numpy() > 0
    fp, far = p & ~g, p & ~dilated
    return {"fp_pixels_at352": int(fp.sum()), "far_fp_pixels_at352": int(far.sum()),
            "far_fp_fraction_at352": float(far.sum()/fp.sum()) if fp.any() else 0.}


def overlay(rgb, masks_colors):
    a = np.asarray(rgb, np.float32).copy()
    for mask, color in masks_colors:
        a[mask] = .55*a[mask] + .45*np.array(color, np.float32)
    return Image.fromarray(np.rint(a).clip(0, 255).astype(np.uint8))


def plate(rgb, pred, gt, row):
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    title = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 21)
    panels = (rgb.copy(), overlay(rgb, [(gt, (30, 220, 80))]), overlay(rgb, [(pred, (255, 70, 50))]),
              overlay(rgb, [(pred & gt, (30, 220, 80)), (pred & ~gt, (255, 65, 45)), (~pred & gt, (55, 135, 255))]))
    labels = ("Original", "Ground truth (green)", "B prediction (red)", "TP green / FP red / FN blue")
    result = Image.new("RGB", (1940, 470), (20, 24, 30))
    draw = ImageDraw.Draw(result)
    draw.text((15, 8), row["stem"], font=title, fill="white")
    for pos, (im, label) in enumerate(zip(panels, labels)):
        tile = ImageOps.contain(im, (470, 350), Image.Resampling.BILINEAR)
        x = 15 + pos*485
        draw.text((x, 43), label, font=font, fill="white")
        result.paste(tile, (x+(470-tile.width)//2, 73+(350-tile.height)//2))
    draw.text((15, 429), f"Dice={row['dice']:.4f}  Precision={row['precision']:.4f}  Recall={row['recall']:.4f}  TP={row['tp']}  FP={row['fp']}  FN={row['fn']}  Far-FP share at352={row['far_fp_fraction_at352']:.3f}", font=small, fill="white")
    draw.text((15, 450), "Cached research diagnosis only. GT band5px at352 is descriptive; not a deployment rule or training input.", font=small, fill=(185, 190, 200))
    return result


def sources():
    _, _, folded = d.c.check()
    # Verify the completed diagnostic artifacts; no re-fitting, new inference or new candidate.
    d.verify_audit(d.OUTPUT / "diagnosis")
    records = {r["sample_id"]: r for r in folded}
    rows = [r for r in d.c.b.a.read_csv(d.OUTPUT / "diagnosis/per_image.csv") if r["model"] == d.c.b.B and float(r["dice"]) < .5]
    if len(rows) != 30 or len({r["sample_id"] for r in rows}) != 30:
        raise ValueError("Require exactly the original30 calibration low-Dice images")
    return records, sorted(rows, key=lambda r: (float(r["dice"]), r["sample_id"]))


def build():
    records, selected = sources()
    if OUTPUT.exists():
        verify()
        print("EXISTING_REVIEW_PRESERVED")
        return
    OUTPUT.mkdir(exist_ok=False)
    index, rows, inputs = d.c.index_e0(), [], {}
    for pos, old in enumerate(selected, 1):
        r = records[old["sample_id"]]
        pb, _, gt = d.c.b.load_pair(r, index)
        metrics = d.c.b.a.replay_row(r, pb)
        if any(not np.isclose(float(old[k]), metrics[k], rtol=0, atol=1e-12) for k in ("dice", "iou", "mae", "tp", "fp", "fn")):
            raise ValueError("Original diagnostic metrics changed")
        pred = pb >= .5
        with Image.open(PROJECT / r["image"]) as im:
            rgb = im.convert("RGB")
        row = {"sample_id": r["sample_id"], "stem": r["stem"], "group_id": r["group_id"],
               **{k: metrics[k] for k in ("dice", "tp", "fp", "fn")},
               "precision": metrics["tp"]/(metrics["tp"]+metrics["fp"]) if metrics["tp"]+metrics["fp"] else 0.,
               "recall": metrics["tp"]/(metrics["tp"]+metrics["fn"]), **band_diagnosis(pred, gt),
               "plate": f"{pos:02d}_{r['stem']}.png"}
        plate(rgb, pred, gt, row).save(OUTPUT / row["plate"])
        rows.append(row)
        for key in ("image", "mask"):
            inputs[r[key]] = file_hash(PROJECT / r[key])
        item_path = d.c.b.a.OUTPUT / "B_expanded/items" / (r["stem"]+".json")
        item = read_json(item_path)
        for path in (item_path, d.c.b.a.safe_path(d.c.b.a.OUTPUT, item["probability_path"])):
            inputs[path.relative_to(PROJECT).as_posix()] = file_hash(path)
    write_csv(OUTPUT / "spatial_diagnosis.csv", rows)
    summary = {"low_images": 30, "far_fp_majority_images": sum(r["far_fp_fraction_at352"] >= .5 for r in rows),
               "pooled_far_fp_fraction": sum(r["far_fp_pixels_at352"] for r in rows)/sum(r["fp_pixels_at352"] for r in rows),
               "definition": "Nearest-resized binary masks at352; GT dilation Chebyshev radius5. Descriptive only; does not identify folds/highlights or separate connected overgrowth from remote components.",
               "new_model_forwards": 0, "training_exports": 0}
    write_json(OUTPUT / "summary.json", summary)
    lines = ["# E11 训练前错误区域复核", "", "仅已缓存FAR calibration30张低Dice图，未重新推理。", "",
             "颜色：第四栏绿色=TP，红色=FP，蓝色=FN；第二栏是原始标注，不是预测。", "",
             f"远于GT外侧5像素（352尺度）的FP占多数：{summary['far_fp_majority_images']}/30；池化比例约{summary['pooled_far_fp_fraction']:.1%}。",
             "该尺度只是诊断，不作为训练规则。不能单凭红色区域判定皱襞/高光，也不能把这些FAR区域加入训练。", "",
             "查看时注意：标注是否完整、息肉是否被找对、FP是连接外扩还是其他位置、是否有明显成像伪影。", ""]
    for r in rows:
        lines += [f"## {r['stem']}", "", f"Dice={r['dice']:.4f}；precision={r['precision']:.4f}；recall={r['recall']:.4f}。", "",
                  "![原图/标注/B预测/错误区域]("+(OUTPUT / r["plate"]).as_posix()+")", ""]
    (OUTPUT / "index.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(OUTPUT / "audit.json", {"script_sha256": file_hash(Path(__file__)), "inputs": inputs,
               "outputs": {p.name: file_hash(p) for p in OUTPUT.iterdir() if p.is_file()}})
    print("E11_CACHED_ERROR_REVIEW_COMPLETE", summary, flush=True)


def verify():
    audit = read_json(OUTPUT / "audit.json")
    if audit["script_sha256"] != file_hash(Path(__file__)):
        raise ValueError("Review source changed")
    for name, expected in audit["inputs"].items():
        if file_hash(PROJECT / name) != expected:
            raise ValueError("Review input changed")
    for name, expected in audit["outputs"].items():
        if file_hash(OUTPUT / name) != expected:
            raise ValueError("Review artifact changed")
    print("E11_CACHED_ERROR_REVIEW_VERIFIED; NO_TRAINING_EXPORTS", flush=True)


if __name__ == "__main__":
    os.chdir(PROJECT)
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    {"build": build, "verify": verify}[args.command]()
