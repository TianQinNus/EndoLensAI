"""Targeted E8 development-error diagnosis. Never trains or reads holdout metrics."""
import math
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
sys.path.insert(0, str(PROJECT / "e8_repeat_windows"))
import run_e8_repeat as repeat
from common import read_json, write_json, write_csv, file_hash, signature, read_mask
from models import make_model
from engine import evaluate_records, require_cuda, set_seed

OUTPUT = PROJECT / "runs/e9_error_diagnosis"


def select_cases(indexed):
    a = indexed["A_baseline", 2024]
    b = indexed["B_expanded", 2024]
    new = {sid for sid, row in b.items() if row["dataset"] == "original_validation"
           and float(row["dice"]) < .5 <= float(a[sid]["dice"])}
    rescued = {sid for sid, row in b.items() if row["dataset"] == "original_validation"
               and float(a[sid]["dice"]) < .5 <= float(row["dice"])}
    common = set.intersection(*[{sid for sid, r in indexed["B_expanded", seed].items()
                                if float(r["dice"]) < .5} for seed in repeat.ALL_SEEDS])
    zeros = set.intersection(*[{sid for sid, r in indexed["B_expanded", seed].items()
                               if float(r["dice"]) == 0} for seed in repeat.ALL_SEEDS])
    return new, rescued, common, zeros


def error_type(row):
    tp, fp, fn = (int(row[k]) for k in ("tp", "fp", "fn"))
    if tp + fp == 0:
        return "empty_prediction"
    if tp == 0:
        return "disjoint_prediction"
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    if recall < .5 and precision >= .5:
        return "mainly_undersegmentation"
    if precision < .5 and recall >= .5:
        return "mainly_oversegmentation"
    if precision < .5 and recall < .5:
        return "mixed_low_precision_recall"
    return "other_overlap_error"


def overlay(rgb, mask, color):
    array = np.asarray(rgb, dtype=np.float32).copy()
    array[mask] = .55 * array[mask] + .45 * np.array(color)
    return Image.fromarray(array.astype(np.uint8))


def render_pages(records, rows_by_model):
    # Scientific diagnostic contact sheets, not generated/edited illustrative art.
    cell_w, cell_h, rows_per_page = 250, 220, 6
    for page in range(math.ceil(len(records) / rows_per_page)):
        subset = records[page * rows_per_page:(page + 1) * rows_per_page]
        sheet = Image.new("RGB", (cell_w * 6, cell_h * len(subset) + 35), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), "E8 validation diagnosis | GT green, prediction red | display only; metrics at native resolution", fill="black")
        for i, record in enumerate(subset):
            sid = record["sample_id"]
            with Image.open(PROJECT / record["image"]) as im:
                rgb = im.convert("RGB")
            gt = read_mask(PROJECT / record["mask"])
            cells = [("Image", rgb), ("GT", overlay(rgb, gt, (0, 255, 0)))]
            for label in ("A2024", "B42", "B17", "B2024"):
                row = rows_by_model[label].get(sid)
                if row is None:
                    cells.append((label + " not targeted", Image.new("RGB", rgb.size, "white")))
                else:
                    with Image.open(OUTPUT / label / row["mask_path"]) as mask_im:
                        mask = np.asarray(mask_im.convert("L")) > 127
                    cells.append((f"{label} Dice={float(row['dice']):.3f}", overlay(rgb, mask, (255, 0, 0))))
            y = 35 + cell_h * i
            draw.text((5, y), sid, fill="black")
            for j, (label, pic) in enumerate(cells):
                pic.thumbnail((cell_w - 10, cell_h - 50))
                sheet.paste(pic, (cell_w * j + 5, y + 40))
                draw.text((cell_w * j + 5, y + 21), label, fill="black")
        sheet.save(OUTPUT / f"contact_sheet_{page + 1:02d}.png")


def verify():
    audit = read_json(OUTPUT / "audit.json")
    if file_hash(Path(__file__)) != audit["diagnostic_code_sha256"]:
        raise ValueError("Diagnostic code changed")
    for kind in ("inputs", "outputs"):
        for path, expected in audit[kind].items():
            if file_hash(PROJECT / path) != expected:
                raise ValueError("Diagnostic artifact changed: " + path)
    repeat.verify()
    print("E9_DIAGNOSIS_VERIFIED; NO_NEW_INFERENCE")


def diagnose():
    if OUTPUT.exists():
        verify()
        return
    repeat.verify()
    _, bundle, manifests, protocol = repeat.check()
    indexed, inputs = {}, {}
    for arm, seed in [("A_baseline", 2024)] + [("B_expanded", s) for s in repeat.ALL_SEEDS]:
        path, _, rows, _ = repeat.verified_run(arm, seed, bundle, manifests, protocol)
        indexed[arm, seed] = {r["sample_id"]: r for r in rows}
        for name in ("checkpoints/best.pth", "validation_best_per_image.csv"):
            file = path / name
            inputs[file.relative_to(PROJECT).as_posix()] = file_hash(file)
    new, rescued, common, zeros = select_cases(indexed)
    ordered = sorted(new) + sorted(rescued) + sorted(common - new - rescued)
    by_id = {r["sample_id"]: r for r in manifests["validation"]}
    # Original E8 records have no source_stem; the shared output writer needs it.
    # Add an output-only identifier without changing any input/label/metric.
    records = [{**by_id[sid], "source_stem": by_id[sid].get("source_stem", Path(by_id[sid]["image"]).stem)}
               for sid in ordered]
    # Verify shape/content without changing labels or preprocessing.
    shape_issues = []
    for record in records:
        with Image.open(PROJECT / record["image"]) as im:
            rgb_shape = (im.height, im.width)
        gt = read_mask(PROJECT / record["mask"])
        if gt.shape != rgb_shape or not gt.any():
            shape_issues.append(record["sample_id"])
    if shape_issues:
        raise ValueError("Targeted shape/empty GT anomaly: " + str(shape_issues))
    device = require_cuda()
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT / "run_info.json", {"status": "RUNNING", "role": "development-only diagnosis",
               "selection_sha256": signature(ordered), "C6_evaluations": 0, "training_runs": 0})
    rows_by_model = {}
    try:
        for label, arm, seed in [("A2024", "A_baseline", 2024), ("B42", "B_expanded", 42),
                                 ("B17", "B_expanded", 17), ("B2024", "B_expanded", 2024)]:
            # Match deterministic CUDA/TF32 settings used by formal E8 validation.
            set_seed(seed, device)
            targeted = [r for r in records if r["sample_id"] in new | rescued] if label == "A2024" else records
            cp = torch.load(repeat.e8.run_path(arm, seed) / "checkpoints/best.pth", map_location="cpu", weights_only=True)
            model = make_model(cp["config"], initialization=False)
            model.load_state_dict(cp["model_state"], strict=True)
            model.to(device).eval()
            _, rows = evaluate_records(model, PROJECT, targeted, device, 352, OUTPUT / label)
            for row in rows:
                old = indexed[arm, seed][row["sample_id"]]
                for key in ("dice", "iou", "mae"):
                    if not np.isclose(float(row[key]), float(old[key]), atol=1e-7, rtol=0):
                        raise ValueError("Diagnostic inference differs from saved validation: " + row["sample_id"])
            rows_by_model[label] = {r["sample_id"]: r for r in rows}
            del model, cp
            torch.cuda.empty_cache()
        case_rows = []
        for record in records:
            sid = record["sample_id"]
            row = {"sample_id": sid, "dataset": record["dataset"],
                   "gt_area_fraction": rows_by_model["B2024"][sid]["gt_area_fraction"],
                   "new_old_failure_2024": sid in new, "rescued_old_2024": sid in rescued,
                   "persistent_low_all_seeds": sid in common, "persistent_zero_all_seeds": sid in zeros}
            for label, values in rows_by_model.items():
                value = values.get(sid)
                row[label + "_dice"] = value["dice"] if value else ""
                row[label + "_error_type"] = error_type(value) if value else ""
            case_rows.append(row)
        write_csv(OUTPUT / "cases.csv", case_rows)
        render_pages(records, rows_by_model)
        from collections import Counter
        counts = {label: dict(Counter(error_type(r) for sid, r in values.items() if sid in common))
                  for label, values in rows_by_model.items() if label.startswith("B")}
        result = {"status": "COMPLETED", "role": "development-only diagnosis", "new_old_failures": len(new),
                  "rescued_old": len(rescued), "persistent_low": len(common), "persistent_zero": len(zeros),
                  "targeted_unique_cases": len(records), "targeted_shape_checks_pass": True,
                  "persistent_low_error_types": counts, "saved_validation_metrics_reproduced": True,
                  "C6_evaluations": 0, "E0_inferences": 0, "training_runs": 0,
                  "limitations": "Selected failure cases, not a new performance test. No label correction or validation-to-training transfer."}
        write_json(OUTPUT / "run_info.json", result)
        outputs = {p.relative_to(PROJECT).as_posix(): file_hash(p) for p in OUTPUT.rglob("*") if p.is_file()}
        write_json(OUTPUT / "audit.json", {"diagnostic_code_sha256": file_hash(Path(__file__)),
                                          "inputs": inputs, "outputs": outputs})
        print("E9_TARGETED_DIAGNOSIS_COMPLETE", result, flush=True)
    except BaseException as error:
        write_json(OUTPUT / "run_info.json", {"status": "FAILED", "error": str(error),
                   "note": "Partial outputs preserved. Request recovery guidance; do not delete or treat as complete."})
        raise


if __name__ == "__main__":
    torch.set_num_threads(4)
    diagnose()
