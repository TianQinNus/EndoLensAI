"""E5: predeclared equal probability fusion of frozen E0 and E4 outputs.

The experiment compares the new E5 output with E0, E4, and the completed
E3/V3 (E0+E2) result.  It does not train a model, run model inference, or
search fusion weights/thresholds.
"""

import argparse
import csv
import hashlib
import html
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DATASETS = (
    "Kvasir",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "ETIS-LaribPolypDB",
    "CVC-300",
)
COUNTS = dict(zip(DATASETS, (100, 62, 380, 196, 60)))
WEIGHT_E0 = 0.5
WEIGHT_E4 = 0.5
THRESHOLD = 0.5


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, data):
    if not data:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def under(root, relative):
    """Resolve a stored POSIX path without allowing it to escape root."""
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Expected relative POSIX path: " + str(relative))
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Unsafe relative path: " + relative)
    return path


def by_id(data):
    result = {}
    for row in data:
        sample_id = row["sample_id"]
        if sample_id in result:
            raise ValueError("Duplicate sample_id: " + sample_id)
        result[sample_id] = row
    return result


def load_gt(path):
    with Image.open(path) as image:
        array = np.array(image.convert("L"), dtype=np.uint8)
    return array > (0 if array.max() <= 1 else 127.5)


def load_probability(path, shape):
    with np.load(path, allow_pickle=False) as archive:
        probability = archive["probability"]
    if (
        probability.dtype != np.float32
        or probability.shape != shape
        or not np.isfinite(probability).all()
    ):
        raise ValueError("Invalid probability dtype/shape/nonfinite value: " + str(path))
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Probability outside [0,1]: " + str(path))
    return probability


def metrics(probability, ground_truth):
    prediction = probability >= THRESHOLD
    true_positive = int(np.count_nonzero(prediction & ground_truth))
    false_positive = int(np.count_nonzero(prediction & ~ground_truth))
    false_negative = int(np.count_nonzero(~prediction & ground_truth))
    dice_denominator = 2 * true_positive + false_positive + false_negative
    union = true_positive + false_positive + false_negative
    return {
        "dice": 2 * true_positive / dice_denominator if dice_denominator else 1.0,
        "iou": true_positive / union if union else 1.0,
        "mae": float(
            np.abs(probability - ground_truth.astype(np.float32)).mean()
        ),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def fuse(e0_probability, e4_probability):
    if e0_probability.shape != e4_probability.shape:
        raise ValueError("Fusion shape mismatch")
    return (
        np.float32(WEIGHT_E0) * e0_probability
        + np.float32(WEIGHT_E4) * e4_probability
    )


def close(actual, reference, name, tolerance=1e-12):
    if abs(float(actual) - float(reference)) > tolerance:
        raise ValueError("Reference metric mismatch: " + name)


def validate_sources(project, e0_run, e4_run, e4_evaluation, e3_run):
    e0_info = read_json(e0_run / "run_info.json")
    e4_info = read_json(e4_run / "run_info.json")
    e4_stage = read_json(e4_run / "e4_stage.json")
    e4_eval_info = read_json(e4_evaluation / "run_info.json")
    e3_info = read_json(e3_run / "run_info.json")

    if (
        e0_info.get("status") != "COMPLETED"
        or e0_info.get("run_kind") != "FULL_E0"
        or e0_info.get("processed_count") != 798
    ):
        raise ValueError("E0 must be completed FULL_E0 with 798 entries")
    if (
        e0_info.get("protocol", {}).get("id")
        != "endolens-e0-probability-v1"
        or e0_info.get("threshold") != THRESHOLD
    ):
        raise ValueError("Wrong E0 protocol; do not use legacy_minmax outputs")

    if (
        e4_info.get("status") != "TRAINED"
        or e4_info.get("run_kind") != "E2_FINAL"
        or e4_info.get("config", {}).get("augmentation") != "strong"
        or e4_info.get("completed_epochs") != 76
        or e4_stage.get("stage") != "E4_FINAL"
    ):
        raise ValueError("E4 must be the completed 76-epoch strong final model")
    if (
        e4_eval_info.get("status") != "COMPLETED"
        or e4_eval_info.get("processed_count") != 798
    ):
        raise ValueError("E4 evaluation must be completed with 798 entries")
    e4_protocol = e4_eval_info.get("protocol", {})
    if (
        e4_protocol.get("id") != "endolens-e2-probability-v1"
        or e4_protocol.get("threshold") != THRESHOLD
        or e4_protocol.get("per_image_minmax") is not False
    ):
        raise ValueError("Wrong E4 probability protocol")

    e4_checkpoint_hash = digest(e4_run / "checkpoints" / "final.pth")
    if (
        e4_checkpoint_hash != e4_info.get("final_checkpoint_sha256")
        or e4_checkpoint_hash != e4_eval_info.get("checkpoint_sha256")
    ):
        raise ValueError("E4 final checkpoint/evaluation hash mismatch")

    if (
        e3_info.get("status") != "COMPLETED"
        or e3_info.get("run_kind") != "FULL_V3"
        or e3_info.get("processed_count") != 798
    ):
        raise ValueError("E3/V3 must be the completed 798-entry formal fusion")
    e3_config = e3_info.get("config", {})
    expected_e3 = {
        "weight_e0": 0.5,
        "weight_e2": 0.5,
        "threshold": THRESHOLD,
        "per_image_minmax": False,
        "probability_precision": "float32",
        "parameter_search": False,
    }
    if e3_config != expected_e3:
        raise ValueError("E3/V3 is not the fixed equal E0+E2 reference")
    if e3_info.get("e0_checkpoint_sha256") != e0_info.get("checkpoint_sha256"):
        raise ValueError("E3/V3 used a different E0 checkpoint")

    if digest(e0_run / "manifest.json") != e0_info.get("manifest_sha256"):
        raise ValueError("E0 manifest hash mismatch")
    e0_manifest = by_id(read_json(e0_run / "manifest.json"))
    e4_manifest = by_id(read_json(e4_run / "test_manifest.json"))
    e3_manifest = by_id(read_json(e3_run / "manifest.json"))
    if e0_manifest != e4_manifest or e0_manifest != e3_manifest:
        raise ValueError("E0/E4/E3 frozen test manifests differ")
    if dict(Counter(row["dataset"] for row in e0_manifest.values())) != COUNTS:
        raise ValueError("Frozen test subset counts are wrong")

    e0_rows = by_id(read_rows(e0_run / "per_image.csv"))
    e4_rows = by_id(read_rows(e4_evaluation / "per_image.csv"))
    e3_rows = by_id(read_rows(e3_run / "per_image.csv"))
    expected_ids = set(e0_manifest)
    if set(e0_rows) != expected_ids or set(e4_rows) != expected_ids or set(e3_rows) != expected_ids:
        raise ValueError("Missing/extra samples in source per_image.csv")

    return e0_manifest, e0_rows, e4_rows, e3_rows, e4_checkpoint_hash


def audit_source(directory, row, record, ground_truth, metric_prefix=""):
    shape = (int(record["height"]), int(record["width"]))
    if row["dataset"] != record["dataset"] or (
        int(row["height"]),
        int(row["width"]),
    ) != shape:
        raise ValueError("Source CSV dataset/shape mismatch: " + record["sample_id"])

    probability_path = under(directory, row["probability_path"])
    probability = load_probability(probability_path, shape)
    calculated = metrics(probability, ground_truth)
    for key, value in calculated.items():
        reference_key = metric_prefix + key
        close(
            value,
            row[reference_key],
            record["sample_id"] + "/" + reference_key,
            2e-6 if key == "mae" else 1e-12,
        )

    with Image.open(under(directory, row["mask_path"])) as image:
        saved_mask = np.array(image.convert("L"))
    expected_mask = (probability >= THRESHOLD).astype(np.uint8) * 255
    if not np.array_equal(saved_mask, expected_mask):
        raise ValueError("Source mask differs from raw probability threshold")

    return probability, calculated, digest(probability_path)


def audit_sample(
    project,
    e0_run,
    e4_evaluation,
    e3_run,
    record,
    e0_row,
    e4_row,
    e3_row,
):
    image_path = under(project, record["image"])
    mask_path = under(project, record["mask"])
    if (
        digest(image_path) != record["image_sha256"]
        or digest(mask_path) != record["mask_sha256"]
    ):
        raise ValueError("Test image/mask changed: " + record["sample_id"])

    shape = (int(record["height"]), int(record["width"]))
    ground_truth = load_gt(mask_path)
    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    if (
        ground_truth.shape != shape
        or rgb.shape[:2] != shape
        or int(ground_truth.sum()) != record["gt_foreground_pixels"]
    ):
        raise ValueError("Test shape/GT count mismatch: " + record["sample_id"])
    if hashlib.sha256(rgb.tobytes()).hexdigest() != record["decoded_rgb_sha256"]:
        raise ValueError("Decoded image mismatch: " + record["sample_id"])

    e0_probability, e0_metrics, e0_hash = audit_source(
        e0_run, e0_row, record, ground_truth
    )
    e4_probability, e4_metrics, e4_hash = audit_source(
        e4_evaluation, e4_row, record, ground_truth
    )
    e3_probability, e3_metrics, e3_hash = audit_source(
        e3_run, e3_row, record, ground_truth, metric_prefix="fusion_"
    )
    return (
        rgb,
        ground_truth,
        [e0_probability, e4_probability, e3_probability],
        [e0_metrics, e4_metrics, e3_metrics],
        [e0_hash, e4_hash, e3_hash],
    )


def summarize(data):
    result = []
    for dataset in ("ALL",) + DATASETS:
        group = data if dataset == "ALL" else [r for r in data if r["dataset"] == dataset]
        summary = {"dataset": dataset, "count": len(group)}
        for model in ("e0", "e4", "e3", "e5"):
            for metric_name in ("dice", "iou", "mae"):
                summary[f"{model}_{metric_name}"] = float(
                    np.mean([row[f"{model}_{metric_name}"] for row in group])
                )
            dice_values = np.array([row[f"{model}_dice"] for row in group])
            summary[f"{model}_median_dice"] = float(np.median(dice_values))
            summary[f"{model}_dice_lt05"] = int((dice_values < 0.5).sum())
            summary[f"{model}_zero_dice"] = int((dice_values == 0).sum())

        for model in ("e0", "e4", "e3"):
            for metric_name in ("dice", "iou", "mae"):
                summary[f"delta_{metric_name}_vs_{model}"] = (
                    summary[f"e5_{metric_name}"] - summary[f"{model}_{metric_name}"]
                )
            delta = np.array(
                [row["e5_dice"] - row[f"{model}_dice"] for row in group]
            )
            summary[f"wins_vs_{model}"] = int((delta > 1e-12).sum())
            summary[f"ties_vs_{model}"] = int((np.abs(delta) <= 1e-12).sum())
            summary[f"losses_vs_{model}"] = int((delta < -1e-12).sum())
        result.append(summary)
    return result


def overlay(rgb, mask):
    if mask is None:
        return rgb
    green = np.array([0, 220, 120], dtype=np.float32)
    return np.where(mask[..., None], green * 0.45 + rgb * 0.55, rgb).astype(np.uint8)


def preview(
    output,
    project,
    e0_run,
    e4_evaluation,
    e3_run,
    manifest,
    e0_rows,
    e4_rows,
    e3_rows,
    data,
):
    # Deterministic descriptive selection; never used to select parameters/models.
    selected = {}
    for dataset in DATASETS:
        group = [row for row in data if row["dataset"] == dataset]
        picks = [
            (
                "middle E5 Dice",
                sorted(group, key=lambda row: (row["e5_dice"], row["sample_id"]))[
                    (len(group) - 1) // 2
                ],
            ),
            (
                "largest gain vs E0",
                min(group, key=lambda row: (-row["delta_dice_vs_e0"], row["sample_id"])),
            ),
            (
                "largest drop vs E0",
                min(group, key=lambda row: (row["delta_dice_vs_e0"], row["sample_id"])),
            ),
            (
                "largest gain vs E3",
                min(group, key=lambda row: (-row["delta_dice_vs_e3"], row["sample_id"])),
            ),
            (
                "largest drop vs E3",
                min(group, key=lambda row: (row["delta_dice_vs_e3"], row["sample_id"])),
            ),
        ]
        for rule, row in picks:
            selected.setdefault(row["sample_id"], {"row": row, "rules": []})[
                "rules"
            ].append(rule)

    visualizations = output / "visualizations"
    visualizations.mkdir()
    blocks = []
    selections = []
    for index, (sample_id, item) in enumerate(selected.items(), 1):
        row = item["row"]
        rgb, ground_truth, probabilities, _, _ = audit_sample(
            project,
            e0_run,
            e4_evaluation,
            e3_run,
            manifest[sample_id],
            e0_rows[sample_id],
            e4_rows[sample_id],
            e3_rows[sample_id],
        )
        e5_probability = fuse(probabilities[0], probabilities[1])
        labels = [
            "Image",
            "Ground truth",
            "E0 Polyp-PVT",
            "E4 strong U-Net",
            "E3 E0+E2 fusion",
            "E5 E0+E4 fusion",
        ]
        masks = [
            None,
            ground_truth,
            probabilities[0] >= THRESHOLD,
            probabilities[1] >= THRESHOLD,
            probabilities[2] >= THRESHOLD,
            e5_probability >= THRESHOLD,
        ]
        canvas = Image.new("RGB", (6 * 300, 260), "white")
        draw = ImageDraw.Draw(canvas)
        for panel, (label, mask) in enumerate(zip(labels, masks)):
            image = Image.fromarray(overlay(rgb, mask))
            image.thumbnail((292, 210))
            canvas.paste(
                image,
                (panel * 300 + (300 - image.width) // 2, 30 + (210 - image.height) // 2),
            )
            draw.text((panel * 300 + 8, 8), label, fill="black")
        filename = f"{index:02d}.png"
        canvas.save(visualizations / filename)

        selection = {
            "sample_id": sample_id,
            "selection_rules": item["rules"],
            "e0_dice": row["e0_dice"],
            "e4_dice": row["e4_dice"],
            "e3_dice": row["e3_dice"],
            "e5_dice": row["e5_dice"],
        }
        selections.append(selection)
        heading = html.escape(sample_id + " | " + ", ".join(item["rules"]))
        blocks.append(
            f'<section><h2>{heading}</h2>'
            f'<p>E0 {row["e0_dice"]:.4f} / E4 {row["e4_dice"]:.4f} / '
            f'E3 {row["e3_dice"]:.4f} / E5 {row["e5_dice"]:.4f} / '
            f'delta E0 {row["delta_dice_vs_e0"]:+.4f} / '
            f'delta E3 {row["delta_dice_vs_e3"]:+.4f}</p>'
            f'<img src="visualizations/{filename}" alt="Segmentation comparison"></section>'
        )

    write_json(output / "visualization_selection.json", selections)
    page = (
        '<!doctype html><meta charset="utf-8"><title>EndoLens AI E5</title>'
        '<style>body{font:16px Arial;margin:32px;max-width:1800px;background:#f4f7fa;'
        'color:#17324b}section{background:white;padding:20px;margin:20px 0}'
        'img{width:100%;height:auto}h2{font-size:18px}</style>'
        '<h1>E5 fixed equal probability fusion: E0 + E4</h1>'
        '<p>Examples include gains and drops against E0 and E3. No weights or thresholds '
        'were selected using these results. Green overlays indicate predicted foreground; '
        'frame counts are not lesion counts.</p>'
        + "".join(blocks)
    )
    (output / "comparison.html").write_text(page, encoding="utf-8")


def resolve_input(project, value):
    path = Path(value)
    return path.expanduser().resolve() if path.is_absolute() else (project / path).resolve()


def run(args):
    project = Path(args.project).expanduser().resolve()
    e0_run = resolve_input(project, args.e0_run)
    e4_run = resolve_input(project, args.e4_run)
    e4_evaluation = resolve_input(project, args.e4_eval)
    e3_run = resolve_input(project, args.e3_run)
    output = resolve_input(project, args.output)
    if output.exists():
        raise FileExistsError(
            "Output already exists; choose a new --output. No overwrite: " + str(output)
        )
    output.mkdir(parents=True)

    started = time.perf_counter()
    mode = "AUDIT_ONLY" if args.audit_only else ("SMOKE_ONLY" if args.smoke else "FULL_E5")
    info = {
        "status": "RUNNING",
        "run_kind": mode,
        "started_at": now(),
        "project": str(project),
        "sources": {
            "e0_run": str(e0_run),
            "e4_run": str(e4_run),
            "e4_eval": str(e4_evaluation),
            "e3_run": str(e3_run),
        },
        "config": {
            "weight_e0": WEIGHT_E0,
            "weight_e4": WEIGHT_E4,
            "threshold": THRESHOLD,
            "per_image_minmax": False,
            "probability_precision": "float32",
            "parameter_search": False,
            "fusion_inputs": "original-resolution stored sigmoid probabilities",
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "platform": platform.platform(),
            "device": "CPU (NumPy cached-probability fusion; no training/model inference)",
        },
        "source_hashes": {
            path.name: digest(path) for path in Path(__file__).parent.glob("*.py")
        },
    }
    write_json(output / "run_info.json", info)

    try:
        manifest, e0_rows, e4_rows, e3_rows, e4_checkpoint = validate_sources(
            project, e0_run, e4_run, e4_evaluation, e3_run
        )
        records = sorted(
            manifest.values(),
            key=lambda record: (DATASETS.index(record["dataset"]), record["sample_id"]),
        )
        if args.smoke and not args.audit_only:
            records = [
                next(record for record in records if record["dataset"] == dataset)
                for dataset in DATASETS
            ]
        write_json(output / "manifest.json", records)

        e0_info = read_json(e0_run / "run_info.json")
        info["e0_checkpoint_sha256"] = e0_info["checkpoint_sha256"]
        info["e4_checkpoint_sha256"] = e4_checkpoint
        info["e3_run_info_sha256"] = digest(e3_run / "run_info.json")
        metadata_files = [
            e0_run / "run_info.json",
            e0_run / "manifest.json",
            e0_run / "per_image.csv",
            e4_run / "run_info.json",
            e4_run / "e4_stage.json",
            e4_run / "test_manifest.json",
            e4_evaluation / "run_info.json",
            e4_evaluation / "per_image.csv",
            e3_run / "run_info.json",
            e3_run / "manifest.json",
            e3_run / "per_image.csv",
        ]
        info["source_metadata_sha256"] = {
            str(path): digest(path) for path in metadata_files
        }

        data = []
        fingerprints = []
        for index, record in enumerate(records, 1):
            sample_id = record["sample_id"]
            _, ground_truth, probabilities, source_metrics, hashes = audit_sample(
                project,
                e0_run,
                e4_evaluation,
                e3_run,
                record,
                e0_rows[sample_id],
                e4_rows[sample_id],
                e3_rows[sample_id],
            )
            fingerprints.append(
                {
                    "sample_id": sample_id,
                    "e0_probability_sha256": hashes[0],
                    "e4_probability_sha256": hashes[1],
                    "e3_probability_sha256": hashes[2],
                    "image_sha256": record["image_sha256"],
                    "mask_sha256": record["mask_sha256"],
                }
            )

            if not args.audit_only:
                e5_probability = fuse(probabilities[0], probabilities[1])
                e5_metrics = metrics(e5_probability, ground_truth)
                probability_path = Path("probabilities") / (sample_id + ".npz")
                mask_path = Path("masks") / (sample_id + ".png")
                for path in (output / probability_path, output / mask_path):
                    path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output / probability_path, probability=e5_probability)
                Image.fromarray(
                    (e5_probability >= THRESHOLD).astype(np.uint8) * 255
                ).save(output / mask_path)

                row = {
                    "sample_id": sample_id,
                    "dataset": record["dataset"],
                    "height": ground_truth.shape[0],
                    "width": ground_truth.shape[1],
                    "gt_area_fraction": float(ground_truth.mean()),
                }
                for model, model_metrics in zip(
                    ("e0", "e4", "e3", "e5"), source_metrics + [e5_metrics]
                ):
                    row.update(
                        {f"{model}_{key}": value for key, value in model_metrics.items()}
                    )
                row.update(
                    {
                        "delta_dice_vs_e0": e5_metrics["dice"]
                        - source_metrics[0]["dice"],
                        "delta_dice_vs_e4": e5_metrics["dice"]
                        - source_metrics[1]["dice"],
                        "delta_dice_vs_e3": e5_metrics["dice"]
                        - source_metrics[2]["dice"],
                        "probability_path": probability_path.as_posix(),
                        "mask_path": mask_path.as_posix(),
                    }
                )
                data.append(row)

            if index % 50 == 0 or index == len(records):
                print(f"[{mode} {index}/{len(records)}] {sample_id}", flush=True)

        write_csv(output / "input_fingerprints.csv", fingerprints)
        if args.audit_only:
            write_json(
                output / "audit.json",
                {
                    "passed": True,
                    "processed_count": len(records),
                    "checks": (
                        "E0/E4/E3 status and provenance, frozen manifests, image/GT bytes, "
                        "probability dtype/shape/range, thresholded masks, and recomputed metrics"
                    ),
                },
            )
            print(
                "AUDIT PASSED: all 798 E0/E4/E3 source entries verified. "
                "No E5 metrics produced.",
                flush=True,
            )
        else:
            write_csv(output / "per_image.csv", data)
            summary = summarize(data)
            write_csv(output / "summary.csv", summary)
            write_json(
                output / "metrics.json",
                {
                    "run_kind": mode,
                    "summary": summary,
                    "selection": "fixed 0.5/0.5 E0+E4 weights and threshold 0.5 before E5",
                    "comparisons": ["E0", "E4", "E3/V3"],
                    "timing": "No model forward pass or video FPS measured",
                },
            )
            preview(
                output,
                project,
                e0_run,
                e4_evaluation,
                e3_run,
                manifest,
                e0_rows,
                e4_rows,
                e3_rows,
                data,
            )
            print(mode, flush=True)
            for summary_row in summary:
                print(
                    f"{summary_row['dataset']:20s} n={summary_row['count']:3d} "
                    f"E0={summary_row['e0_dice']:.4f} "
                    f"E4={summary_row['e4_dice']:.4f} "
                    f"E3={summary_row['e3_dice']:.4f} "
                    f"E5={summary_row['e5_dice']:.4f} "
                    f"delta_E0={summary_row['delta_dice_vs_e0']:+.4f} "
                    f"delta_E3={summary_row['delta_dice_vs_e3']:+.4f}",
                    flush=True,
                )

        info.update(
            {
                "status": "COMPLETED",
                "processed_count": len(records),
                "finished_at": now(),
                "elapsed_seconds": time.perf_counter() - started,
                "timing": (
                    "Offline audit/fusion/file-output wall time; not model inference or video FPS"
                ),
            }
        )
        write_json(output / "run_info.json", info)
        print("Output:", output, flush=True)
    except Exception as error:
        info.update(
            {"status": "FAILED", "finished_at": now(), "error": repr(error)}
        )
        write_json(output / "run_info.json", info)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", default=str(Path(__file__).resolve().parent.parent)
    )
    parser.add_argument(
        "--e0-run", default="runs/e0_20261004_221345_872444"
    )
    parser.add_argument("--e4-run", default="runs/e4_final_seed42")
    parser.add_argument(
        "--e4-eval",
        default="runs/e4_final_seed42/evaluation_cuda_20261006_235221_022843",
    )
    parser.add_argument("--e3-run", default="runs/v3_fusion_equal_seed42")
    parser.add_argument(
        "--output", required=True, help="New output folder; it must not exist"
    )
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--smoke", action="store_true")
    flags.add_argument("--audit-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
