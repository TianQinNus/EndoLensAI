#!/usr/bin/env python3
"""E6: calibration-aware E0/E4 fusion with a locked development/test protocol.

Workflow:
  prepare     audit a new, unseen development dataset and freeze calibration/selection splits
  infer       generate E0/E4 original-resolution sigmoid probabilities on that dataset
  select      fit temperatures on calibration, search alpha/threshold on selection, lock config
  apply-test  apply a PASS configuration once to the frozen 798-image test set

The existing 798-image test set and the 1450-image training set are explicitly blocked
from development-set preparation. No model is trained by this script.
"""

import argparse
import csv
import hashlib
import importlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


DATASETS = (
    "Kvasir",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "ETIS-LaribPolypDB",
    "CVC-300",
)
TEST_COUNTS = dict(zip(DATASETS, (100, 62, 380, 196, 60)))
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
DEFAULT_GATES = {
    "min_mean_dice_gain": 0.003,
    "min_source_dice_delta": -0.003,
    "max_mae_increase": 0.003,
    "max_dice_lt05_increase": 0,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def signature(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve(project, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(project) / path).resolve()


def safe_under(root, relative):
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Expected safe relative POSIX path: " + str(relative))
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Path escapes root: " + relative)
    return path


def by_id(rows):
    result = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in result:
            raise ValueError("Duplicate sample_id: " + sample_id)
        result[sample_id] = row
    return result


def image_index(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError("Missing folder: " + str(folder))
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in result:
                raise ValueError("Duplicate stem: " + path.stem)
            result[path.stem] = path
    if not result:
        raise ValueError("No supported images in: " + str(folder))
    return result


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def read_mask(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array > (0 if int(array.max()) <= 1 else 127.5)


def load_probability(path, expected_shape=None):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"probability"}:
            raise ValueError("NPZ must contain only key 'probability': " + str(path))
        probability = archive["probability"]
    if probability.dtype != np.float32 or probability.ndim != 2:
        raise ValueError("Probability must be float32 HxW: " + str(path))
    if expected_shape is not None and probability.shape != tuple(expected_shape):
        raise ValueError("Probability shape mismatch: " + str(path))
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError("Invalid probability range/nonfinite values: " + str(path))
    return probability


def probability_metrics(probability, ground_truth, threshold=0.5):
    prediction = probability >= threshold
    ground_truth = ground_truth.astype(bool)
    tp = int(np.count_nonzero(prediction & ground_truth))
    fp = int(np.count_nonzero(prediction & ~ground_truth))
    fn = int(np.count_nonzero(~prediction & ground_truth))
    denominator = 2 * tp + fp + fn
    union = tp + fp + fn
    return {
        "dice": 2 * tp / denominator if denominator else 1.0,
        "iou": tp / union if union else 1.0,
        "mae": float(np.abs(probability - ground_truth.astype(np.float32)).mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def probability_to_logit(probability):
    clipped = np.clip(probability.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def calibrate_probability(probability, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = probability_to_logit(probability) / float(temperature)
    result = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1.0 + exp_logits)
    return result.astype(np.float32)


def binary_nll(logits, targets, temperature):
    scaled = logits / float(temperature)
    return float(np.mean(np.maximum(scaled, 0) - scaled * targets + np.log1p(np.exp(-np.abs(scaled)))))


def deterministic_sample(probability, ground_truth, sample_id, seed, maximum):
    flat_probability = probability.reshape(-1)
    flat_ground_truth = ground_truth.reshape(-1).astype(np.float64)
    if flat_probability.size <= maximum:
        indices = np.arange(flat_probability.size)
    else:
        token = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(token[:8], "little"))
        indices = np.sort(rng.choice(flat_probability.size, size=maximum, replace=False))
    return probability_to_logit(flat_probability[indices]), flat_ground_truth[indices]


def fit_temperature(samples):
    if not samples:
        raise ValueError("No calibration samples")
    logits = np.concatenate([item[0] for item in samples])
    targets = np.concatenate([item[1] for item in samples])
    if logits.size == 0 or targets.min() == targets.max():
        raise ValueError("Calibration pixels must contain foreground and background")
    coarse = np.exp(np.linspace(math.log(0.25), math.log(4.0), 161))
    losses = np.array([binary_nll(logits, targets, value) for value in coarse])
    best = int(np.argmin(losses))
    left = coarse[max(0, best - 1)]
    right = coarse[min(len(coarse) - 1, best + 1)]
    fine = np.exp(np.linspace(math.log(left), math.log(right), 81))
    fine_losses = np.array([binary_nll(logits, targets, value) for value in fine])
    winner = int(np.argmin(fine_losses))
    temperature = float(fine[winner])
    calibrated = calibrate_probability(
        (1.0 / (1.0 + np.exp(-logits))).astype(np.float32), temperature
    ).astype(np.float64)
    before = 1.0 / (1.0 + np.exp(-logits))
    return {
        "temperature": temperature,
        "sampled_pixels": int(logits.size),
        "nll_before": binary_nll(logits, targets, 1.0),
        "nll_after": float(fine_losses[winner]),
        "brier_before": float(np.mean((before - targets) ** 2)),
        "brier_after": float(np.mean((calibrated - targets) ** 2)),
    }


def area_bin(fraction):
    if fraction < 0.01:
        return "small_lt1pct"
    if fraction < 0.10:
        return "medium_1to10pct"
    return "large_ge10pct"


def load_metadata(path, stems):
    if path is None:
        return {stem: {"source": "E6Development", "group_id": stem} for stem in stems}
    rows = read_csv(path)
    required = {"stem", "source", "group_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Metadata needs columns: stem,source,group_id")
    metadata = {}
    for row in rows:
        stem = row["stem"].strip()
        if not stem or stem in metadata or not row["source"].strip() or not row["group_id"].strip():
            raise ValueError("Metadata contains blank/duplicate identifiers")
        metadata[stem] = {"source": row["source"].strip(), "group_id": row["group_id"].strip()}
    if set(metadata) != set(stems):
        raise ValueError("Metadata must contain exactly one row for every image stem")
    return metadata


def split_records(records, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("Calibration fraction must be between 0 and 1")
    parent = list(range(len(records)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = root(left), root(right)
        if left != right:
            parent[right] = left

    seen_group = {}
    seen_image = {}
    for index, record in enumerate(records):
        group_key = (record["source"], record["declared_group_id"])
        image_key = (record["height"], record["width"], record["decoded_rgb_sha256"])
        if group_key in seen_group:
            union(index, seen_group[group_key])
        if image_key in seen_image:
            union(index, seen_image[image_key])
        seen_group[group_key] = index
        seen_image[image_key] = index

    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[root(index)].append(record)
    if len(groups) < 2:
        raise ValueError("At least two independent groups are required")

    strata = defaultdict(list)
    for group in groups.values():
        key = Counter((r["source"], r["area_bin"]) for r in group).most_common(1)[0][0]
        strata[key].append(group)
    target = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    allocations = {key: math.floor(len(values) * fraction) for key, values in strata.items()}
    remainder = target - sum(allocations.values())
    order = sorted(
        strata,
        key=lambda key: (-(len(strata[key]) * fraction - allocations[key]), key),
    )
    for key in order[:remainder]:
        allocations[key] += 1

    calibration_roots = set()
    for key in sorted(strata):
        values = sorted(
            strata[key],
            key=lambda group: signature([seed, min(r["sample_id"] for r in group)]),
        )
        calibration_roots.update(root(records.index(group[0])) for group in values[: allocations[key]])

    output = []
    for index, record in enumerate(records):
        value = dict(record)
        value["split"] = "calibration" if root(index) in calibration_roots else "selection"
        value["group_id"] = min(r["sample_id"] for r in groups[root(index)])
        output.append(value)
    if {r["split"] for r in output} != {"calibration", "selection"}:
        raise ValueError("Split produced an empty partition")
    if {r["group_id"] for r in output if r["split"] == "calibration"} & {
        r["group_id"] for r in output if r["split"] == "selection"
    }:
        raise AssertionError("Group leakage between calibration and selection")
    return sorted(output, key=lambda row: row["sample_id"])


def existing_image_keys(project, e0_run, e4_run):
    e0_manifest = read_json(e0_run / "manifest.json")
    full_train = read_json(e4_run / "full_train_manifest.json")
    test_keys = {
        (int(r["height"]), int(r["width"]), r["decoded_rgb_sha256"])
        for r in e0_manifest
    }
    train_keys = {
        (int(r["height"]), int(r["width"]), r["decoded_rgb_sha256"])
        for r in full_train
    }
    return train_keys, test_keys


def command_prepare(args):
    project = Path(args.project).expanduser().resolve()
    dataset = resolve(project, args.dataset)
    output = resolve(project, args.output)
    e0_run = resolve(project, args.e0_run)
    e4_run = resolve(project, args.e4_run)
    metadata_path = resolve(project, args.metadata) if args.metadata else None
    if output.exists():
        raise FileExistsError("Output already exists; choose a new directory: " + str(output))
    output.mkdir(parents=True)
    info = {
        "status": "RUNNING",
        "run_kind": "E6_PREPARE",
        "started_at": utc_now(),
        "dataset": str(dataset),
        "calibration_fraction": args.calibration_fraction,
        "seed": args.seed,
    }
    write_json(output / "run_info.json", info)
    try:
        images = image_index(dataset / "images")
        masks = image_index(dataset / "masks")
        if images.keys() != masks.keys():
            raise ValueError("Development images and masks are not paired by stem")
        metadata = load_metadata(metadata_path, images.keys())
        train_keys, test_keys = existing_image_keys(project, e0_run, e4_run)
        records = []
        overlaps = []
        for stem in sorted(images):
            rgb = read_rgb(images[stem])
            ground_truth = read_mask(masks[stem])
            if rgb.shape[:2] != ground_truth.shape:
                raise ValueError("Image/mask shape mismatch: " + stem)
            decoded = hashlib.sha256(rgb.tobytes()).hexdigest()
            key = (rgb.shape[0], rgb.shape[1], decoded)
            if key in train_keys:
                overlaps.append({"stem": stem, "overlap": "existing_1450_training_set"})
            if key in test_keys:
                overlaps.append({"stem": stem, "overlap": "locked_798_test_set"})
            source = metadata[stem]["source"]
            sample_id = source.replace("\\", "_").replace("/", "_") + "/" + stem
            records.append(
                {
                    "sample_id": sample_id,
                    "stem": stem,
                    "source": source,
                    "declared_group_id": metadata[stem]["group_id"],
                    "image": images[stem].relative_to(project).as_posix(),
                    "mask": masks[stem].relative_to(project).as_posix(),
                    "height": rgb.shape[0],
                    "width": rgb.shape[1],
                    "gt_foreground_pixels": int(ground_truth.sum()),
                    "gt_area_fraction": float(ground_truth.mean()),
                    "area_bin": area_bin(float(ground_truth.mean())),
                    "decoded_rgb_sha256": decoded,
                    "image_sha256": file_hash(images[stem]),
                    "mask_sha256": file_hash(masks[stem]),
                }
            )
        if overlaps:
            write_json(output / "overlap_failures.json", overlaps)
            raise ValueError(
                "Development data overlaps existing train/test data; see overlap_failures.json"
            )
        if len({r["sample_id"] for r in records}) != len(records):
            raise ValueError("source/stem combinations do not produce unique sample_id values")
        records = split_records(records, args.calibration_fraction, args.seed)
        write_json(output / "development_manifest.json", records)
        counts = dict(Counter(r["split"] for r in records))
        source_counts = {
            split: dict(Counter(r["source"] for r in records if r["split"] == split))
            for split in ("calibration", "selection")
        }
        info.update(
            {
                "status": "COMPLETED",
                "finished_at": utc_now(),
                "processed_count": len(records),
                "counts": counts,
                "source_counts": source_counts,
                "manifest_sha256": file_hash(output / "development_manifest.json"),
                "manifest_signature": signature(records),
                "e0_test_manifest_sha256": file_hash(e0_run / "manifest.json"),
                "e4_full_train_manifest_sha256": file_hash(e4_run / "full_train_manifest.json"),
                "exact_overlap_check": "PASS",
                "limitation": (
                    "Exact decoded-image overlap is blocked. Near-duplicate/patient independence "
                    "requires reliable metadata and manual source verification."
                ),
            }
        )
        write_json(output / "run_info.json", info)
        print("E6 development manifest prepared:", output)
        print("Counts:", counts)
    except BaseException as error:
        info.update(status="FAILED", finished_at=utc_now(), error=f"{type(error).__name__}: {error}")
        write_json(output / "run_info.json", info)
        raise


def verify_development_manifest(project, manifest_run):
    info = read_json(manifest_run / "run_info.json")
    if info.get("status") != "COMPLETED" or info.get("run_kind") != "E6_PREPARE":
        raise ValueError("Expected a completed E6_PREPARE directory")
    manifest_path = manifest_run / "development_manifest.json"
    if file_hash(manifest_path) != info.get("manifest_sha256"):
        raise ValueError("Development manifest changed")
    records = read_json(manifest_path)
    if signature(records) != info.get("manifest_signature"):
        raise ValueError("Development manifest signature mismatch")
    for record in records:
        image = safe_under(project, record["image"])
        mask = safe_under(project, record["mask"])
        if file_hash(image) != record["image_sha256"] or file_hash(mask) != record["mask_sha256"]:
            raise ValueError("Development file changed: " + record["sample_id"])
    return info, records


def normalized_tensor(image, size, torch):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    array = (array - mean) / std
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0)


def infer_model(records, project, output, model_name, model, device, torch, functional, size=352, warmup=3):
    rows = []
    model.eval()
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, size, size, device=device)
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for index, record in enumerate(records, 1):
            ground_truth = read_mask(safe_under(project, record["mask"]))
            with Image.open(safe_under(project, record["image"])) as image:
                tensor = normalized_tensor(image.convert("RGB"), size, torch).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            raw = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_ms = (time.perf_counter() - started) * 1000
            logits = raw[0] + raw[1] if model_name == "e0" else raw
            probability = torch.sigmoid(
                functional.interpolate(logits.float(), size=ground_truth.shape, mode="bilinear", align_corners=False)
            )[0, 0].cpu().numpy().astype(np.float32)
            if not np.isfinite(probability).all():
                raise ValueError("Nonfinite inference output: " + record["sample_id"])
            relative = Path("probabilities") / model_name / (record["sample_id"] + ".npz")
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, probability=probability)
            metrics = probability_metrics(probability, ground_truth, 0.5)
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "split": record["split"],
                    "source": record["source"],
                    "model": model_name,
                    "height": ground_truth.shape[0],
                    "width": ground_truth.shape[1],
                    **metrics,
                    "forward_ms": forward_ms,
                    "probability_path": relative.as_posix(),
                    "probability_sha256": file_hash(path),
                }
            )
            if index % 25 == 0 or index == len(records):
                print(f"[{model_name} {index}/{len(records)}] {record['sample_id']}", flush=True)
    return rows


def command_infer(args):
    project = Path(args.project).expanduser().resolve()
    manifest_run = resolve(project, args.manifest_run)
    output = resolve(project, args.output)
    e0_run = resolve(project, args.e0_run)
    e4_run = resolve(project, args.e4_run)
    e0_checkpoint = resolve(project, args.e0_checkpoint)
    if output.exists():
        raise FileExistsError("Output already exists: " + str(output))
    output.mkdir(parents=True)
    manifest_info, all_records = verify_development_manifest(project, manifest_run)
    records = all_records
    run_kind = "FULL_DEVELOPMENT"
    if args.smoke:
        calibration = next(r for r in records if r["split"] == "calibration")
        selection = next(r for r in records if r["split"] == "selection")
        records = [calibration, selection]
        run_kind = "SMOKE_DEVELOPMENT"
    info = {
        "status": "RUNNING",
        "run_kind": run_kind,
        "started_at": utc_now(),
        "manifest_signature": manifest_info["manifest_signature"],
        "expected_count": len(records),
        "device_requested": args.device,
    }
    write_json(output / "run_info.json", info)
    try:
        import torch
        import torch.nn.functional as functional

        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; activate endolens-e2-win")
            device = torch.device("cuda:0")
            test = torch.randn(16, 16, device=device)
            (test @ test).sum().item()
        else:
            device = torch.device("cpu")
        torch.set_num_threads(args.threads)

        e0_info = read_json(e0_run / "run_info.json")
        if e0_info.get("status") != "COMPLETED" or e0_info.get("run_kind") != "FULL_E0":
            raise ValueError("Invalid E0 reference run")
        if file_hash(e0_checkpoint) != e0_info.get("checkpoint_sha256"):
            raise ValueError("Local E0 checkpoint does not match the frozen E0 run")
        if str(project) not in sys.path:
            sys.path.insert(0, str(project))
        from pvt_timm_compat import install_if_missing

        used_timm_compat = install_if_missing()
        from lib.pvt import PolypPVT

        e0_model = PolypPVT()
        e0_state = torch.load(e0_checkpoint, map_location="cpu", weights_only=True)
        e0_model.load_state_dict(e0_state, strict=True)
        e0_model.to(device).eval()
        rows = infer_model(records, project, output, "e0", e0_model, device, torch, functional)
        del e0_model, e0_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

        e4_info = read_json(e4_run / "run_info.json")
        e4_checkpoint = e4_run / "checkpoints" / "final.pth"
        if (
            e4_info.get("status") != "TRAINED"
            or e4_info.get("run_kind") != "E2_FINAL"
            or file_hash(e4_checkpoint) != e4_info.get("final_checkpoint_sha256")
        ):
            raise ValueError("Invalid E4 final checkpoint/run")
        core = project / "e4_windows" / "core"
        if str(core) not in sys.path:
            sys.path.insert(0, str(core))
        e4_models = importlib.import_module("models")
        checkpoint = torch.load(e4_checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["config"] != e4_info["config"]:
            raise ValueError("E4 checkpoint config mismatch")
        e4_model = e4_models.make_model(checkpoint["config"], initialization=False)
        e4_model.load_state_dict(checkpoint["model_state"], strict=True)
        e4_model.to(device).eval()
        rows.extend(infer_model(records, project, output, "e4", e4_model, device, torch, functional))
        write_csv(output / "per_image.csv", rows)
        info.update(
            {
                "status": "COMPLETED",
                "finished_at": utc_now(),
                "processed_count": len(records),
                "probability_count": len(rows),
                "e0_checkpoint_sha256": file_hash(e0_checkpoint),
                "e4_checkpoint_sha256": file_hash(e4_checkpoint),
                "manifest_sha256": manifest_info["manifest_sha256"],
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "pvt_timm_compatibility_fallback": used_timm_compat,
                },
                "protocol": {
                    "input": "RGB, Pillow bilinear 352x352, ImageNet normalization",
                    "e0_output": "sigmoid(resize(P1+P2 logits to original size))",
                    "e4_output": "sigmoid(resize(logits to original size))",
                    "per_image_minmax": False,
                    "storage": "float32 NPZ at original resolution",
                },
            }
        )
        write_json(output / "run_info.json", info)
        print("Development inference complete:", output)
    except BaseException as error:
        info.update(status="FAILED", finished_at=utc_now(), error=f"{type(error).__name__}: {error}")
        write_json(output / "run_info.json", info)
        raise


def verify_inference(project, manifest_info, records, inference_run):
    info = read_json(inference_run / "run_info.json")
    if info.get("status") != "COMPLETED" or info.get("run_kind") != "FULL_DEVELOPMENT":
        raise ValueError("Selection requires FULL_DEVELOPMENT inference, never smoke")
    if info.get("processed_count") != len(records) or info.get("manifest_signature") != manifest_info["manifest_signature"]:
        raise ValueError("Inference/manifest mismatch")
    rows = read_csv(inference_run / "per_image.csv")
    index = {(row["model"], row["sample_id"]): row for row in rows}
    if len(index) != 2 * len(records):
        raise ValueError("Missing/duplicate development probability rows")
    for record in records:
        for model in ("e0", "e4"):
            row = index[(model, record["sample_id"])]
            path = safe_under(inference_run, row["probability_path"])
            if file_hash(path) != row["probability_sha256"]:
                raise ValueError("Development probability changed: " + record["sample_id"])
    return info, index


def probability_path(inference_run, index, model, sample_id):
    return safe_under(inference_run, index[(model, sample_id)]["probability_path"])


def empty_accumulator():
    return {"dice": [], "iou": [], "mae": [], "dice_lt05": 0, "zero_dice": 0, "sources": defaultdict(list)}


def add_metrics(accumulator, source, metrics):
    for name in ("dice", "iou", "mae"):
        accumulator[name].append(metrics[name])
    accumulator["dice_lt05"] += int(metrics["dice"] < 0.5)
    accumulator["zero_dice"] += int(metrics["dice"] == 0)
    accumulator["sources"][source].append(metrics["dice"])


def finish_accumulator(accumulator):
    return {
        "count": len(accumulator["dice"]),
        "dice": float(np.mean(accumulator["dice"])),
        "iou": float(np.mean(accumulator["iou"])),
        "mae": float(np.mean(accumulator["mae"])),
        "median_dice": float(np.median(accumulator["dice"])),
        "dice_lt05": accumulator["dice_lt05"],
        "zero_dice": accumulator["zero_dice"],
        "source_dice": {key: float(np.mean(value)) for key, value in accumulator["sources"].items()},
    }


def gate_candidate(candidate, baseline, gates):
    source_deltas = [
        candidate["source_dice"][source] - baseline["source_dice"][source]
        for source in baseline["source_dice"]
    ]
    candidate["delta_dice_vs_e0"] = candidate["dice"] - baseline["dice"]
    candidate["delta_mae_vs_e0"] = candidate["mae"] - baseline["mae"]
    candidate["worst_source_delta_vs_e0"] = min(source_deltas)
    candidate["dice_lt05_delta_vs_e0"] = candidate["dice_lt05"] - baseline["dice_lt05"]
    checks = {
        "mean_dice_gain": candidate["delta_dice_vs_e0"] >= gates["min_mean_dice_gain"],
        "worst_source": candidate["worst_source_delta_vs_e0"] >= gates["min_source_dice_delta"],
        "mae": candidate["delta_mae_vs_e0"] <= gates["max_mae_increase"],
        "failure_tail": candidate["dice_lt05_delta_vs_e0"] <= gates["max_dice_lt05_increase"],
    }
    candidate["passes"] = all(checks.values())
    candidate["checks"] = checks
    candidate["violation_score"] = (
        max(0.0, gates["min_mean_dice_gain"] - candidate["delta_dice_vs_e0"])
        + max(0.0, gates["min_source_dice_delta"] - candidate["worst_source_delta_vs_e0"])
        + max(0.0, candidate["delta_mae_vs_e0"] - gates["max_mae_increase"])
        + max(0, candidate["dice_lt05_delta_vs_e0"] - gates["max_dice_lt05_increase"])
        / max(1, candidate["count"])
    )
    return candidate


def choose_candidate(candidates):
    passing = [candidate for candidate in candidates if candidate["passes"]]
    if passing:
        return sorted(
            passing,
            key=lambda c: (-c["dice"], -c["worst_source_delta_vs_e0"], c["mae"], c["dice_lt05"], c["alpha"], abs(c["threshold"] - 0.5)),
        )[0], "PASS"
    return sorted(
        candidates,
        key=lambda c: (c["violation_score"], -c["dice"], c["mae"], c["dice_lt05"], c["alpha"], abs(c["threshold"] - 0.5)),
    )[0], "NO_PASS"


def command_select(args):
    project = Path(args.project).expanduser().resolve()
    manifest_run = resolve(project, args.manifest_run)
    inference_run = resolve(project, args.inference_run)
    output = resolve(project, args.output)
    if output.exists():
        raise FileExistsError("Output already exists: " + str(output))
    output.mkdir(parents=True)
    info = {"status": "RUNNING", "run_kind": "E6_SELECT", "started_at": utc_now()}
    write_json(output / "run_info.json", info)
    try:
        manifest_info, records = verify_development_manifest(project, manifest_run)
        inference_info, index = verify_inference(project, manifest_info, records, inference_run)
        calibration_records = [r for r in records if r["split"] == "calibration"]
        selection_records = [r for r in records if r["split"] == "selection"]
        fitted = {}
        for model in ("e0", "e4"):
            samples = []
            for record in calibration_records:
                ground_truth = read_mask(safe_under(project, record["mask"]))
                probability = load_probability(
                    probability_path(inference_run, index, model, record["sample_id"]),
                    ground_truth.shape,
                )
                samples.append(
                    deterministic_sample(
                        probability,
                        ground_truth,
                        record["sample_id"],
                        args.seed,
                        args.max_pixels_per_image,
                    )
                )
            fitted[model] = fit_temperature(samples)
            print(
                f"{model} temperature={fitted[model]['temperature']:.6f} "
                f"NLL {fitted[model]['nll_before']:.6f}->{fitted[model]['nll_after']:.6f}",
                flush=True,
            )

        baseline_acc = empty_accumulator()
        candidate_acc = {(alpha, threshold): empty_accumulator() for alpha in ALPHAS for threshold in THRESHOLDS}
        for position, record in enumerate(selection_records, 1):
            ground_truth = read_mask(safe_under(project, record["mask"]))
            raw_e0 = load_probability(
                probability_path(inference_run, index, "e0", record["sample_id"]), ground_truth.shape
            )
            raw_e4 = load_probability(
                probability_path(inference_run, index, "e4", record["sample_id"]), ground_truth.shape
            )
            add_metrics(baseline_acc, record["source"], probability_metrics(raw_e0, ground_truth, 0.5))
            e0 = calibrate_probability(raw_e0, fitted["e0"]["temperature"])
            e4 = calibrate_probability(raw_e4, fitted["e4"]["temperature"])
            for alpha in ALPHAS:
                fused = ((1.0 - alpha) * e0 + alpha * e4).astype(np.float32)
                for threshold in THRESHOLDS:
                    add_metrics(
                        candidate_acc[(alpha, threshold)],
                        record["source"],
                        probability_metrics(fused, ground_truth, threshold),
                    )
            if position % 25 == 0 or position == len(selection_records):
                print(f"[search {position}/{len(selection_records)}] {record['sample_id']}", flush=True)

        baseline = finish_accumulator(baseline_acc)
        gates = dict(DEFAULT_GATES)
        candidates = []
        by_source_rows = []
        for alpha in ALPHAS:
            for threshold in THRESHOLDS:
                candidate = finish_accumulator(candidate_acc[(alpha, threshold)])
                candidate.update(alpha=alpha, threshold=threshold)
                gate_candidate(candidate, baseline, gates)
                candidates.append(candidate)
                for source, score in sorted(candidate["source_dice"].items()):
                    by_source_rows.append(
                        {
                            "alpha": alpha,
                            "threshold": threshold,
                            "source": source,
                            "e6_dice": score,
                            "e0_dice": baseline["source_dice"][source],
                            "delta_vs_e0": score - baseline["source_dice"][source],
                        }
                    )
        selected, status = choose_candidate(candidates)
        flat_rows = []
        for candidate in candidates:
            flat_rows.append(
                {key: value for key, value in candidate.items() if key not in ("source_dice", "checks")}
            )
        write_csv(output / "grid_search.csv", flat_rows)
        write_csv(output / "grid_by_source.csv", by_source_rows)
        write_json(output / "calibration.json", fitted)
        config = {
            "format_version": 1,
            "experiment": "E6_CALIBRATED_FIXED_FUSION",
            "validation_status": status,
            "selection_rule": (
                "Among gate-passing candidates maximize selection Dice, then worst-source delta, "
                "then minimize MAE, failure tail, alpha, and distance from threshold 0.5. "
                "If none pass, record nearest candidate but block formal test by default."
            ),
            "temperature_e0": fitted["e0"]["temperature"],
            "temperature_e4": fitted["e4"]["temperature"],
            "alpha_e4": selected["alpha"],
            "weight_e0": 1.0 - selected["alpha"],
            "threshold": selected["threshold"],
            "gates": gates,
            "selection_baseline_e0": baseline,
            "selected_validation_metrics": selected,
            "development_manifest_signature": manifest_info["manifest_signature"],
            "development_manifest_sha256": manifest_info["manifest_sha256"],
            "development_inference_run_info_sha256": file_hash(inference_run / "run_info.json"),
            "e0_checkpoint_sha256": inference_info["e0_checkpoint_sha256"],
            "e4_checkpoint_sha256": inference_info["e4_checkpoint_sha256"],
            "search_space": {"alpha_e4": list(ALPHAS), "threshold": list(THRESHOLDS)},
            "calibration_partition_count": len(calibration_records),
            "selection_partition_count": len(selection_records),
            "created_at": utc_now(),
        }
        config["config_signature"] = signature(config)
        write_json(output / "selection_config.json", config)
        info.update(
            {
                "status": "COMPLETED",
                "finished_at": utc_now(),
                "selection_status": status,
                "selected": {
                    "alpha_e4": selected["alpha"],
                    "threshold": selected["threshold"],
                    "temperature_e0": fitted["e0"]["temperature"],
                    "temperature_e4": fitted["e4"]["temperature"],
                },
                "config_signature": config["config_signature"],
            }
        )
        write_json(output / "run_info.json", info)
        print("Selection status:", status)
        print("Locked configuration:", output / "selection_config.json")
        if status != "PASS":
            print("No candidate passed the predeclared validation gates. Formal test is blocked.")
    except BaseException as error:
        info.update(status="FAILED", finished_at=utc_now(), error=f"{type(error).__name__}: {error}")
        write_json(output / "run_info.json", info)
        raise


def validate_test_sources(project, e0_run, e4_run, e4_eval):
    e0_info = read_json(e0_run / "run_info.json")
    e4_info = read_json(e4_run / "run_info.json")
    e4_eval_info = read_json(e4_eval / "run_info.json")
    if (
        e0_info.get("status") != "COMPLETED"
        or e0_info.get("run_kind") != "FULL_E0"
        or e0_info.get("processed_count") != 798
        or e0_info.get("protocol", {}).get("id") != "endolens-e0-probability-v1"
    ):
        raise ValueError("Invalid frozen E0 test run")
    checkpoint = e4_run / "checkpoints" / "final.pth"
    if (
        e4_info.get("status") != "TRAINED"
        or e4_info.get("run_kind") != "E2_FINAL"
        or file_hash(checkpoint) != e4_info.get("final_checkpoint_sha256")
    ):
        raise ValueError("Invalid frozen E4 model")
    if (
        e4_eval_info.get("status") != "COMPLETED"
        or e4_eval_info.get("processed_count") != 798
        or e4_eval_info.get("checkpoint_sha256") != e4_info.get("final_checkpoint_sha256")
        or e4_eval_info.get("protocol", {}).get("per_image_minmax") is not False
    ):
        raise ValueError("Invalid frozen E4 evaluation")
    if file_hash(e0_run / "manifest.json") != e0_info.get("manifest_sha256"):
        raise ValueError("E0 manifest changed")
    manifest = read_json(e0_run / "manifest.json")
    if manifest != read_json(e4_run / "test_manifest.json"):
        raise ValueError("E0/E4 frozen test manifests differ")
    if dict(Counter(r["dataset"] for r in manifest)) != TEST_COUNTS:
        raise ValueError("Frozen test counts changed")
    e0_rows = by_id(read_csv(e0_run / "per_image.csv"))
    e4_rows = by_id(read_csv(e4_eval / "per_image.csv"))
    if set(e0_rows) != {r["sample_id"] for r in manifest} or set(e4_rows) != set(e0_rows):
        raise ValueError("Frozen test per-image rows are incomplete")
    return manifest, e0_rows, e4_rows, e0_info, e4_info


def summarize_test(rows):
    result = []
    for dataset in ("ALL",) + DATASETS:
        group = rows if dataset == "ALL" else [row for row in rows if row["dataset"] == dataset]
        item = {"dataset": dataset, "count": len(group)}
        for model in ("e0", "e4", "e5", "e6"):
            for metric in ("dice", "iou", "mae"):
                item[f"{model}_{metric}"] = float(np.mean([r[f"{model}_{metric}"] for r in group]))
            dice = np.array([r[f"{model}_dice"] for r in group])
            item[f"{model}_median_dice"] = float(np.median(dice))
            item[f"{model}_dice_lt05"] = int((dice < 0.5).sum())
            item[f"{model}_zero_dice"] = int((dice == 0).sum())
        for reference in ("e0", "e5"):
            for metric in ("dice", "iou", "mae"):
                item[f"delta_{metric}_vs_{reference}"] = item[f"e6_{metric}"] - item[f"{reference}_{metric}"]
        result.append(item)
    return result


def paired_bootstrap(delta, seed=42, repetitions=20000):
    delta = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    batch = 1000
    for start in range(0, repetitions, batch):
        stop = min(repetitions, start + batch)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean_delta": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high), "repetitions": repetitions, "seed": seed}


def command_apply_test(args):
    project = Path(args.project).expanduser().resolve()
    config_path = resolve(project, args.config)
    output = resolve(project, args.output)
    e0_run = resolve(project, args.e0_run)
    e4_run = resolve(project, args.e4_run)
    e4_eval = resolve(project, args.e4_eval)
    if output.exists():
        raise FileExistsError("Output already exists: " + str(output))
    config = read_json(config_path)
    recorded_signature = config.pop("config_signature")
    if signature(config) != recorded_signature:
        raise ValueError("Locked selection config changed")
    config["config_signature"] = recorded_signature
    if config.get("validation_status") != "PASS" and not args.allow_exploratory:
        raise ValueError("Validation status is NO_PASS; formal test is blocked")
    output.mkdir(parents=True)
    info = {
        "status": "RUNNING",
        "run_kind": "FULL_E6" if config["validation_status"] == "PASS" else "EXPLORATORY_E6",
        "started_at": utc_now(),
        "config_signature": recorded_signature,
    }
    write_json(output / "run_info.json", info)
    try:
        manifest, e0_rows, e4_rows, e0_info, e4_info = validate_test_sources(project, e0_run, e4_run, e4_eval)
        if config["e0_checkpoint_sha256"] != e0_info["checkpoint_sha256"]:
            raise ValueError("Selection used a different E0 checkpoint")
        if config["e4_checkpoint_sha256"] != e4_info["final_checkpoint_sha256"]:
            raise ValueError("Selection used a different E4 checkpoint")
        rows = []
        fingerprints = []
        for position, record in enumerate(manifest, 1):
            sample_id = record["sample_id"]
            image_path = safe_under(project, record["image"])
            mask_path = safe_under(project, record["mask"])
            if file_hash(image_path) != record["image_sha256"] or file_hash(mask_path) != record["mask_sha256"]:
                raise ValueError("Frozen test file changed: " + sample_id)
            ground_truth = read_mask(mask_path)
            e0_path = safe_under(e0_run, e0_rows[sample_id]["probability_path"])
            e4_path = safe_under(e4_eval, e4_rows[sample_id]["probability_path"])
            raw_e0 = load_probability(e0_path, ground_truth.shape)
            raw_e4 = load_probability(e4_path, ground_truth.shape)
            e0_metrics = probability_metrics(raw_e0, ground_truth, 0.5)
            e4_metrics = probability_metrics(raw_e4, ground_truth, 0.5)
            e5_probability = (0.5 * raw_e0 + 0.5 * raw_e4).astype(np.float32)
            e5_metrics = probability_metrics(e5_probability, ground_truth, 0.5)
            calibrated_e0 = calibrate_probability(raw_e0, config["temperature_e0"])
            calibrated_e4 = calibrate_probability(raw_e4, config["temperature_e4"])
            alpha = config["alpha_e4"]
            e6_probability = ((1.0 - alpha) * calibrated_e0 + alpha * calibrated_e4).astype(np.float32)
            e6_metrics = probability_metrics(e6_probability, ground_truth, config["threshold"])
            probability_relative = Path("probabilities") / (sample_id + ".npz")
            mask_relative = Path("masks") / (sample_id + ".png")
            for relative in (probability_relative, mask_relative):
                (output / relative.parent).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output / probability_relative, probability=e6_probability)
            Image.fromarray((e6_probability >= config["threshold"]).astype(np.uint8) * 255).save(output / mask_relative)
            row = {
                "sample_id": sample_id,
                "dataset": record["dataset"],
                "height": ground_truth.shape[0],
                "width": ground_truth.shape[1],
                "gt_area_fraction": float(ground_truth.mean()),
            }
            for model, metrics in (("e0", e0_metrics), ("e4", e4_metrics), ("e5", e5_metrics), ("e6", e6_metrics)):
                row.update({f"{model}_{key}": value for key, value in metrics.items()})
            row.update(
                {
                    "delta_dice_vs_e0": e6_metrics["dice"] - e0_metrics["dice"],
                    "delta_dice_vs_e5": e6_metrics["dice"] - e5_metrics["dice"],
                    "probability_path": probability_relative.as_posix(),
                    "mask_path": mask_relative.as_posix(),
                }
            )
            rows.append(row)
            fingerprints.append(
                {
                    "sample_id": sample_id,
                    "e0_probability_sha256": file_hash(e0_path),
                    "e4_probability_sha256": file_hash(e4_path),
                    "e6_probability_sha256": file_hash(output / probability_relative),
                }
            )
            if position % 50 == 0 or position == len(manifest):
                print(f"[E6 test {position}/{len(manifest)}] {sample_id}", flush=True)
        write_csv(output / "per_image.csv", rows)
        write_csv(output / "input_fingerprints.csv", fingerprints)
        summary = summarize_test(rows)
        write_csv(output / "summary.csv", summary)
        all_summary = summary[0]
        gates = config["gates"]
        dataset_deltas = {
            row["dataset"]: row["delta_dice_vs_e0"] for row in summary if row["dataset"] != "ALL"
        }
        checks = {
            "mean_dice_gain": all_summary["delta_dice_vs_e0"] >= gates["min_mean_dice_gain"],
            "all_datasets": min(dataset_deltas.values()) >= gates["min_source_dice_delta"],
            "mae": all_summary["delta_mae_vs_e0"] <= gates["max_mae_increase"],
            "failure_tail": all_summary["e6_dice_lt05"] - all_summary["e0_dice_lt05"] <= gates["max_dice_lt05_increase"],
        }
        decision = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "gates": gates,
            "dataset_dice_deltas": dataset_deltas,
            "paired_bootstrap": {
                "e6_minus_e0_dice": paired_bootstrap([r["e6_dice"] - r["e0_dice"] for r in rows]),
                "e6_minus_e5_dice": paired_bootstrap([r["e6_dice"] - r["e5_dice"] for r in rows]),
            },
            "interpretation": (
                "PASS means the predeclared held-out test gates were met. FAIL does not authorize "
                "retuning on the test set; return to a new development hypothesis instead."
            ),
        }
        write_json(output / "decision.json", decision)
        write_json(output / "locked_config.json", config)
        write_json(output / "manifest.json", manifest)
        info.update(
            {
                "status": "COMPLETED",
                "finished_at": utc_now(),
                "processed_count": len(rows),
                "decision": decision["status"],
                "e0_checkpoint_sha256": e0_info["checkpoint_sha256"],
                "e4_checkpoint_sha256": e4_info["final_checkpoint_sha256"],
                "test_manifest_sha256": file_hash(e0_run / "manifest.json"),
                "timing": "Offline probability calibration/fusion/audit/output; not neural inference FPS",
            }
        )
        write_json(output / "run_info.json", info)
        print("FULL_E6 complete. Decision:", decision["status"])
        print("Output:", output)
    except BaseException as error:
        info.update(status="FAILED", finished_at=utc_now(), error=f"{type(error).__name__}: {error}")
        write_json(output / "run_info.json", info)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parent.parent))
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset", default="dataset/E6Development")
    prepare.add_argument("--metadata")
    prepare.add_argument("--output", default="runs/e6_development_seed42")
    prepare.add_argument("--e0-run", default="runs/e0_20261004_221345_872444")
    prepare.add_argument("--e4-run", default="runs/e4_final_seed42")
    prepare.add_argument("--calibration-fraction", type=float, default=0.4)
    prepare.add_argument("--seed", type=int, default=42)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--manifest-run", default="runs/e6_development_seed42")
    infer.add_argument("--output", required=True)
    infer.add_argument("--e0-run", default="runs/e0_20261004_221345_872444")
    infer.add_argument("--e0-checkpoint", default="model_pth/PolypPVT.pth")
    infer.add_argument("--e4-run", default="runs/e4_final_seed42")
    infer.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    infer.add_argument("--threads", type=int, default=4)
    infer.add_argument("--smoke", action="store_true")

    select = subparsers.add_parser("select")
    select.add_argument("--manifest-run", default="runs/e6_development_seed42")
    select.add_argument("--inference-run", default="runs/e6_development_inference_seed42")
    select.add_argument("--output", default="runs/e6_selection_seed42")
    select.add_argument("--seed", type=int, default=42)
    select.add_argument("--max-pixels-per-image", type=int, default=8192)

    apply_test = subparsers.add_parser("apply-test")
    apply_test.add_argument("--config", default="runs/e6_selection_seed42/selection_config.json")
    apply_test.add_argument("--output", default="runs/e6_final_seed42")
    apply_test.add_argument("--e0-run", default="runs/e0_20261004_221345_872444")
    apply_test.add_argument("--e4-run", default="runs/e4_final_seed42")
    apply_test.add_argument(
        "--e4-eval", default="runs/e4_final_seed42/evaluation_cuda_20261006_235221_022843"
    )
    apply_test.add_argument("--allow-exploratory", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "seed") and args.seed < 0:
        parser.error("seed must be nonnegative")
    if hasattr(args, "threads") and args.threads < 1:
        parser.error("threads must be positive")
    if hasattr(args, "max_pixels_per_image") and args.max_pixels_per_image < 256:
        parser.error("max-pixels-per-image must be at least 256")
    {
        "prepare": command_prepare,
        "infer": command_infer,
        "select": command_select,
        "apply-test": command_apply_test,
    }[args.command](args)


if __name__ == "__main__":
    main()
