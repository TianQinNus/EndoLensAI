"""Audit original PolypGen masks; freeze centre-isolated E8 partitions.

C1-C4 train, C5 validate, C6 locked holdout. Only positive single frames are
used initially: sequence centre/patient correspondence is not guessed.
All original masks are preserved at the E0 threshold; no tiny components are
removed and no negative labels are synthesized. Near-duplicate quarantine
is conservative, image-only, and applied before model fitting.
"""
import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from common import (PROJECT, BUNDLE, E0_RUN, read_json, write_json, file_hash,
                    read_mask, signature, write_csv, frozen_e0)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NEAR_HAMMING = 2
NEAR_RGB_MAE = .01


def fingerprint(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
        gray = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
        dhash = np.packbits((gray[:, 1:] > gray[:, :-1]).reshape(-1)).tobytes().hex()
        thumb = np.asarray(rgb.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.uint8)
    return array, dhash, thumb


def make_record(image, mask, dataset, group_id):
    rgb, dhash, thumb = fingerprint(image)
    gt = read_mask(mask)
    if gt.shape != rgb.shape[:2]:
        raise ValueError("Image/mask size mismatch: " + str(image))
    record = {"sample_id": dataset + "/" + image.stem, "dataset": dataset,
              "source_stem": image.stem, "group_id": group_id,
              "image": image.relative_to(PROJECT).as_posix(), "mask": mask.relative_to(PROJECT).as_posix(),
              "height": gt.shape[0], "width": gt.shape[1], "gt_foreground_pixels": int(gt.sum()),
              "gt_area_fraction": float(gt.mean()), "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
              "image_sha256": file_hash(image), "mask_sha256": file_hash(mask), "dhash64": dhash}
    return record, thumb


def near_matches(record, thumb, records, thumbs):
    if not records:
        return []
    # Hash prefilter is fast; RGB comparison prevents hash-only false alarms.
    value = int(record["dhash64"], 16)
    candidates = [i for i, r in enumerate(records)
                  if (value ^ int(r["dhash64"], 16)).bit_count() <= NEAR_HAMMING]
    matches = []
    for i in candidates:
        r = records[i]
        if abs(record["width"] / record["height"] - r["width"] / r["height"]) > .01:
            continue
        distance = float(np.abs(thumb.astype(np.float32) - thumbs[i].astype(np.float32)).mean() / 255)
        if record["decoded_rgb_sha256"] == r["decoded_rgb_sha256"] or distance <= NEAR_RGB_MAE:
            matches.append((r["sample_id"], distance))
    return matches


def discover_single_frames(root):
    pairs, skipped = [], []
    # Supported official v3 layout: data_Cn/images_Cn, data_Cn/masks_Cn.
    for centre in range(1, 7):
        folders = [p for p in root.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts
                   and re.fullmatch(r"data[-_]C" + str(centre), p.name, re.I)]
        if len(folders) != 1:
            raise ValueError(f"Expected one data_C{centre} directory; found {folders}")
        folder = folders[0]
        image_dirs = [p for p in folder.iterdir() if p.is_dir() and p.name.lower().startswith("images")
                      and "box" not in p.name.lower()]
        mask_dirs = [p for p in folder.iterdir() if p.is_dir() and p.name.lower().startswith("masks")]
        if len(image_dirs) != 1 or len(mask_dirs) != 1:
            raise ValueError("Ambiguous official image/mask folders: " + str(folder))
        images = {p.stem: p for p in image_dirs[0].iterdir() if p.suffix.lower() in EXTENSIONS}
        masks = {p.stem: p for p in mask_dirs[0].iterdir() if p.suffix.lower() in EXTENSIONS}
        if len(images) != len(list(p for p in image_dirs[0].iterdir() if p.suffix.lower() in EXTENSIONS)):
            raise ValueError("Duplicate image stem")
        # Official masks often append _mask; normalize this suffix only.
        if images.keys() != masks.keys():
            normalized = {re.sub(r"_mask$", "", k, flags=re.I): v for k, v in masks.items()}
            # One literal author-archive filename typo; preserve both original
            # files and record the explicit correction in bundle provenance.
            if centre == 3 and "C3_EndoCV2021_00489]" in images and "C3_EndoCV2021_00489" in normalized:
                normalized["C3_EndoCV2021_00489]"] = normalized.pop("C3_EndoCV2021_00489")
            if len(normalized) != len(masks) or images.keys() != normalized.keys():
                raise ValueError("Unpaired official masks: " + str(folder))
            masks = normalized
        for stem in sorted(images):
            pairs.append((centre, images[stem], masks[stem]))
    return pairs


def prepare(root, output):
    if output.exists():
        raise FileExistsError("E8 data bundle exists; preserve it instead of changing the split")
    provenance = read_json(PROJECT / "dataset/_downloads/PolypGen-v3/download_provenance.json")
    provenance["license"] = "Archive license.txt: CC BY-NC-SA 4.0; README/paper: CC BY 4.0. Discrepancy recorded; non-commercial research only, retain attribution and share-alike restrictions for derived data; seek author clarification before redistribution/commercial use."
    if provenance["bytes"] != 3419253506:
        raise ValueError("Unrecognized author archive")
    frozen = frozen_e0()
    old = {k: read_json(PROJECT / "runs/e2_data_seed42" / (k + ".json"))
           for k in ("dev_train", "validation", "test")}
    far = read_json(PROJECT / "runs/e6_development_seed42/development_manifest.json")
    cache, thumbs = {}, {}
    for source in old["dev_train"] + old["validation"] + old["test"] + far:
        r = dict(source)
        rgb, dhash, thumb = fingerprint(PROJECT / r["image"])
        if hashlib.sha256(rgb.tobytes()).hexdigest() != r["decoded_rgb_sha256"]:
            raise ValueError("Protected data decoded bytes changed: " + r["sample_id"])
        for kind in ("image", "mask"):
            if file_hash(PROJECT / r[kind]) != r[kind + "_sha256"]:
                raise ValueError("Protected data file changed")
        r["dhash64"] = dhash
        r["dataset"] = "original_validation" if source in old["validation"] else "original_training"
        cache[r["sample_id"]], thumbs[r["sample_id"]] = r, thumb
    quarantine = []
    protected = [cache[r["sample_id"]] for r in old["validation"] + old["test"] + far]
    protected_thumbs = [thumbs[r["sample_id"]] for r in protected]
    original_train = []
    for r in old["dev_train"]:
        record = cache[r["sample_id"]]
        matches = near_matches(record, thumbs[r["sample_id"]], protected, protected_thumbs)
        if matches:
            quarantine.append({"sample_id": r["sample_id"], "reason": "original_train_near_protected", "matches": matches})
        else:
            original_train.append(record)
    original_val = [cache[r["sample_id"]] for r in old["validation"]]
    # All old partitions and FAR remain protected against new-data contamination.
    protected = list(cache.values())
    protected_thumbs = [thumbs[r["sample_id"]] for r in protected]
    new, new_thumbs = [], []
    pairs = discover_single_frames(root)
    raw_counts = Counter()
    for index, (centre, image, mask) in enumerate(pairs, 1):
        dataset = f"PolypGen-C{centre}"
        r, thumb = make_record(image, mask, dataset, f"PolypGen/centre-C{centre}")
        raw_counts[dataset] += 1
        if r["gt_foreground_pixels"] == 0:
            quarantine.append({"sample_id": r["sample_id"], "reason": "empty_gt_excluded_from_positive_only_stage", "matches": []})
            continue
        matches = near_matches(r, thumb, protected, protected_thumbs)
        if matches:
            quarantine.append({"sample_id": r["sample_id"], "reason": "new_near_old_or_FAR", "matches": matches})
            continue
        new.append(r)
        new_thumbs.append(thumb)
        if index % 200 == 0:
            print(f"[audit] original single frames {index}/{len(pairs)}", flush=True)
    # Prioritize holdout, then validation, then training when near duplicates
    # occur across centres. Within-centre dedup also reduces repeated content.
    order = sorted(range(len(new)), key=lambda i: (-int(new[i]["dataset"][-1]), new[i]["sample_id"]))
    kept, kept_thumbs = [], []
    for i in order:
        r = new[i]
        matches = near_matches(r, new_thumbs[i], kept, kept_thumbs)
        if matches:
            quarantine.append({"sample_id": r["sample_id"], "reason": "new_near_duplicate_quarantine", "matches": matches})
        else:
            kept.append(r)
            kept_thumbs.append(new_thumbs[i])
    manifests = {
        "original_train": sorted(original_train, key=lambda r: r["sample_id"]),
        "new_train": sorted([r for r in kept if r["dataset"] in {f"PolypGen-C{i}" for i in range(1, 5)}], key=lambda r: r["sample_id"]),
        "validation": sorted(original_val + [r for r in kept if r["dataset"] == "PolypGen-C5"], key=lambda r: r["sample_id"]),
        "holdout": sorted([r for r in kept if r["dataset"] == "PolypGen-C6"], key=lambda r: r["sample_id"]),
    }
    if not all(manifests.values()):
        raise ValueError("An E8 partition is empty after auditing")
    output.mkdir(parents=True)
    write_json(output / "quarantine.json", quarantine)
    descriptor = {"stage": "E8_PRETRAINED_UNET_DATA_AND_HARD_SAMPLING", "split_seed": 42,
                  "new_data": "PolypGen v3 original single frames, positive masks only",
                  "centre_roles": {"train": ["C1", "C2", "C3", "C4"], "validation": ["C5"], "holdout": ["C6"]},
                  "raw_single_frame_counts": dict(raw_counts), "counts": {k: len(v) for k, v in manifests.items()},
                  "source_pairing_notes": ["Official C3_EndoCV2021_00489].jpg matched to C3_EndoCV2021_00489_mask.jpg; literal trailing-bracket filename typo, original files unchanged."],
                  "strata": {k: dict(Counter(r["dataset"] for r in v)) for k, v in manifests.items()},
                  "quarantine_reasons": dict(Counter(r["reason"] for r in quarantine)),
                  "near_duplicate_rule": {"dhash64_hamming_max": NEAR_HAMMING, "rgb16_mean_absolute_difference_max": NEAR_RGB_MAE,
                                          "aspect_ratio_absolute_difference_max": .01},
                  "download_provenance": provenance, "frozen_e0": frozen,
                  "old_manifest_hashes": {k: file_hash(PROJECT / "runs/e2_data_seed42" / (k + ".json")) for k in old},
                  "validation_selection": "0.5*old validation mean Dice + 0.5*C5 mean Dice; earlier checkpoint breaks ties",
                  "limitations": ["Old split lacks reliable patient identities; image split is not patient-independent.",
                                  "E0 pvtv2.py differs from the macOS source hash only by CRLF line endings; LF identity verified and current Windows bytes frozen.",
                                  "New partitions are whole-centre isolated; no inferred per-patient identities are claimed.",
                                  "Near-duplicate heuristic is conservative, not proof of complete absence of related frames.",
                                  "Sequence frames and negatives intentionally excluded from first-stage training.",
                                  "C6 is E8-uninspected, not necessarily unseen by public-model authors or prior publications.",
                                  "Historical 798 and FAR were previously inspected and cannot be called fresh blind tests."],
                  "files": {}}
    for role, records in manifests.items():
        write_json(output / (role + ".json"), records)
        descriptor["files"][role] = {"filename": role + ".json", "sha256": file_hash(output / (role + ".json"))}
    descriptor["bundle_signature"] = signature(descriptor)
    write_json(output / "bundle.json", descriptor)
    print("E8_DATA_READY", descriptor["counts"], descriptor["strata"], descriptor["quarantine_reasons"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "dataset/_downloads/PolypGen-v3/extracted")
    args = parser.parse_args()
    prepare(args.source.resolve(), PROJECT / BUNDLE)
