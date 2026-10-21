"""E8 shared provenance and immutable-bundle helpers (no model training)."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
CORE = PROJECT / "e4_windows/core"
sys.path.insert(0, str(CORE))
from e0_protocol import file_hash, read_mask, write_csv
from data import signature

E0_RUN = "runs/e0_20261004_221345_872444"
BUNDLE = "runs/e8_data_seed42"
ARMS = ("A_baseline", "B_expanded", "C_hard", "D_expanded_hard")
RUNTIME = ("common.py", "prepare_data.py", "train_e8.py", "run_e8.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def code_hashes():
    files = [PACKAGE / name for name in RUNTIME]
    files += list(CORE.glob("*.py"))
    files += [PROJECT / "e6_windows/pvt_timm_compat.py", PROJECT / "lib/pvt.py", PROJECT / "lib/pvtv2.py"]
    return {p.relative_to(PROJECT).as_posix(): file_hash(p) for p in sorted(files)}


def frozen_e0():
    info = read_json(PROJECT / E0_RUN / "run_info.json")
    paths = {"model_pth/PolypPVT.pth": info["checkpoint_sha256"],
             E0_RUN + "/manifest.json": info["manifest_sha256"]}
    for name, expected in info["source_hashes"].items():
        current = file_hash(PROJECT / name)
        normalized = hashlib.sha256((PROJECT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if current != expected and normalized != expected:
            raise ValueError("E0 source differs beyond Windows line endings: " + name)
        # Freeze actual Windows bytes from now on, while verifying the original
        # macOS identity above. No E0 source file is edited.
        paths[name] = current
    paths[E0_RUN + "/run_info.json"] = file_hash(PROJECT / E0_RUN / "run_info.json")
    paths[E0_RUN + "/per_image.csv"] = file_hash(PROJECT / E0_RUN / "per_image.csv")
    for name, expected in paths.items():
        if file_hash(PROJECT / name) != expected:
            raise ValueError("E0 frozen artifact changed: " + name)
    return paths


def load_bundle(verify_content=True):
    folder = PROJECT / BUNDLE
    info = read_json(folder / "bundle.json")
    plain = {k: v for k, v in info.items() if k != "bundle_signature"}
    if signature(plain) != info["bundle_signature"]:
        raise ValueError("E8 bundle descriptor changed")
    for name, expected in info["frozen_e0"].items():
        if file_hash(PROJECT / name) != expected:
            raise ValueError("E0 changed: " + name)
    manifests = {}
    for role, descriptor in info["files"].items():
        path = folder / descriptor["filename"]
        if file_hash(path) != descriptor["sha256"]:
            raise ValueError("E8 manifest changed: " + role)
        manifests[role] = read_json(path)
    if verify_content:
        unique = {r["image"]: r for rows in manifests.values() for r in rows}
        for r in unique.values():
            for kind in ("image", "mask"):
                if file_hash(PROJECT / r[kind]) != r[kind + "_sha256"]:
                    raise ValueError("Data bytes changed: " + r["sample_id"])
    train = manifests["original_train"] + manifests["new_train"]
    validation = manifests["validation"]
    holdout = manifests["holdout"]
    for left, right in ((train, validation), (train, holdout), (validation, holdout)):
        if {r["sample_id"] for r in left} & {r["sample_id"] for r in right}:
            raise ValueError("Partition identity overlap")
        if {r["decoded_rgb_sha256"] for r in left} & {r["decoded_rgb_sha256"] for r in right}:
            raise ValueError("Partition exact-content overlap")
    if not all(manifests[k] for k in ("original_train", "new_train", "validation", "holdout")):
        raise ValueError("Empty E8 partition")
    return info, manifests


def metric_summary(rows):
    import numpy as np
    if not rows:
        return {"count": 0}
    dice = np.array([float(r["dice"]) for r in rows])
    return {"count": len(rows), "dice": float(dice.mean()), "iou": float(np.mean([float(r["iou"]) for r in rows])),
            "mae": float(np.mean([float(r["mae"]) for r in rows])), "median_dice": float(np.median(dice)),
            "dice_p10": float(np.quantile(dice, .1)), "dice_lt05": int((dice < .5).sum()),
            "dice_lt01": int((dice < .1).sum()), "zero_dice": int((dice == 0).sum())}


def summarize_groups(rows):
    groups = [{"group": "ALL", **metric_summary(rows)}]
    for name in sorted({r["dataset"] for r in rows}):
        groups.append({"group": name, **metric_summary([r for r in rows if r["dataset"] == name])})
    for name, condition in (("small_lt1pct", lambda r: float(r["gt_area_fraction"]) < .01),
                            ("medium_1to10pct", lambda r: .01 <= float(r["gt_area_fraction"]) < .1),
                            ("large_ge10pct", lambda r: float(r["gt_area_fraction"]) >= .1)):
        subset = [r for r in rows if condition(r)]
        if subset:
            groups.append({"group": name, **metric_summary(subset)})
    return groups


def validation_score(rows):
    # Fixed equal weight of old internal validation and new held-out centre C5.
    sources = ("original_validation", "PolypGen-C5")
    scores = []
    for source in sources:
        subset = [r for r in rows if r["dataset"] == source]
        if not subset:
            raise ValueError("Missing validation stratum: " + source)
        scores.append(metric_summary(subset)["dice"])
    return sum(scores) / len(scores)
