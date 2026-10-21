"""E8 Windows CLI: matched controls, frozen E0, validation-only model choice."""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from common import (PROJECT, PACKAGE, CORE, BUNDLE, E0_RUN, ARMS, read_json, write_json,
                    file_hash, write_csv, code_hashes, load_bundle, metric_summary,
                    summarize_groups, validation_score, signature)
from train_e8 import make_config, train
from models import make_model, read_pretrained
from engine import require_cuda, evaluate_records, utc

PROTOCOL_PATH = PROJECT / BUNDLE / "protocol.json"
SEEDS = (42, 17, 2024)


def run_path(arm, seed, smoke=False):
    return PROJECT / "runs" / (f"e8_smoke_{arm}_seed{seed}" if smoke else f"e8_{arm}_seed{seed}")


def freeze_protocol():
    info, _ = load_bundle()
    _, weight_hash = read_pretrained(PROJECT / "e2_weights/resnet34-b627a593.pth")
    e0_info = read_json(PROJECT / E0_RUN / "run_info.json")
    backbone_hash = file_hash(PROJECT / "pretrained_pth/pvt_v2_b2.pth")
    if backbone_hash != e0_info["backbone_sha256"]:
        raise ValueError("E0 backbone initializer changed")
    protocol = {"stage": "E8", "created_utc": utc(), "bundle_signature": info["bundle_signature"],
                "source_hashes": code_hashes(), "initial_encoder_sha256": weight_hash,
                "e0_backbone_sha256": backbone_hash,
                "seeds": list(SEEDS), "screening_seed": 42,
                "configs": {f"{arm}/seed{seed}": make_config(arm, seed) for arm in ARMS for seed in SEEDS},
                "primary_contrasts": ["B-A: data expansion policy", "C-A: hard sampling", "D-B: hard sampling with expanded data", "D-C: expansion with hard sampling"],
                "success_gates": {"min_validation_macro_dice_gain_vs_A": .003, "max_each_validation_stratum_mae_increase": .003,
                                  "max_each_validation_stratum_dice_lt05_increase": 0},
                "selection": "Highest validation macro Dice among B/C/D passing gates; fixed arm order breaks ties; otherwise retain A",
                "holdout": "C6; single locked evaluation of all four seed42 checkpoints and frozen E0; no holdout tuning",
                "confirmatory_note": "Seed42 screening is exploratory; additional seeds and broader centre evaluation needed for robust claims.",
                "budget_note": "Equal image draws and planned updates, not full-dataset epochs. AMP skipped steps reported separately.",
                "new_stage_note": "These are E8 U-Net-vs-A development criteria, not a revision of E7 fusion gates or outcomes."}
    protocol["protocol_signature"] = signature(protocol)
    if PROTOCOL_PATH.exists():
        raise FileExistsError("Protocol exists; use preflight, never overwrite the frozen protocol")
    write_json(PROTOCOL_PATH, protocol)
    print("E8_PROTOCOL_LOCKED", PROTOCOL_PATH)


def preflight(verify_content=True):
    info, manifests = load_bundle(verify_content)
    protocol = read_json(PROTOCOL_PATH)
    if signature({k: v for k, v in protocol.items() if k != "protocol_signature"}) != protocol["protocol_signature"]:
        raise ValueError("Protocol signature mismatch")
    if protocol["source_hashes"] != code_hashes() or protocol["bundle_signature"] != info["bundle_signature"]:
        raise ValueError("Code/bundle no longer matches frozen E8 protocol")
    _, digest = read_pretrained(PROJECT / "e2_weights/resnet34-b627a593.pth")
    if digest != protocol["initial_encoder_sha256"]:
        raise ValueError("ImageNet initialization changed")
    if file_hash(PROJECT / "pretrained_pth/pvt_v2_b2.pth") != protocol["e0_backbone_sha256"]:
        raise ValueError("Frozen E0 backbone file changed")
    for arm in ARMS:
        for seed in SEEDS:
            if make_config(arm, seed) != protocol["configs"][f"{arm}/seed{seed}"]:
                raise ValueError("Experiment configuration changed")
    print("PREFLIGHT_PASS", info["counts"], "E0_FROZEN; NO_HOLDOUT_METRICS_READ", flush=True)
    return info, manifests, protocol


def completed_run(arm, seed, protocol, bundle_info):
    path = run_path(arm, seed)
    info = read_json(path / "run_info.json")
    if info["status"] != "TRAINED" or info["run_kind"] != "E8_DEVELOPMENT":
        raise ValueError("Require completed formal development run: " + str(path))
    if info["config"] != protocol["configs"][f"{arm}/seed{seed}"] or info["source_hashes"] != protocol["source_hashes"]:
        raise ValueError("Run does not match frozen protocol")
    if info["bundle_signature"] != bundle_info["bundle_signature"]:
        raise ValueError("Run data mismatch")
    cp = path / "checkpoints/best.pth"
    if file_hash(cp) != info["best_checkpoint_sha256"]:
        raise ValueError("Best checkpoint changed")
    weights = torch.load(cp, map_location="cpu", weights_only=True)
    if (weights["config"] != info["config"] or weights["epoch"] != info["best_epoch"]
            or weights["source_hashes"] != protocol["source_hashes"]
            or weights["bundle_signature"] != info["bundle_signature"]):
        raise ValueError("Best checkpoint metadata mismatch")
    del weights
    return path, info


def development_gates(result, base, rows, base_rows):
    gain = result["macro_dice"] - base["macro_dice"]
    gates = {"macro_dice": gain >= .003}
    for source in ("original_validation", "PolypGen-C5"):
        metrics = metric_summary([r for r in rows if r["dataset"] == source])
        baseline = metric_summary([r for r in base_rows if r["dataset"] == source])
        gates[source + "_mae"] = metrics["mae"] - baseline["mae"] <= .003
        gates[source + "_low_dice"] = metrics["dice_lt05"] <= baseline["dice_lt05"]
    return gain, gates


def compare(seed):
    info, manifests, protocol = preflight()
    folder = PROJECT / "runs" / f"e8_validation_comparison_seed{seed}"
    if folder.exists():
        raise FileExistsError("Comparison exists; preserve the first comparison")
    results, grouped, per_image = [], [], {}
    expected_ids = {r["sample_id"] for r in manifests["validation"]}
    for arm in ARMS:
        path, run_info = completed_run(arm, seed, protocol, info)
        if file_hash(path / "validation_best_per_image.csv") != run_info["best_validation_csv_sha256"]:
            raise ValueError("Best validation per-image artifact changed")
        with (path / "validation_best_per_image.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(expected_ids) or {r["sample_id"] for r in rows} != expected_ids:
            raise ValueError("Validation rows incomplete/duplicated")
        if not np.isclose(validation_score(rows), run_info["best_validation_macro_dice"], atol=1e-12, rtol=0):
            raise ValueError("Validation summary mismatch")
        per_image[arm] = rows
        results.append({"arm": arm, "seed": seed, "best_round": run_info["best_epoch"],
                        "macro_dice": validation_score(rows), **metric_summary(rows),
                        "optimizer_updates_total": run_info["optimizer_updates_total"],
                        "amp_skipped_updates_total": run_info["amp_skipped_updates_total"]})
        grouped.extend({"arm": arm, **r} for r in summarize_groups(rows))
    base = results[0]
    gate_rows, eligible = [], []
    for result in results[1:]:
        arm = result["arm"]
        gain, gates = development_gates(result, base, per_image[arm], per_image[ARMS[0]])
        passed = all(gates.values())
        gate_rows.append({"arm": arm, "macro_dice_gain_vs_A": gain, "development_pass": passed, "gates": gates})
        if passed:
            eligible.append(result)
    selected = sorted(eligible, key=lambda r: (-r["macro_dice"], ARMS.index(r["arm"])))[0]["arm"] if eligible else ARMS[0]
    folder.mkdir()
    write_csv(folder / "summary.csv", results)
    write_csv(folder / "groups.csv", grouped)
    decision = {"stage": "E8_VALIDATION_ONLY", "seed": seed, "selected_arm": selected,
                "status": "DEVELOPMENT_PASS" if eligible else "NO_DEVELOPMENT_PASS_RETAIN_A",
                "results": results, "gates": gate_rows, "protocol_signature": protocol["protocol_signature"],
                "bundle_signature": info["bundle_signature"], "holdout_metrics_used": False,
                "limitation": "Model choice and reported gains use development validation; not external evidence."}
    write_json(folder / "decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


def lock_holdout(seed):
    if seed != 42:
        raise ValueError("First locked screening holdout is predeclared for seed42 only")
    info, _, protocol = preflight()
    decision_path = PROJECT / "runs" / f"e8_validation_comparison_seed{seed}" / "decision.json"
    decision = read_json(decision_path)
    if decision["protocol_signature"] != protocol["protocol_signature"] or decision["bundle_signature"] != info["bundle_signature"]:
        raise ValueError("Validation decision provenance mismatch")
    path = PROJECT / "runs/e8_holdout_lock_seed42.json"
    if path.exists():
        raise FileExistsError("Holdout configuration already locked")
    models = {}
    for arm in ARMS:
        folder, run_info = completed_run(arm, seed, protocol, info)
        checkpoint = folder / "checkpoints/best.pth"
        models[arm] = {"checkpoint": checkpoint.relative_to(PROJECT).as_posix(),
                       "sha256": run_info["best_checkpoint_sha256"], "epoch": run_info["best_epoch"]}
    models["E0_frozen_reference"] = {"checkpoint": "model_pth/PolypPVT.pth", "sha256": info["frozen_e0"]["model_pth/PolypPVT.pth"]}
    lock = {"created_utc": utc(), "seed": seed, "selected_arm_from_validation": decision["selected_arm"],
            "validation_decision_sha256": file_hash(decision_path), "models": models,
            "protocol_signature": protocol["protocol_signature"], "bundle_signature": info["bundle_signature"],
            "holdout_manifest_sha256": info["files"]["holdout"]["sha256"],
            "evaluation_policy": "All four prespecified arms and E0; no reselection/tuning on C6"}
    lock["lock_signature"] = signature(lock)
    write_json(path, lock)
    print("HOLDOUT_LOCKED", path, "Formal evaluation has NOT been performed.")


class E0Adapter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        p1, p2 = self.model(x)
        return p1 + p2


def evaluate_holdout(resume=False):
    info, manifests, protocol = preflight()
    lock_path = PROJECT / "runs/e8_holdout_lock_seed42.json"
    lock = read_json(lock_path)
    if signature({k: v for k, v in lock.items() if k != "lock_signature"}) != lock["lock_signature"]:
        raise ValueError("Holdout lock changed")
    if lock["protocol_signature"] != protocol["protocol_signature"] or lock["bundle_signature"] != info["bundle_signature"]:
        raise ValueError("Holdout lock no longer matches protocol/data")
    decision_path = PROJECT / "runs/e8_validation_comparison_seed42/decision.json"
    if file_hash(decision_path) != lock["validation_decision_sha256"]:
        raise ValueError("Validation decision changed after lock")
    for descriptor in lock["models"].values():
        if file_hash(PROJECT / descriptor["checkpoint"]) != descriptor["sha256"]:
            raise ValueError("Locked model bytes changed")
    output = PROJECT / "runs/e8_holdout_seed42"
    if resume:
        status = read_json(output / "run_info.json")
        if status["status"] == "COMPLETED" or status["lock_sha256"] != file_hash(lock_path):
            raise ValueError("Cannot rerun completed/changed holdout evaluation")
    else:
        output.mkdir(exist_ok=False)
        status = {"status": "RUNNING", "started_utc": utc(), "lock_sha256": file_hash(lock_path), "completed_models": []}
    write_json(output / "run_info.json", status)
    device = require_cuda()
    results = []
    try:
        for name, descriptor in lock["models"].items():
            model_output = output / name
            if name in status["completed_models"]:
                saved = read_json(model_output / "metrics.json")
                results.append({"model": name, **saved["summary"]})
                continue
            model_output.mkdir(exist_ok=resume)
            if name == "E0_frozen_reference":
                sys.path.insert(0, str(PROJECT / "e6_windows"))
                from pvt_timm_compat import install_if_missing
                install_if_missing()
                sys.path.insert(0, str(PROJECT))
                from lib.pvt import PolypPVT
                base = PolypPVT()
                base.load_state_dict(torch.load(PROJECT / descriptor["checkpoint"], map_location="cpu", weights_only=True), strict=True)
                model = E0Adapter(base).to(device).eval()
            else:
                checkpoint = torch.load(PROJECT / descriptor["checkpoint"], map_location="cpu", weights_only=True)
                model = make_model(checkpoint["config"], initialization=False)
                model.load_state_dict(checkpoint["model_state"], strict=True)
                model.to(device).eval()
                del checkpoint
            _, rows = evaluate_records(model, PROJECT, manifests["holdout"], device, 352, model_output, warmup=3)
            summary = metric_summary(rows)
            write_json(model_output / "metrics.json", {"summary": summary, "checkpoint_sha256": descriptor["sha256"],
                                                       "groups": summarize_groups(rows), "formal_role": "prelocked C6 holdout"})
            write_csv(model_output / "groups.csv", summarize_groups(rows))
            results.append({"model": name, **summary})
            status["completed_models"].append(name)
            write_json(output / "run_info.json", status)
            del model
            if name == "E0_frozen_reference":
                del base
            torch.cuda.empty_cache()
        write_csv(output / "summary.csv", results)
        baseline = next(r for r in results if r["model"] == ARMS[0])
        contrasts = [{"model": r["model"], "dice_delta_vs_A": r["dice"]-baseline["dice"],
                      "mae_delta_vs_A": r["mae"]-baseline["mae"], "low_dice_delta_vs_A": r["dice_lt05"]-baseline["dice_lt05"]}
                     for r in results]
        write_csv(output / "contrasts_vs_A.csv", contrasts)
        status.update(status="COMPLETED", finished_utc=utc(), selected_arm_from_validation=lock["selected_arm_from_validation"],
                      note="No holdout model reselection; C6 single-centre small sample, patient-level CI not claimed.")
        write_json(output / "run_info.json", status)
        print("LOCKED_HOLDOUT_COMPLETE", output)
    except BaseException as error:
        status.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
        write_json(output / "run_info.json", status)
        raise


def main():
    # The original PVT constructor resolves its backbone relative to cwd.
    os.chdir(PROJECT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("lock-protocol", "preflight", "check-gpu", "smoke", "train", "resume", "compare", "lock-holdout", "evaluate-holdout"))
    parser.add_argument("--arm", choices=ARMS, default=ARMS[0])
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume-evaluation", action="store_true")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    if args.command == "lock-protocol":
        freeze_protocol()
    elif args.command == "preflight":
        preflight()
    elif args.command == "check-gpu":
        preflight()
        output = PROJECT / "runs/e8_gpu_device_proof.json"
        if output.exists():
            saved = read_json(output)
            if saved.get("passed") and all(str(saved.get(k, "")).startswith("cuda") for k in ("model_device", "input_device", "output_device")):
                print("GPU_PROOF_ALREADY_PRESENT", output)
                return
            raise ValueError("Existing GPU proof did not pass; inspect it")
        subprocess.run([sys.executable, str(CORE / "check_device.py"), "--device", "cuda", "--output", str(output)], check=True, cwd=PROJECT)
    elif args.command in ("smoke", "train", "resume"):
        info, manifests, _ = preflight()
        smoke = args.command == "smoke"
        smoke_resume = False
        if smoke and run_path(args.arm, args.seed, True).exists():
            saved = read_json(run_path(args.arm, args.seed, True) / "run_info.json")
            if saved["config"] != make_config(args.arm, args.seed, True) or saved["source_hashes"] != code_hashes() or saved["bundle_signature"] != info["bundle_signature"]:
                raise ValueError("Existing smoke provenance mismatch")
            if saved["status"] == "TRAINED":
                print("SMOKE_ALREADY_VERIFIED", args.arm)
                return
            smoke_resume = True
        train(run_path(args.arm, args.seed, smoke), make_config(args.arm, args.seed, smoke), info, manifests,
              workers=args.workers, resume=args.command == "resume" or smoke_resume)
    elif args.command == "compare":
        compare(args.seed)
    elif args.command == "lock-holdout":
        lock_holdout(args.seed)
    else:
        evaluate_holdout(args.resume_evaluation)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
