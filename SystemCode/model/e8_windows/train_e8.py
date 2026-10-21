"""Matched-exposure ResNet34-U-Net training; bounded training-only hard sampling."""
import gc
import math
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from common import (PROJECT, PACKAGE, CORE, ARMS, code_hashes, read_json, write_json,
                    file_hash, write_csv, validation_score, metric_summary)
from data import TrainingDataset, signature
from models import make_model
from engine import (require_cuda, set_seed, environment, atomic_save, evaluate_records,
                    lr_at_epoch, sync, utc)


def make_config(arm, seed=42, smoke=False):
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    return {"arm": arm, "seed": seed, "experiment": "pretrained", "phase": "development", "smoke": smoke,
            "epochs": 1 if smoke else 100, "schedule_horizon": 100, "warmup_epochs": 5,
            "samples_per_round": 8 if smoke else 1160, "input_size": 352,
            "batch_size": 2, "accumulation": 4, "lr": 1e-4, "weight_decay": 1e-4,
            "amp": True, "augmentation": "strong", "encoder_weights": "e2_weights/resnet34-b627a593.pth",
            "expanded": arm in ("B_expanded", "D_expanded_hard"), "hard": arm in ("C_hard", "D_expanded_hard"),
            "hard_warmup_rounds": 20, "hard_top_fraction": .25, "hard_multiplier": 2.0, "hard_ema_decay": .8,
            "sampling": "replacement; original mass 1.0 in A/C, 0.5 in B/D; remaining mass equal across new centres",
            "loss": "FP32 BCE + per-image soft Dice; 1+1; smooth=1; unchanged",
            "selection": "equal-weight old/C5 mean binary Dice; threshold 0.5; earliest tie",
            "budget": "100 rounds x 1160 sampled images; 145 planned updates/round; all arms matched"}


def per_image_loss(logits, target):
    logits, target = logits.float(), target.float()
    axes = (1, 2, 3)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(axes)
    p = logits.sigmoid()
    dice = 1 - (2 * (p * target).sum(axes) + 1) / (p.sum(axes) + target.sum(axes) + 1)
    return bce + dice, bce, dice


def sampling_probabilities(records, losses, config, epoch):
    strata = {}
    for i, r in enumerate(records):
        # No difficulty score, label, or prediction from validation/holdout enters this function.
        strata.setdefault(r["dataset"] if r["dataset"].startswith("PolypGen-") else "original", []).append(i)
    if "original" not in strata:
        raise ValueError("Missing original training data")
    probabilities = np.zeros(len(records), np.float64)
    new_strata = sorted(set(strata) - {"original"})
    if config["expanded"] and not new_strata:
        raise ValueError("Expanded arm lacks new training data")
    boosted = 0
    for name, indices in strata.items():
        mass = (.5 if name == "original" else .5 / len(new_strata)) if config["expanded"] else 1.0
        weights = np.ones(len(indices), np.float64)
        known = [j for j, i in enumerate(indices) if np.isfinite(losses[i])]
        if config["hard"] and epoch > config["hard_warmup_rounds"] and len(known) >= max(2, math.ceil(len(indices) / 2)):
            # Top quartile among observed training losses. Unobserved images stay
            # at normal weight, and all ordinary images retain nonzero probability.
            ranked = sorted(known, key=lambda j: (-losses[indices[j]], records[indices[j]]["sample_id"]))
            top = ranked[:max(1, math.ceil(config["hard_top_fraction"] * len(known)))]
            weights[top] = config["hard_multiplier"]
            boosted += len(top)
        probabilities[indices] = mass * weights / weights.sum()
    if not np.isclose(probabilities.sum(), 1) or np.any(probabilities <= 0):
        raise ValueError("Invalid sampling probabilities")
    return probabilities, boosted


class DrawDataset(Dataset):
    def __init__(self, project, records, indices, config, epoch):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.inner = TrainingDataset(project, [records[i] for i in self.indices],
                                     config["input_size"], config["seed"], epoch, config["augmentation"])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, y = self.inner[index]
        return x, y, int(self.indices[index])


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(v) for v in value)
    return value


def train(output, config, bundle_info, manifests, workers=2, resume=False, _test_device=None, _test_interrupt_epoch=None):
    device = _test_device if _test_device is not None else require_cuda()
    if device.type != "cuda" and _test_device is None:
        raise RuntimeError("CUDA required; no CPU fallback")
    output = Path(output)
    records = manifests["original_train"] + (manifests["new_train"] if config["expanded"] else [])
    validation = manifests["validation"]
    if config["smoke"]:
        # Include every required validation stratum in smoke selection.
        validation = [r for source in ("original_validation", "PolypGen-C5")
                      for r in [x for x in validation if x["dataset"] == source][:2]]
    hashes, env = code_hashes(), environment(device)
    previous = None
    if resume:
        info = read_json(output / "run_info.json")
        if info["status"] == "TRAINED":
            raise ValueError("Run is already complete")
        previous = torch.load(output / "checkpoints/last.pth", map_location="cpu", weights_only=True)
        if previous["config"] != config or previous["source_hashes"] != hashes:
            raise ValueError("Resume code/config mismatch")
        if previous["bundle_signature"] != bundle_info["bundle_signature"]:
            raise ValueError("Resume bundle mismatch")
        if previous["environment"]["versions"] != env["versions"] or previous["environment"].get("gpu") != env.get("gpu"):
            raise ValueError("Resume environment differs")
        history = previous["history"]
        losses = previous["loss_ema"].numpy().copy()
        best, best_score = previous["best"], previous["best_score"]
        start = previous["epoch"] + 1
        # Recover best sidecar from the committed last checkpoint, even if an
        # interruption occurred between last and best sidecar writes.
        atomic_save(best["checkpoint"], output / "checkpoints/best.pth")
        write_csv(output / "validation_best_per_image.csv", best["rows"])
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "checkpoints").mkdir()
        (output / "code_snapshot").mkdir()
        for name in hashes:
            destination = output / "code_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, destination)
        write_json(output / "bundle_snapshot.json", bundle_info)
        write_json(output / "train_manifest.json", records)
        write_json(output / "validation_manifest.json", validation)
        info = {"status": "INITIALIZING", "run_kind": "E8_GPU_SMOKE" if config["smoke"] else "E8_DEVELOPMENT",
                "config": config, "environment": env, "source_hashes": hashes, "bundle_signature": bundle_info["bundle_signature"],
                "started_at": utc(), "completed_epochs": 0, "formal_holdout_evaluations": 0,
                "note": "No E0 training; no historical test or FAR images used for training or selection."}
        history, losses, best, best_score, start = [], np.full(len(records), np.nan), None, -1.0, 1
    write_json(output / "run_info.json", info)

    def say(message):
        print(message, flush=True)
        with (output / "console.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    model = optimizer = scaler = None
    try:
        set_seed(config["seed"], device)
        model = make_model(config, PROJECT, initialization=not resume).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        use_amp = config["amp"] and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        if previous is not None:
            model.load_state_dict(previous["model_state"], strict=True)
            optimizer.load_state_dict(previous["optimizer_state"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and key != "step":
                        state[key] = value.to(device)
            scaler.load_state_dict(previous["scaler_state"])
            torch.set_rng_state(previous["cpu_rng_state"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(previous["cuda_rng_states"])
        elif hasattr(model, "initial_weight_sha256"):
            info["initial_encoder_sha256"] = model.initial_weight_sha256
        del previous
        info.update(status="TRAINING", training_unique_images=len(records), validation_images=len(validation),
                    model_device=str(next(model.parameters()).device), amp_enabled=use_amp,
                    parameter_count=sum(p.numel() for p in model.parameters()))
        write_json(output / "run_info.json", info)
        write_csv(output / "train_history.csv", history)
        say(f"E8 {config['arm']} seed={config['seed']} | DEVICE={device} GPU={env.get('gpu','unit-test only')} | "
            f"unique train={len(records)} validation={len(validation)} | {config['epochs']} rounds x {config['samples_per_round']} draws")
        for epoch in range(start, config["epochs"] + 1):
            if code_hashes() != hashes:
                raise ValueError("Training code changed mid-run")
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            lr = lr_at_epoch(config, epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            probabilities, boosted = sampling_probabilities(records, losses, config, epoch)
            rng = np.random.default_rng(np.random.SeedSequence([config["seed"], epoch, 808]))
            indices = rng.choice(len(records), config["samples_per_round"], replace=True, p=probabilities)
            draw_hash = hashlib_sha256_indices(indices)
            dataset = DrawDataset(PROJECT, records, indices, config, epoch)
            loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                                generator=torch.Generator().manual_seed(config["seed"] + epoch),
                                num_workers=workers, pin_memory=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            totals = np.zeros(3)
            seen, updates, skipped_updates = 0, 0, 0
            epoch_losses = [[] for _ in records]
            sync(device)
            started = time.perf_counter()
            proof = None
            for batch, (x, y, record_indices) in enumerate(loader):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                vector, bce, dice = per_image_loss(logits, y)
                loss = vector.mean()
                if not bool(torch.isfinite(vector).all()):
                    raise RuntimeError("Nonfinite training loss")
                group_start = (batch // config["accumulation"]) * config["accumulation"] * config["batch_size"]
                group_samples = min(config["batch_size"] * config["accumulation"], len(dataset) - group_start)
                scaler.scale(loss * len(x) / group_samples).backward()
                totals += np.array([loss.item(), bce.mean().item(), dice.mean().item()]) * len(x)
                seen += len(x)
                for index, value in zip(record_indices.tolist(), vector.detach().cpu().tolist()):
                    epoch_losses[index].append(value)
                if proof is None:
                    proof = {"model": str(next(model.parameters()).device), "images": str(x.device),
                             "masks": str(y.device), "logits": str(logits.device), "forward_dtype": str(logits.dtype),
                             "loss_dtype": str(loss.dtype)}
                    if not (next(model.parameters()).device == x.device == y.device == logits.device == device):
                        raise RuntimeError("GPU device mismatch")
                if (batch + 1) % config["accumulation"] == 0 or batch + 1 == len(loader):
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped = use_amp and scaler.get_scale() < scale_before
                    skipped_updates += int(skipped)
                    updates += int(not skipped)
                    optimizer.zero_grad(set_to_none=True)
                if (batch + 1) % 40 == 0 or batch + 1 == len(loader):
                    say(f"[round {epoch}/{config['epochs']}] batch {batch+1}/{len(loader)} loss={loss.item():.5f}")
            if seen != config["samples_per_round"]:
                raise RuntimeError("Matched sample budget mismatch")
            for index, values in enumerate(epoch_losses):
                if values:
                    current = float(np.mean(values))
                    losses[index] = current if not np.isfinite(losses[index]) else (
                        config["hard_ema_decay"] * losses[index] + (1-config["hard_ema_decay"]) * current)
            sync(device)
            train_seconds = time.perf_counter() - started
            val_started = time.perf_counter()
            _, rows = evaluate_records(model, PROJECT, validation, device, config["input_size"])
            val_seconds = time.perf_counter() - val_started
            score = validation_score(rows)
            metrics = metric_summary(rows)
            row = {"round": epoch, "train_loss": float(totals[0]/seen), "bce_loss": float(totals[1]/seen), "soft_dice_loss": float(totals[2]/seen),
                   "lr": lr, "train_draws": seen, "unique_drawn": len(set(indices.tolist())), "optimizer_updates": updates,
                   "amp_skipped_updates": skipped_updates, "hard_boosted_images": boosted,
                   "sample_indices_sha256": draw_hash, "validation_macro_dice": score,
                   "validation_image_mean_dice": metrics["dice"], "validation_mae": metrics["mae"],
                   "validation_dice_lt05": metrics["dice_lt05"], "train_seconds": train_seconds,
                   "validation_seconds": val_seconds,
                   "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0}
            history.append(row)
            weights = {"format_version": 1, "epoch": epoch, "config": config, "model_state": snapshot(model.state_dict()),
                       "source_hashes": hashes, "bundle_signature": bundle_info["bundle_signature"]}
            if score > best_score:
                best_score = score
                best = {"checkpoint": weights, "rows": rows}
            state = {**weights, "environment": env, "optimizer_state": snapshot(optimizer.state_dict()),
                     "scaler_state": scaler.state_dict(), "history": history, "loss_ema": torch.from_numpy(losses.copy()),
                     "best": best, "best_score": best_score, "cpu_rng_state": torch.get_rng_state(),
                     "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
            atomic_save(state, output / "checkpoints/last.pth")
            # last.pth commits first and contains a recoverable best checkpoint.
            if best["checkpoint"]["epoch"] == epoch:
                atomic_save(best["checkpoint"], output / "checkpoints/best.pth")
                write_csv(output / "validation_best_per_image.csv", best["rows"])
            write_csv(output / "validation_last_per_image.csv", rows)
            write_csv(output / "train_history.csv", history)
            info.update(completed_epochs=epoch, best_epoch=best["checkpoint"]["epoch"], best_validation_macro_dice=best_score,
                        best_checkpoint_sha256=file_hash(output / "checkpoints/best.pth"), last_device_proof=proof,
                        best_validation_csv_sha256=file_hash(output / "validation_best_per_image.csv"),
                        optimizer_updates_total=sum(r["optimizer_updates"] for r in history),
                        amp_skipped_updates_total=sum(r["amp_skipped_updates"] for r in history), updated_at=utc())
            write_json(output / "run_info.json", info)
            remaining = (train_seconds + val_seconds) * (config["epochs"]-epoch) / 3600
            say(f"Round {epoch}: macro Dice={score:.4f}; low Dice={metrics['dice_lt05']}; MAE={metrics['mae']:.5f}; "
                f"best={best_score:.4f}@{best['checkpoint']['epoch']}; train={train_seconds:.1f}s val={val_seconds:.1f}s; "
                f"rough remaining={remaining:.2f}h; hard boosted={boosted}")
            del state, weights
            if _test_device is not None and _test_interrupt_epoch == epoch:
                raise KeyboardInterrupt("unit-test interruption")
        info.update(status="TRAINED", finished_at=utc(), formal_holdout_evaluations=0)
        write_json(output / "run_info.json", info)
        say("TRAINING COMPLETE. C6 holdout, historical 798 and FAR metrics were NOT evaluated.")
    except BaseException as error:
        info.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                    error=f"{type(error).__name__}: {error}", updated_at=utc())
        write_json(output / "run_info.json", info)
        say("Run incomplete; resume the last committed round with unchanged code/data. No CPU fallback.")
        raise
    finally:
        del model, optimizer, scaler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def hashlib_sha256_indices(indices):
    import hashlib
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
