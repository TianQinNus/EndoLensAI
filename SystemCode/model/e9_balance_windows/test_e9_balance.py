"""Synthetic tests; no real training, holdout metrics or E0 inference."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import run_e9_balance as e9
from diagnose_e8 import select_cases, error_type


class BalanceTests(unittest.TestCase):
    def records(self):
        return [{"dataset": source, "sample_id": f"{source}/{i}"}
                for source, n in [("original_training", 7)] + [(f"PolypGen-C{j}", j + 1) for j in range(1, 5)]
                for i in range(n)]

    def rows(self, old=.91, c5=.69):
        return [{"sample_id": str(i), "dataset": source, "dice": value, "iou": .6,
                 "mae": .02, "gt_area_fraction": .05}
                for i, (source, value) in enumerate([("original_validation", old), ("PolypGen-C5", c5)])]

    def test_only_training_change_and_all_seeds(self):
        for seed in e9.SEEDS:
            candidate = e9.make_config(seed)
            base = e9.trainer.make_config("B_expanded", seed)
            candidate.pop("stage"); candidate.pop("original_sampling_mass"); candidate["sampling"] = base["sampling"]
            self.assertEqual(candidate, base)
        with self.assertRaises(ValueError): e9.make_config(99)

    def test_source_mass_and_no_difficulty_effect(self):
        records = self.records()
        config = e9.make_config(42)
        probabilities, boosted = e9.sampling_probabilities(records, np.full(len(records), np.nan), config, 1)
        again, _ = e9.sampling_probabilities(records, np.arange(len(records)), config, 99)
        np.testing.assert_array_equal(probabilities, again)
        self.assertEqual(boosted, 0)
        for source in {r["dataset"] for r in records}:
            indices = [i for i, row in enumerate(records) if row["dataset"] == source]
            self.assertAlmostEqual(probabilities[indices].sum(), .7 if source == "original_training" else .075)
            self.assertEqual(len(set(probabilities[indices])), 1)

    def test_50_50_sampler_exact_parent_parity(self):
        records = self.records(); losses = np.full(len(records), np.nan)
        config = e9.make_config(42); config["original_sampling_mass"] = .5
        new, _ = e9.sampling_probabilities(records, losses, config, 40)
        old, _ = e9.trainer.sampling_probabilities(records, losses, config, 40)
        np.testing.assert_array_equal(new, old)
        for seed in e9.SEEDS:
            a = np.random.default_rng(np.random.SeedSequence([seed, 40, 808]))
            b = np.random.default_rng(np.random.SeedSequence([seed, 40, 808]))
            np.testing.assert_array_equal(a.choice(len(records), 1160, p=new), b.choice(len(records), 1160, p=old))

    def test_holdout_validation_missing_centres_and_hard_forbidden(self):
        for source in ("PolypGen-C5", "PolypGen-C6"):
            records = self.records() + [{"dataset": source, "sample_id": "bad"}]
            with self.assertRaises(ValueError): e9.sampling_probabilities(records, [], e9.make_config(42), 1)
        config = e9.make_config(42); config["hard"] = True
        with self.assertRaises(ValueError): e9.sampling_probabilities(self.records(), [], config, 21)
        config = e9.make_config(42); config["original_sampling_mass"] = 1
        with self.assertRaises(ValueError): e9.sampling_probabilities(self.records(), [], config, 1)

    def test_adapter_restores_parent_even_on_failure(self):
        original = (e9.trainer.sampling_probabilities, e9.trainer.code_hashes, e9.trainer.write_json)
        with self.assertRaises(RuntimeError):
            with e9.training_adapter():
                self.assertIs(e9.trainer.sampling_probabilities, e9.sampling_probabilities)
                raise RuntimeError("synthetic")
        self.assertEqual(original, (e9.trainer.sampling_probabilities, e9.trainer.code_hashes, e9.trainer.write_json))

    def test_gates_retain_expansion_and_protect_old_source(self):
        a = self.rows(.90, .59); b = self.rows(.92, .69); candidate = self.rows(.92, .69)
        gates, details = e9.development_gates(candidate, a, b)
        self.assertTrue(all(gates.values())); self.assertEqual(len(details), 2)
        candidate = self.rows(.88, .74)
        gates, _ = e9.development_gates(candidate, a, b)
        self.assertFalse(gates["original_dice_protected_vs_A"])
        candidate = self.rows(.94, .675)
        gates, _ = e9.development_gates(candidate, a, b)
        self.assertFalse(gates["C5_dice_retained_vs_B"])

    def test_zero_failure_gate_not_hidden_by_mean(self):
        a = self.rows(.90, .59); b = self.rows(.92, .69)
        candidate = self.rows(.93, .70)
        # Same validation case count: add one previously near-zero case to all arms.
        for rows, value in ((a, .09), (b, .02), (candidate, 0)):
            row = dict(rows[1], sample_id="tail", dice=value); rows.append(row)
        gates, _ = e9.development_gates(candidate, a, b)
        self.assertFalse(gates["PolypGen-C5_zero_vs_B"])

    def test_validation_membership_and_metrics(self):
        rows = self.rows()
        e9.validate_rows(rows, rows)
        for invalid in (rows[:1], rows + [rows[0]]):
            with self.assertRaises(ValueError): e9.validate_rows(invalid, rows)
        bad = copy.deepcopy(rows); bad[0]["dice"] = float("nan")
        with self.assertRaises(ValueError): e9.validate_rows(bad, rows)

    def test_continuation_no_overwrite_or_active_duplicate(self):
        with tempfile.TemporaryDirectory(dir=e9.PACKAGE, prefix="_test_") as tmp:
            root = Path(tmp) / "run"
            self.assertEqual(e9.continuation_action(root, {}, {}, "data"), "train")
            root.mkdir()
            info = {"config": {}, "source_hashes": {}, "bundle_signature": "data", "status": "TRAINED"}
            (root / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
            self.assertEqual(e9.continuation_action(root, {}, {}, "data"), "skip")
            with self.assertRaises(ValueError): e9.continuation_action(root, {"seed": 17}, {}, "data")
            info["status"] = "TRAINING"
            (root / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaises(ValueError): e9.continuation_action(root, {}, {}, "data")
            info["status"] = "INTERRUPTED"
            (root / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaises(ValueError): e9.continuation_action(root, {}, {}, "data")
            (root / "checkpoints").mkdir(); (root / "checkpoints/last.pth").touch()
            self.assertEqual(e9.continuation_action(root, {}, {}, "data"), "resume")

    def test_diagnosis_selection_and_error_types(self):
        indexed = {}
        for seed in e9.SEEDS:
            indexed["B_expanded", seed] = {"new": {"dataset": "original_validation", "dice": .2},
                                            "persistent": {"dataset": "PolypGen-C5", "dice": 0},
                                            "rescue": {"dataset": "original_validation", "dice": .7}}
        indexed["A_baseline", 2024] = {"new": {"dice": .8}, "persistent": {"dice": .1}, "rescue": {"dice": .2}}
        new, rescued, common, zeros = select_cases(indexed)
        self.assertEqual(new, {"new"}); self.assertEqual(rescued, {"rescue"})
        self.assertEqual(common, {"new", "persistent"}); self.assertEqual(zeros, {"persistent"})
        self.assertEqual(error_type(dict(tp=0, fp=0, fn=10)), "empty_prediction")
        self.assertEqual(error_type(dict(tp=0, fp=10, fn=10)), "disjoint_prediction")
        self.assertEqual(error_type(dict(tp=2, fp=1, fn=8)), "mainly_undersegmentation")
        self.assertEqual(error_type(dict(tp=8, fp=12, fn=2)), "mainly_oversegmentation")

    def test_synthetic_adapter_training_resume_exact(self):
        with tempfile.TemporaryDirectory(dir=e9.PACKAGE, prefix="_test_") as tmp:
            root = Path(tmp)
            mask = np.zeros((32, 32), np.uint8); mask[6:16, 8:20] = 255
            Image.fromarray(np.stack([mask] * 3, axis=-1)).save(root / "x.png")
            Image.fromarray(mask).save(root / "y.png")
            def row(i, source):
                return dict(sample_id=str(i), dataset=source, image="x.png", mask="y.png")
            manifests = {"original_train": [row(i, "original_training") for i in range(4)],
                         "new_train": [row(10 + i, f"PolypGen-C{i + 1}") for i in range(4)],
                         "validation": [row(20, "original_validation"), row(21, "PolypGen-C5")]}
            config = dict(e9.make_config(42, smoke=True), input_size=32, epochs=2, samples_per_round=16)
            with patch.object(e9.trainer, "PROJECT", root), patch.object(e9, "stage_hashes", return_value={}), \
                 patch.object(e9.trainer, "make_model", side_effect=lambda *a, **k: torch.nn.Conv2d(3, 1, 1)), e9.training_adapter():
                with self.assertRaises(KeyboardInterrupt):
                    e9.trainer.train(root / "run", config, {"bundle_signature": "test"}, manifests,
                                     workers=0, _test_device=torch.device("cpu"), _test_interrupt_epoch=1)
                e9.trainer.train(root / "run", config, {"bundle_signature": "test"}, manifests,
                                 workers=0, resume=True, _test_device=torch.device("cpu"))
                e9.trainer.train(root / "reference", config, {"bundle_signature": "test"}, manifests,
                                 workers=0, _test_device=torch.device("cpu"))
                cp = torch.load(root / "run/checkpoints/last.pth", weights_only=True)
                ref = torch.load(root / "reference/checkpoints/last.pth", weights_only=True)
                for name in cp["model_state"]:
                    torch.testing.assert_close(cp["model_state"][name], ref["model_state"][name], rtol=0, atol=0)
                self.assertEqual([r["sample_indices_sha256"] for r in cp["history"]],
                                 [r["sample_indices_sha256"] for r in ref["history"]])
                info = e9.read_json(root / "run/run_info.json")
                self.assertEqual(info["run_kind"], "E9_GPU_SMOKE")
                self.assertEqual(info["status"], "TRAINED")
                self.assertTrue(all(r["hard_boosted_images"] == 0 for r in cp["history"]))

    def test_summary_single_seed_failure_cannot_hide_in_mean(self):
        with tempfile.TemporaryDirectory(dir=e9.PACKAGE, prefix="_test_") as tmp:
            root = Path(tmp); plan = root / "plan.json"; plan.write_text("synthetic", encoding="utf-8")
            output = root / "summary"
            fixtures = {}
            for seed in e9.SEEDS:
                for arm in ("A_baseline", "B_expanded", "B_balance70"):
                    folder = root / f"{arm}_{seed}"; (folder / "checkpoints").mkdir(parents=True)
                    for name in ("run_info.json", "checkpoints/best.pth", "validation_best_per_image.csv", "train_history.csv"):
                        (folder / name).write_text("synthetic", encoding="utf-8")
                    rows = self.rows(.90, .59) if arm == "A_baseline" else self.rows(.92, .69)
                    if seed == 2024 and arm == "B_balance70": rows = self.rows(.88, .74)
                    result = {"arm": arm, "seed": seed, "best_round": 1, "macro_dice": e9.validation_score(rows),
                              **e9.metric_summary(rows)}
                    fixtures[arm, seed] = (folder, {}, rows, result)
            with patch.object(e9, "PROJECT", root), patch.object(e9, "PLAN", plan), patch.object(e9, "OUTPUT", output), \
                 patch.object(e9, "check", return_value=({}, {}, {}, {})), \
                 patch.object(e9, "verified_run", side_effect=lambda s, *a: fixtures["B_balance70", s]), \
                 patch.object(e9.repeat, "verified_run", side_effect=lambda a, s, *args: fixtures[a, s]):
                e9.summarize()
                decision = e9.read_json(output / "decision.json")
                self.assertEqual(decision["status"], "NO_PASS_RETAIN_B")
                self.assertEqual([r["passed"] for r in decision["per_seed"]], [True, True, False])
                self.assertEqual(len(e9.read_csv(output / "per_seed.csv")), 9)
                self.assertEqual(len(e9.read_csv(output / "paired_contrast_mean_sd.csv")), 2)
                e9.verify_summary()
                e9.summarize()  # verified existing output is preserved
                (output / "mean_sd.csv").write_text("tampered", encoding="utf-8")
                with self.assertRaises(ValueError): e9.verify_summary()

    def test_summary_refuses_partial_runs_before_creating_output(self):
        with tempfile.TemporaryDirectory(dir=e9.PACKAGE, prefix="_test_") as tmp:
            output = Path(tmp) / "summary"
            with patch.object(e9, "OUTPUT", output), patch.object(e9, "check", return_value=({}, {}, {}, {})), \
                 patch.object(e9, "verified_run", side_effect=FileNotFoundError("Incomplete training")):
                with self.assertRaises(FileNotFoundError): e9.summarize()
                self.assertFalse(output.exists())


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
