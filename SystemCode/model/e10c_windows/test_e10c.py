"""Synthetic temperature/CV workflow tests; no real FAR outcome calculations."""
import copy
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_e10c as e
from temperature import logits, apply_temperature, assign_folds, sample_pair, fit_temperature


class TemperatureTests(unittest.TestCase):
    def test_identity_endpoints_and_positive_temperature_preserve_single_masks(self):
        p = np.array([0, 1e-8, np.nextafter(np.float32(.5), np.float32(0)), .5, .9, 1], np.float32)
        np.testing.assert_array_equal(apply_temperature(p, 1), p)
        for t in (.25, .5, 2, 4):
            np.testing.assert_array_equal(apply_temperature(p, t) >= .5, p >= .5)
        self.assertEqual(logits(np.array([0, 1], np.float32))[0], -logits(np.array([0, 1], np.float32))[1])
        with self.assertRaises(ValueError): apply_temperature(p, 0)
        with self.assertRaises(ValueError): apply_temperature(p, 5)

    def test_identity_equal_fusion_exactly_replays_raw(self):
        rng = np.random.default_rng(42)
        pb, p0 = rng.random((16, 16), dtype=np.float32), rng.random((16, 16), dtype=np.float32)
        np.testing.assert_array_equal(e.b.fuse(apply_temperature(pb, 1), apply_temperature(p0, 1), .5), e.b.fuse(pb, p0, .5))

    def test_shared_temperature_does_not_create_new_segmentation_information(self):
        # Non-boundary values: common positive T leaves sign(logitB+logitE0)
        # unchanged. Independent temperatures alter relative confidence only.
        pb = np.array([.1, .9, .4, .8], np.float32)
        p0 = np.array([.2, .2, .4, .7], np.float32)
        for t in (.25, 4):
            np.testing.assert_array_equal(e.b.fuse(apply_temperature(pb, t), apply_temperature(p0, t), .5) >= .5,
                                          e.b.fuse(pb, p0, .5) >= .5)

    def test_bounded_fit_softens_errors_and_sharpens_correct_predictions(self):
        errors = dict(b_logits=np.array([-4, -4, 4, 4.]), e0_logits=np.array([-4, -4, 4, 4.]), targets=np.array([0, 1, 0, 1.]))
        fit = fit_temperature([errors], "b")
        self.assertEqual(fit["temperature"], 4)
        self.assertLess(fit["sampled_image_equal_nll_after"], fit["sampled_image_equal_nll_before"])
        correct = dict(errors, targets=np.array([0, 0, 1, 1.]))
        self.assertEqual(fit_temperature([correct], "b")["temperature"], .25)
        with self.assertRaises(ValueError): fit_temperature([], "b")

    def test_image_equal_weights_not_pixel_count_weights(self):
        first = dict(b_logits=np.array([-2., 2.]), e0_logits=np.array([-2., 2.]), targets=np.array([0., 1.]))
        second = dict(b_logits=np.array([-2., 2.]), e0_logits=np.array([-2., 2.]), targets=np.array([1., 0.]))
        repeated = {k: np.tile(v, 100) for k, v in second.items()}
        original, duplicate = fit_temperature([first, second], "b"), fit_temperature([first, repeated], "b")
        self.assertAlmostEqual(original["temperature"], duplicate["temperature"])
        self.assertAlmostEqual(original["sampled_image_equal_nll_before"], duplicate["sampled_image_equal_nll_before"])

    def records(self):
        return [dict(sample_id=str(i), source="FAR-POLYP-SEG", split="calibration", group_id=f"Patient{i//2}",
                     declared_group_id=f"Patient{i//2}", stem=str(i), height=2, width=2) for i in range(10)]

    def test_patient_folds_deterministic_and_group_exclusive(self):
        rows = self.records(); folded = assign_folds(rows, 5, 42)
        self.assertEqual(folded, assign_folds(rows, 5, 42))
        for group in {r["group_id"] for r in rows}:
            self.assertEqual(len({r["cv_fold"] for r in folded if r["group_id"] == group}), 1)
        bad = copy.deepcopy(rows); bad[0]["split"] = "selection"
        with self.assertRaises(ValueError): assign_folds(bad)

    def test_uniform_paired_sampling_deterministic_without_class_balancing(self):
        pb = np.full((8, 8), .8, np.float32); p0 = np.full_like(pb, .2); gt = np.zeros((8, 8), bool); gt[:2] = True
        first = sample_pair(pb, p0, gt, "sample", maximum=20)
        second = sample_pair(pb, p0, gt, "sample", maximum=20)
        for k in first:
            if isinstance(first[k], np.ndarray): np.testing.assert_array_equal(first[k], second[k])
            else: self.assertEqual(first[k], second[k])
        self.assertEqual(len(first["targets"]), 20)
        self.assertNotEqual(first["sample_indices_sha256"], sample_pair(pb, p0, gt, "other", maximum=20)["sample_indices_sha256"])

    def test_fit_pair_cannot_receive_selection_and_only_uses_training_ids(self):
        records = self.records()
        samples = {r["sample_id"]: {"b_logits": np.array([-2., 2.]), "e0_logits": np.array([-1., 1.]),
                                    "targets": np.array([0., 1.]), "sample_indices_sha256": r["sample_id"]} for r in records}
        fit = e.fit_pair(records[:4], samples)
        self.assertEqual(fit["training_ids"], ["0", "1", "2", "3"])
        self.assertEqual(fit["B"]["image_count"], 4)
        invalid = [dict(records[0], split="selection")]
        with self.assertRaises(ValueError): e.fit_pair(invalid, samples)

    def test_oof_block_stops_full_refit_and_selection(self):
        prior = {"status": "NO_OOF_PASS_RETAIN_B", "full_calibration_fit_allowed": False}
        with patch.object(e, "check", return_value=({}, [], [])), patch.object(e, "verify_cv"), \
             patch.object(e, "read_json", return_value=prior), \
             patch.object(e.b.a.e9.repeat, "serial_training_lock", return_value=nullcontext()), \
             patch.object(e, "collect_samples", side_effect=AssertionError("No refit")), \
             patch.object(e, "evaluate_records", side_effect=AssertionError("No selection")):
            e.refit(); e.evaluate()

    def test_full_synthetic_oof_failure_preserves_b_and_replays(self):
        with tempfile.TemporaryDirectory(dir=e.PACKAGE, prefix="_test_") as tmp:
            root = Path(tmp); output = root / "output"; plan = root / "plan.json"
            plan.write_text("synthetic-only", encoding="utf-8")
            folded = assign_folds(self.records(), 5, 42)
            def loader(record, index):
                # Identical errors cannot beat either single by0.003.
                return np.array([[.9, .1], [.9, .1]], np.float32), np.array([[.9, .1], [.9, .1]], np.float32), np.array([[1, 1], [0, 0]], bool)
            def fit(training, samples):
                return {"B": {"temperature": 1.}, "E0": {"temperature": 1.},
                        "training_ids": [r["sample_id"] for r in training],
                        "training_groups": sorted({r["group_id"] for r in training})}
            with patch.object(e, "PROJECT", root), patch.object(e, "PLAN", plan), patch.object(e, "OUTPUT", output), \
                 patch.object(e, "check", return_value=({"source_hashes": {}}, folded, folded)), \
                 patch.object(e, "index_e0", return_value={}), patch.object(e.b, "load_pair", side_effect=loader), \
                 patch.object(e, "fit_pair", side_effect=fit), \
                 patch.object(e.b.a.e9.repeat, "serial_training_lock", return_value=nullcontext()):
                e.cv()
                self.assertEqual(e.read_json(output / "cross_validation/decision.json")["status"], "NO_OOF_PASS_RETAIN_B")
                e.verify()
                e.refit(); e.evaluate()
                self.assertFalse((output / "full_calibration").exists())
                self.assertFalse((output / "selection").exists())
                self.assertFalse((output / "locked_config.json").exists())
                e.cv()  # replay only
                (output / "cross_validation/summary.csv").write_text("tampered", encoding="utf-8")
                with self.assertRaises(ValueError): e.verify_cv(folded)

    def test_fold_guard_checks_failures_even_if_other_folds_improve(self):
        rows = []
        for fold in range(5):
            for label in e.LABELS:
                dice = .9 if label in (e.RAW, e.CAL) else .7
                if fold == 0 and label == e.RAW: dice = .1
                if fold == 0 and label == e.CAL: dice = 0
                rows.append(dict(model=label, sample_id=str(fold), cv_fold=fold, dice=dice, iou=.6, mae=.01,
                                 full_image_nll=.1, full_image_brier=.02, gt_area_fraction=.02, dataset="FAR"))
        gates, _, _ = e.assess_oof(rows)
        self.assertTrue(gates["at_least_four_folds_retain_raw_dice"])
        self.assertFalse(gates["fold0_zero_dice_vs_raw"])

    def test_full_synthetic_pass_refit_lock_selection_and_verify(self):
        with tempfile.TemporaryDirectory(dir=e.PACKAGE, prefix="_test_") as tmp:
            root = Path(tmp); output = root / "output"; plan = root / "plan.json"
            plan.write_text("synthetic-only", encoding="utf-8")
            folded = assign_folds(self.records(), 5, 42)
            selection = [dict(r, sample_id="s" + r["sample_id"], stem="s" + r["stem"], split="selection", group_id="new_patient")
                         for r in self.records()[:2]]
            def loader(record, index):
                return np.array([[.9, .1], [.9, .1]], np.float32), np.array([[.1, .9], [.1, .9]], np.float32), np.array([[1, 1], [0, 0]], bool)
            def fit(training, samples):
                return {"B": {"temperature": 1.}, "E0": {"temperature": 1.},
                        "training_ids": [r["sample_id"] for r in training],
                        "training_groups": sorted({r["group_id"] for r in training})}
            with patch.object(e, "PROJECT", root), patch.object(e, "PLAN", plan), patch.object(e, "OUTPUT", output), \
                 patch.object(e, "check", return_value=({"source_hashes": {}, "data_role": "synthetic"}, folded + selection, folded)), \
                 patch.object(e, "index_e0", return_value={}), patch.object(e.b, "load_pair", side_effect=loader), \
                 patch.object(e, "fit_pair", side_effect=fit), \
                 patch.object(e.b.a.e9.repeat, "serial_training_lock", return_value=nullcontext()):
                e.cv(); e.refit()
                self.assertEqual(e.read_lock()["status"], "READY_FOR_SELECTION")
                before = e.file_hash(output / "locked_config.json")
                e.evaluate(); e.verify()
                self.assertEqual(e.read_json(output / "selection/decision.json")["status"], "DEVELOPMENT_PASS")
                self.assertEqual(before, e.file_hash(output / "locked_config.json"))
                e.evaluate()  # preserves completed selection; no new computation
                self.assertEqual(before, e.file_hash(output / "locked_config.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
