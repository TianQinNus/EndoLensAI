"""Synthetic E10-B tests; no real fusion outcomes, training or model inference."""
import copy
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_e10b as e


class FusionTests(unittest.TestCase):
    def metric(self, dice=.8, mae=.02, low=2, near=1, zero=0):
        return dict(count=10, dice=dice, mae=mae, dice_lt05=low, dice_lt01=near, zero_dice=zero)

    def test_probability_formula_e0_direction_and_not_logit_average(self):
        b = np.array([[.9, .8]], np.float32); p0 = np.array([[.1, .2]], np.float32)
        p = e.fuse(b, p0, .25)
        np.testing.assert_array_equal(p, (.75 * b + .25 * p0).astype(np.float32))
        self.assertGreater(p[0, 0], .5)
        with self.assertRaises(ValueError): e.fuse(b, p0, .3)
        with self.assertRaises(ValueError): e.fuse(b.astype(np.float64), p0, .5)
        with self.assertRaises(ValueError): e.fuse(np.full_like(b, np.nan), p0, .5)
        with self.assertRaises(ValueError): e.fuse(b, p0[:, :1], .5)

    def test_dice_must_beat_both_not_just_weaker_single(self):
        refs = {e.B: self.metric(.80), e.E0: self.metric(.75)}
        gates, _ = e.performance_gates(self.metric(.78), refs)
        self.assertFalse(gates[e.B + "_dice"]); self.assertTrue(gates[e.E0 + "_dice"])
        gates, _ = e.performance_gates(self.metric(.803), refs)
        self.assertTrue(all(gates.values()))

    def test_mae_and_failure_counts_protected_against_both(self):
        refs = {e.B: self.metric(.80, .02, 2, 1, 0), e.E0: self.metric(.79, .03, 3, 2, 1)}
        for change, gate in ((dict(mae=.024), "_mae"), (dict(dice_lt05=3), "_low05"),
                             (dict(dice_lt01=2), "_near_zero"), (dict(zero_dice=1), "_zero")):
            candidate = self.metric(.82); candidate.update(change)
            gates, _ = e.performance_gates(candidate, refs)
            self.assertFalse(gates[e.B + gate])
        with self.assertRaises(ValueError): e.performance_gates(self.metric(), {e.B: self.metric()})

    def candidates(self):
        return [dict(alpha_e0=alpha, dice=dice, gates={"pass": passed})
                for alpha, dice, passed in zip(e.ALPHAS, (.82, .83, .84), (True, True, False))]

    def test_eligible_only_calibration_highest_and_tie_break(self):
        rows = self.candidates()
        self.assertEqual(e.choose_candidate(rows)["alpha_e0"], .5)
        rows[0]["dice"] = rows[1]["dice"]
        self.assertEqual(e.choose_candidate(rows)["alpha_e0"], .25)
        for row in rows: row["gates"]["pass"] = False
        self.assertIsNone(e.choose_candidate(rows))
        with self.assertRaises(ValueError): e.choose_candidate(rows[:-1])

    def records(self):
        return [dict(sample_id=str(i), source="FAR-POLYP-SEG", split="calibration" if i < 2 else "selection",
                     group_id="cal_patient" if i < 2 else "sel_patient", declared_group_id="Patient1" if i < 2 else "Patient2",
                     stem=str(i), height=2, width=2) for i in range(4)]

    def loader(self, record, index):
        gt = np.array([[1, 1], [0, 0]], bool)
        if record["split"] == "calibration":
            return np.array([[1, 0], [0, 0]], np.float32), np.array([[0, 1], [0, 0]], np.float32), gt
        return np.array([[1, 1], [0, 0]], np.float32), np.array([[1, 0], [0, 0]], np.float32), gt

    def test_calibration_never_decodes_selection_predictions_or_gt(self):
        records = self.records()
        def guarded(record, index):
            self.assertEqual(record["split"], "calibration")
            return self.loader(record, index)
        with patch.object(e.a, "read_csv", return_value=[dict(model="e0", sample_id=r["sample_id"]) for r in records]), \
             patch.object(e, "load_pair", side_effect=guarded):
            rows = e.compute_rows(records, "calibration", e.ALPHAS, quiet=True)
            self.assertEqual(len(rows), 10)
            self.assertEqual({r["split"] for r in rows}, {"calibration"})
        with self.assertRaises(ValueError): e.compute_rows(records, "calibration", e.ALPHAS, Path("unused"))

    def test_saved_membership_and_metric_tampering(self):
        records = self.records()
        with patch.object(e.a, "read_csv", return_value=[dict(model="e0", sample_id=r["sample_id"]) for r in records]), \
             patch.object(e, "load_pair", side_effect=self.loader):
            rows = e.compute_rows(records, "calibration", e.ALPHAS, quiet=True)
        e.compare_saved(rows, rows)
        invalid = copy.deepcopy(rows); invalid[0]["dice"] = .99
        with self.assertRaises(ValueError): e.compare_saved(invalid, rows)
        with self.assertRaises(ValueError): e.compare_saved(rows[:-1], rows)
        invalid = copy.deepcopy(rows); invalid[1] = invalid[0]
        with self.assertRaises(ValueError): e.compare_saved(invalid, rows)

    def test_full_calibration_lock_selection_failure_no_runner_up(self):
        with tempfile.TemporaryDirectory(dir=e.PACKAGE, prefix="_test_") as tmp:
            root = Path(tmp); plan_path = root / "plan.json"; output = root / "output"
            plan_path.write_text("synthetic-only", encoding="utf-8")
            records = self.records(); plan = dict(source_hashes={}, data_role="synthetic adaptive development")
            original_read = e.a.read_csv
            def read_csv(path):
                if Path(path) == e.a.CACHE / "per_image.csv":
                    return [dict(model="e0", sample_id=r["sample_id"]) for r in records]
                return original_read(path)
            with patch.object(e, "PROJECT", root), patch.object(e, "PLAN", plan_path), patch.object(e, "OUTPUT", output), \
                 patch.object(e, "check", return_value=(plan, records)), \
                 patch.object(e.a.e9.repeat, "serial_training_lock", return_value=nullcontext()), \
                 patch.object(e.a, "read_csv", side_effect=read_csv), patch.object(e, "load_pair", side_effect=self.loader):
                e.calibrate()
                lock = e.read_lock(); self.assertEqual(lock["selected_alpha_e0"], .5)
                before = e.file_hash(output / "locked_config.json")
                e.evaluate()
                decision = e.read_json(output / "selection/decision.json")
                self.assertEqual(decision["status"], "NO_PASS_RETAIN_B")
                self.assertFalse(decision["gates"][e.B + "_dice"])
                self.assertEqual(before, e.file_hash(output / "locked_config.json"))
                self.assertEqual({r["model"] for r in e.a.read_csv(output / "selection/per_image.csv")},
                                 {e.B, e.E0, e.candidate_name(.5)})
                e.verify()
                e.calibrate(); e.evaluate()  # integrity replay, no new evaluation
                self.assertEqual(before, e.file_hash(output / "locked_config.json"))
                (output / "selection/summary.csv").write_text("tampered", encoding="utf-8")
                with self.assertRaises(ValueError): e.verify_selection(records)

    def test_no_calibration_pass_blocks_selection_without_loading_it(self):
        lock = {"status": "NO_CALIBRATION_PASS_SELECTION_BLOCKED"}
        with patch.object(e, "check", return_value=({}, self.records())), patch.object(e, "verify_calibration"), \
             patch.object(e, "read_lock", return_value=lock), \
             patch.object(e.a.e9.repeat, "serial_training_lock", return_value=nullcontext()), \
             patch.object(e, "compute_rows", side_effect=AssertionError("Selection must stay blocked")):
            e.evaluate()

    def test_incomplete_selection_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=e.PACKAGE, prefix="_test_") as tmp:
            root = Path(tmp); (root / "selection").mkdir()
            e.write_json(root / "selection/run_info.json", {"status": "FAILED"})
            lock = dict(status="READY_FOR_SELECTION", selected_alpha_e0=.5)
            with patch.object(e, "OUTPUT", root), patch.object(e, "check", return_value=({}, self.records())), \
                 patch.object(e, "verify_calibration"), patch.object(e, "read_lock", return_value=lock), \
                 patch.object(e.a.e9.repeat, "serial_training_lock", return_value=nullcontext()), \
                 patch.object(e, "compute_rows", side_effect=AssertionError("No repeat")):
                with self.assertRaises(ValueError): e.evaluate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
