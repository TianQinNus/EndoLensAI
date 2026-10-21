"""Synthetic E10-A tests. Never trains or reads real FAR outcomes."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
import run_e10a as e10


class E10Tests(unittest.TestCase):
    def test_legacy_manifest_signature_convention_is_preserved(self):
        import hashlib
        value = {"x": 1, "y": 2}
        expected = hashlib.sha256(b'{"x":1,"y":2}').hexdigest()
        self.assertEqual(e10.legacy_e6_signature(value), expected)
        self.assertNotEqual(e10.signature(value), expected)

    def membership(self):
        records = []
        for group in range(187):
            split = "calibration" if group < 75 else "selection"
            number = (3 if group < 10 else 2) if split == "calibration" else (3 if group < 123 else 2)
            for i in range(number):
                stem = f"Patient{group}__frame{i}"
                records.append(dict(sample_id="FAR-POLYP-SEG/" + stem, stem=stem, source="FAR-POLYP-SEG",
                                    declared_group_id=f"Patient{group}", group_id=f"group{group}", split=split))
        return records

    def test_original_patient_counts_and_no_cross_split(self):
        records = self.membership()
        self.assertEqual(e10.audit_membership(records)["patient_groups"], 187)
        invalid = copy.deepcopy(records); invalid[-1]["group_id"] = invalid[0]["group_id"]
        with self.assertRaises(ValueError): e10.audit_membership(invalid)
        invalid = copy.deepcopy(records); invalid[0]["declared_group_id"] = "Patient9999"
        with self.assertRaises(ValueError): e10.audit_membership(invalid)
        with self.assertRaises(ValueError): e10.audit_membership(records[:-1])

    def test_path_traversal_and_absolute_paths_blocked(self):
        for value in ("../escape", "C:/escape", "masks\\escape", ""):
            with self.assertRaises(ValueError): e10.safe_path(e10.PACKAGE, value)
        self.assertTrue(e10.safe_path(e10.PACKAGE, "predictions/x.npz").is_relative_to(e10.PACKAGE))

    def test_cached_probability_validation(self):
        with tempfile.TemporaryDirectory(dir=e10.PACKAGE, prefix="_test_") as tmp:
            path = Path(tmp) / "p.npz"
            for array in (np.zeros((2, 2), np.float64), np.full((2, 2), np.nan, np.float32),
                          np.full((2, 2), 1.1, np.float32), np.zeros((2, 3), np.float32)):
                np.savez_compressed(path, probability=array)
                with self.assertRaises(ValueError): e10.load_probability(path, (2, 2))
            np.savez_compressed(path, probability=np.full((2, 2), .5, np.float32))
            self.assertEqual(e10.load_probability(path, (2, 2)).dtype, np.float32)
            np.savez_compressed(path, wrong=np.zeros((2, 2), np.float32))
            with self.assertRaises(ValueError): e10.load_probability(path, (2, 2))

    def test_metric_replay_threshold_and_tampering(self):
        with tempfile.TemporaryDirectory(dir=e10.PACKAGE, prefix="_test_") as tmp:
            root = Path(tmp)
            Image.fromarray(np.array([[255, 0], [0, 0]], np.uint8)).save(root / "gt.png")
            record = dict(sample_id="x", source="FAR-POLYP-SEG", split="calibration", group_id="g", declared_group_id="Patient1", mask="gt.png")
            p = np.array([[.5, .49], [0, 0]], np.float32)
            with patch.object(e10, "PROJECT", root):
                row = e10.replay_row(record, p)
                self.assertEqual(row["dice"], 1)
                e10.replay_row(record, p, row)
                bad = dict(row, dice=.9)
                with self.assertRaises(ValueError): e10.replay_row(record, p, bad)

    def test_normalization_matches_declared_protocol(self):
        im = Image.fromarray(np.full((8, 8, 3), 128, np.uint8))
        tensor = e10.normalize_rgb(im, 352)
        expected = (128 / 255 - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        np.testing.assert_allclose(tensor.numpy()[:, 0, 0], expected, atol=1e-6, rtol=0)
        self.assertEqual(tuple(tensor.shape), (3, 352, 352))

    def test_exact_training_overlap_blocked_without_loading_images(self):
        far = [dict(sample_id="far", decoded_rgb_sha256="same")]
        train = [dict(sample_id="train", decoded_rgb_sha256="same")]
        with patch.object(e10, "fingerprint", side_effect=AssertionError("Should stop before file reads")):
            with self.assertRaises(ValueError): e10.audit_training_overlap(far, train)

    def test_both_directions_and_zero_rescues(self):
        values = [(.2, .8), (.8, .2), (.1, .2), (.9, .9), (0, .2), (.3, 0)]
        rows = [dict(e0_dice=a, b_dice=b, e0_mae=.02, b_mae=.02) for a, b in values]
        result = e10.descriptive.pair_summary(rows)
        self.assertEqual(result["B_rescues_E0_low"], 1)
        self.assertEqual(result["E0_rescues_B_low"], 1)
        self.assertEqual(result["zero_rescued_by_B"], 1)
        self.assertEqual(result["zero_new_in_B"], 1)
        self.assertEqual(result["both_lt05"], 3)

    def test_cluster_bootstrap_preserves_images_within_patient(self):
        rows = [dict(group_id="patient1", dice_delta_B_minus_E0=.1)] * 3
        rows += [dict(group_id="patient2", dice_delta_B_minus_E0=-.1)]
        result = e10.bootstrap(rows, repetitions=2000, seed=42)
        self.assertEqual(result["group_count"], 2)
        self.assertEqual(result["image_count"], 4)
        self.assertAlmostEqual(result["mean_delta"], .05)
        self.assertAlmostEqual(result["ci95_low"], -.1)
        self.assertAlmostEqual(result["ci95_high"], .1)
        self.assertEqual(result, e10.bootstrap(rows, repetitions=2000, seed=42))
        with self.assertRaises(ValueError): e10.bootstrap([])

    def fixture(self):
        records = [dict(sample_id=str(i), source="FAR-POLYP-SEG", group_id=f"p{i//2}", declared_group_id=f"Patient{i//2}",
                        split="calibration" if i < 2 else "selection") for i in range(4)]
        e0 = [dict(**r, dataset=r["source"], dice=d, iou=d / 2, mae=.02, gt_area_fraction=.02)
              for r, d in zip(records, (.2, .8, .9, .2))]
        b = [dict(**r, dataset=r["source"], dice=d, iou=d / 2, mae=.01, gt_area_fraction=.02)
             for r, d in zip(records, (.8, .2, .9, .1))]
        return records, e0, b, {r["sample_id"]: .5 for r in records}

    def test_analysis_main_image_mean_and_secondary_patient_mean(self):
        records, e0, b, agreements = self.fixture()
        paired, summary, groups, pairs, intervals, patients, patient_summary = e10.build_analysis(records, e0, b, agreements)
        self.assertEqual(len(summary), 6)
        self.assertEqual(pairs["ALL"]["B_rescues_E0_low"], 1)
        self.assertEqual(pairs["ALL"]["E0_rescues_B_low"], 1)
        self.assertEqual(len(patients), 2)
        self.assertEqual(patient_summary[0]["patient_groups"], 2)
        self.assertEqual(len(paired), 4)
        self.assertEqual(intervals["selection"]["group_count"], 1)

    def test_unpaired_missing_and_duplicate_rows_blocked(self):
        records, e0, b, agreements = self.fixture()
        for invalid in (b[:-1], b + [b[0]]):
            with self.assertRaises(ValueError): e10.build_analysis(records, e0, invalid, agreements)

    def test_cuda_absent_does_not_fall_back(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError): e10.require_cuda()

    def test_existing_plan_not_overwritten_and_outputs_before_freeze_blocked(self):
        with tempfile.TemporaryDirectory(dir=e10.PACKAGE, prefix="_test_") as tmp:
            plan = Path(tmp) / "plan.json"; output = Path(tmp) / "output"
            plan.write_text("do not overwrite", encoding="utf-8")
            with patch.object(e10, "PLAN", plan), patch.object(e10, "OUTPUT", output), patch.object(e10, "check") as check:
                e10.freeze(); check.assert_called_once()
                self.assertEqual(plan.read_text(encoding="utf-8"), "do not overwrite")
            other = Path(tmp) / "missing.json"; output.mkdir()
            with patch.object(e10, "PLAN", other), patch.object(e10, "OUTPUT", output):
                with self.assertRaises(ValueError): e10.freeze()


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
