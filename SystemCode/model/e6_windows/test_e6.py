"""Unit tests for E6 calibration, gating, safety, and statistics."""

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import run_e6 as e6


class E6Tests(unittest.TestCase):
    def test_calibration_identity_and_boundaries(self):
        probability = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
        calibrated = e6.calibrate_probability(probability, 1.0)
        np.testing.assert_allclose(calibrated, np.clip(probability, 1e-6, 1 - 1e-6), atol=2e-6)
        self.assertTrue(np.all(np.diff(calibrated[0]) > 0))
        with self.assertRaises(ValueError):
            e6.calibrate_probability(probability, 0)

    def test_temperature_fit_improves_nll(self):
        probability = np.array([0.001, 0.01, 0.1, 0.9, 0.99, 0.999] * 200, dtype=np.float32)
        target = np.array([0, 1, 0, 1, 0, 1] * 200, dtype=np.float64)
        fitted = e6.fit_temperature([(e6.probability_to_logit(probability), target)])
        self.assertGreater(fitted["temperature"], 1.0)
        self.assertLessEqual(fitted["nll_after"], fitted["nll_before"])

    def test_probability_metrics_threshold_boundary(self):
        probability = np.array([[0.5, 0.49], [0.8, 0.1]], dtype=np.float32)
        ground_truth = np.array([[1, 0], [0, 0]], dtype=bool)
        metrics = e6.probability_metrics(probability, ground_truth, 0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 0))
        self.assertAlmostEqual(metrics["dice"], 2 / 3)

    def test_gate_and_selection(self):
        baseline = {
            "count": 100,
            "dice": 0.80,
            "iou": 0.70,
            "mae": 0.05,
            "median_dice": 0.9,
            "dice_lt05": 10,
            "zero_dice": 2,
            "source_dice": {"A": 0.80, "B": 0.80},
        }
        passing = {
            "count": 100,
            "dice": 0.805,
            "iou": 0.71,
            "mae": 0.052,
            "median_dice": 0.9,
            "dice_lt05": 10,
            "zero_dice": 2,
            "source_dice": {"A": 0.805, "B": 0.804},
            "alpha": 0.2,
            "threshold": 0.5,
        }
        failing = dict(passing, dice=0.801, alpha=0.1)
        passing = e6.gate_candidate(passing, baseline, e6.DEFAULT_GATES)
        failing = e6.gate_candidate(failing, baseline, e6.DEFAULT_GATES)
        selected, status = e6.choose_candidate([failing, passing])
        self.assertEqual(status, "PASS")
        self.assertEqual(selected["alpha"], 0.2)

    def test_group_split_has_no_leakage(self):
        records = []
        for index in range(12):
            records.append(
                {
                    "sample_id": f"S/{index}",
                    "source": "S",
                    "declared_group_id": str(index // 2),
                    "height": 2,
                    "width": 2,
                    "decoded_rgb_sha256": str(index),
                    "area_bin": "medium_1to10pct",
                }
            )
        split = e6.split_records(records, 0.4, 42)
        groups = {}
        for row in split:
            groups.setdefault(row["declared_group_id"], set()).add(row["split"])
        self.assertTrue(all(len(value) == 1 for value in groups.values()))
        self.assertEqual({row["split"] for row in split}, {"calibration", "selection"})

    def test_safe_paths_and_probability_integrity(self):
        root = Path.cwd().resolve()
        for relative in ("../escape", "C:/escape", "sub\\file"):
            with self.assertRaises(ValueError):
                e6.safe_under(root, relative)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "p.npz"
            np.savez(path, probability=np.zeros((2, 2), dtype=np.float64))
            with self.assertRaises(ValueError):
                e6.load_probability(path)

    def test_bootstrap_is_deterministic(self):
        delta = np.linspace(-0.1, 0.2, 30)
        first = e6.paired_bootstrap(delta, seed=42, repetitions=2000)
        second = e6.paired_bootstrap(delta, seed=42, repetitions=2000)
        self.assertEqual(first, second)

    def test_prepare_end_to_end_and_overlap_rejection(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            project = Path(directory)
            dataset = project / "dataset/E6Development"
            (dataset / "images").mkdir(parents=True)
            (dataset / "masks").mkdir()
            e0_run = project / "runs/e0"
            e4_run = project / "runs/e4"
            e0_run.mkdir(parents=True)
            e4_run.mkdir(parents=True)
            e6.write_json(e0_run / "manifest.json", [])
            e6.write_json(e4_run / "full_train_manifest.json", [])
            for index in range(10):
                rgb = np.full((8, 9, 3), index * 10, dtype=np.uint8)
                mask = np.zeros((8, 9), dtype=np.uint8)
                mask[1:4, 2:5] = 255
                Image.fromarray(rgb).save(dataset / "images" / f"{index}.png")
                Image.fromarray(mask).save(dataset / "masks" / f"{index}.png")
            arguments = argparse.Namespace(
                project=str(project),
                dataset=str(dataset),
                metadata=None,
                output="runs/prepared",
                e0_run=str(e0_run),
                e4_run=str(e4_run),
                calibration_fraction=0.4,
                seed=42,
            )
            e6.command_prepare(arguments)
            info = e6.read_json(project / "runs/prepared/run_info.json")
            self.assertEqual(info["status"], "COMPLETED")
            self.assertEqual(info["counts"], {"calibration": 4, "selection": 6})

    def test_selection_end_to_end(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            project = Path(directory)
            dataset = project / "dataset/E6Development"
            (dataset / "images").mkdir(parents=True)
            (dataset / "masks").mkdir()
            e0_run = project / "runs/e0"
            e4_run = project / "runs/e4"
            e0_run.mkdir(parents=True)
            e4_run.mkdir(parents=True)
            e6.write_json(e0_run / "manifest.json", [])
            e6.write_json(e4_run / "full_train_manifest.json", [])
            for index in range(10):
                rgb = np.full((8, 9, 3), index * 11, dtype=np.uint8)
                mask = np.zeros((8, 9), dtype=np.uint8)
                mask[1:5, 2:6] = 255
                Image.fromarray(rgb).save(dataset / "images" / f"{index}.png")
                Image.fromarray(mask).save(dataset / "masks" / f"{index}.png")
            prepare_args = argparse.Namespace(
                project=str(project), dataset=str(dataset), metadata=None,
                output="runs/prepared", e0_run=str(e0_run), e4_run=str(e4_run),
                calibration_fraction=0.4, seed=42,
            )
            e6.command_prepare(prepare_args)
            manifest_run = project / "runs/prepared"
            manifest_info = e6.read_json(manifest_run / "run_info.json")
            records = e6.read_json(manifest_run / "development_manifest.json")
            inference = project / "runs/inference"
            inference.mkdir()
            rows = []
            for record in records:
                ground_truth = e6.read_mask(project / record["mask"])
                for model in ("e0", "e4"):
                    probability = np.where(
                        ground_truth,
                        0.65 if model == "e0" else 0.85,
                        0.15 if model == "e0" else 0.05,
                    ).astype(np.float32)
                    relative = Path("probabilities") / model / (record["sample_id"] + ".npz")
                    path = inference / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, probability=probability)
                    rows.append(
                        {
                            "sample_id": record["sample_id"], "split": record["split"],
                            "source": record["source"], "model": model,
                            "height": 8, "width": 9, "probability_path": relative.as_posix(),
                            "probability_sha256": e6.file_hash(path),
                        }
                    )
            e6.write_csv(inference / "per_image.csv", rows)
            e6.write_json(
                inference / "run_info.json",
                {
                    "status": "COMPLETED", "run_kind": "FULL_DEVELOPMENT",
                    "processed_count": len(records),
                    "manifest_signature": manifest_info["manifest_signature"],
                    "e0_checkpoint_sha256": "e0", "e4_checkpoint_sha256": "e4",
                },
            )
            select_args = argparse.Namespace(
                project=str(project), manifest_run=str(manifest_run),
                inference_run=str(inference), output="runs/selection",
                seed=42, max_pixels_per_image=8192,
            )
            e6.command_select(select_args)
            config = e6.read_json(project / "runs/selection/selection_config.json")
            recorded = config.pop("config_signature")
            self.assertEqual(e6.signature(config), recorded)
            self.assertEqual(len(e6.read_csv(project / "runs/selection/grid_search.csv")), 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
