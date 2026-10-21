"""Unit tests for E7-A affine calibration, bootstrap, safety, and selection."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_e7a as e7a

e6 = e7a.e6


class E7ATests(unittest.TestCase):
    def test_affine_identity_and_monotonicity(self):
        probability = np.array([[0.0, 0.1, 0.5, 0.9, 1.0]], dtype=np.float32)
        calibrated = e7a.affine_calibrate(probability, 1.0, 0.0)
        np.testing.assert_allclose(calibrated, np.clip(probability, 1e-6, 1 - 1e-6), atol=2e-6)
        self.assertTrue(np.all(np.diff(calibrated[0]) > 0))
        with self.assertRaises(ValueError):
            e7a.affine_calibrate(probability, 0.0, 0.0)

    def test_affine_fit_improves_nll(self):
        rng = np.random.default_rng(42)
        base = np.linspace(0.01, 0.99, 12000, dtype=np.float32)
        true_probability = e7a.sigmoid(1.7 * e6.probability_to_logit(base) - 0.6)
        targets = (rng.random(base.size) < true_probability).astype(np.float64)
        fit = e7a.fit_affine(base, targets)
        self.assertLess(fit["nll_after"], fit["nll_before"])
        self.assertAlmostEqual(fit["slope"], 1.7, delta=0.15)
        self.assertAlmostEqual(fit["intercept"], -0.6, delta=0.12)

    def test_cluster_bootstrap_is_deterministic(self):
        rows = [
            {"group_id": f"patient_{index // 2}", "delta_dice_vs_e0": index / 100.0 - 0.03}
            for index in range(20)
        ]
        first = e7a.cluster_bootstrap(rows, seed=42, repetitions=2000)
        second = e7a.cluster_bootstrap(rows, seed=42, repetitions=2000)
        self.assertEqual(first, second)
        self.assertEqual(first["group_count"], 10)

    def test_no_pass_config_blocks_formal_test(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            project = Path(directory)
            config = {
                "experiment": "E7A_POST_FUSION_AFFINE_CALIBRATION",
                "validation_status": "NO_PASS",
            }
            config["config_signature"] = e6.signature(config)
            config_path = project / "config.json"
            e6.write_json(config_path, config)
            arguments = argparse.Namespace(
                project=str(project),
                config=str(config_path),
                output="runs/final",
                e0_run="runs/e0",
                e4_run="runs/e4",
                e4_eval="runs/e4/eval",
                allow_exploratory=False,
            )
            with self.assertRaisesRegex(ValueError, "formal test is blocked"):
                e7a.command_apply_test(arguments)
            self.assertFalse((project / "runs/final").exists())

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
            for index in range(12):
                rgb = np.full((10, 11, 3), index * 9, dtype=np.uint8)
                mask = np.zeros((10, 11), dtype=np.uint8)
                mask[1 + index % 2 : 6 + index % 2, 2:8] = 255
                Image.fromarray(rgb).save(dataset / "images" / f"{index}.png")
                Image.fromarray(mask).save(dataset / "masks" / f"{index}.png")
            prepare_args = argparse.Namespace(
                project=str(project),
                dataset=str(dataset),
                metadata=None,
                output="runs/prepared",
                e0_run=str(e0_run),
                e4_run=str(e4_run),
                calibration_fraction=0.4,
                seed=42,
            )
            e6.command_prepare(prepare_args)
            manifest_run = project / "runs/prepared"
            manifest_info = e6.read_json(manifest_run / "run_info.json")
            records = e6.read_json(manifest_run / "development_manifest.json")

            inference = project / "runs/inference"
            inference.mkdir()
            probability_rows = []
            for record in records:
                ground_truth = e6.read_mask(project / record["mask"])
                for model in ("e0", "e4"):
                    foreground = 0.72 if model == "e0" else 0.88
                    background = 0.08 if model == "e0" else 0.03
                    probability = np.where(ground_truth, foreground, background).astype(np.float32)
                    relative = Path("probabilities") / model / (record["sample_id"] + ".npz")
                    path = inference / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, probability=probability)
                    probability_rows.append(
                        {
                            "sample_id": record["sample_id"],
                            "split": record["split"],
                            "source": record["source"],
                            "model": model,
                            "height": 10,
                            "width": 11,
                            "probability_path": relative.as_posix(),
                            "probability_sha256": e6.file_hash(path),
                        }
                    )
            e6.write_csv(inference / "per_image.csv", probability_rows)
            inference_info = {
                "status": "COMPLETED",
                "run_kind": "FULL_DEVELOPMENT",
                "processed_count": len(records),
                "manifest_signature": manifest_info["manifest_signature"],
                "e0_checkpoint_sha256": "e0",
                "e4_checkpoint_sha256": "e4",
            }
            e6.write_json(inference / "run_info.json", inference_info)

            e6_selection = project / "runs/e6_selection"
            e6_selection.mkdir()
            old_config = {
                "validation_status": "NO_PASS",
                "development_manifest_signature": manifest_info["manifest_signature"],
                "e0_checkpoint_sha256": "e0",
                "e4_checkpoint_sha256": "e4",
            }
            old_config["config_signature"] = e6.signature(old_config)
            e6.write_json(e6_selection / "selection_config.json", old_config)
            e6.write_json(
                e6_selection / "run_info.json",
                {"status": "COMPLETED", "selection_status": "NO_PASS"},
            )

            arguments = argparse.Namespace(
                project=str(project),
                manifest_run=str(manifest_run),
                inference_run=str(inference),
                e6_selection_run=str(e6_selection),
                output="runs/e7a_selection",
                seed=42,
                max_pixels_per_image=8192,
                bootstrap_repetitions=2000,
            )
            e7a.command_select(arguments)
            output = project / "runs/e7a_selection"
            config = e6.read_json(output / "selection_config.json")
            recorded_signature = config.pop("config_signature")
            self.assertEqual(e6.signature(config), recorded_signature)
            self.assertEqual(len(e6.read_csv(output / "grid_search.csv")), 91)
            self.assertEqual(len(e6.read_csv(output / "affine_calibration.csv")), 13)
            self.assertEqual(e6.read_json(output / "run_info.json")["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
