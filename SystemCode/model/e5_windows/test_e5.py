"""Checks for E5 arithmetic, provenance rejection, and end-to-end smoke output."""

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import run_e5 as e5


class E5Tests(unittest.TestCase):
    def test_equal_probability_and_threshold_boundary(self):
        e0 = np.array([[0.9, 0.2], [0.4, 0.1]], dtype=np.float32)
        e4 = np.array([[0.1, 0.8], [0.2, 0.3]], dtype=np.float32)
        probability = e5.fuse(e0, e4)
        self.assertEqual(probability.dtype, np.float32)
        np.testing.assert_allclose(
            probability,
            np.array([[0.5, 0.5], [0.3, 0.2]], dtype=np.float32),
        )
        ground_truth = np.array([[1, 0], [1, 0]], dtype=bool)
        result = e5.metrics(probability, ground_truth)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["dice"], 0.5)
        self.assertEqual(result["iou"], 1 / 3)
        with self.assertRaises(ValueError):
            e5.fuse(e0, e4[:1])

    def test_probability_integrity_and_empty_rules(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "probability.npz"
            invalid = [
                np.array([[np.nan]], dtype=np.float32),
                np.array([[1.1]], dtype=np.float32),
                np.zeros((1, 1), dtype=np.float64),
            ]
            for array in invalid:
                np.savez(path, probability=array)
                with self.assertRaises(ValueError):
                    e5.load_probability(path, (1, 1))
        empty_probability = np.zeros((2, 2), dtype=np.float32)
        empty_gt = np.zeros((2, 2), dtype=bool)
        self.assertEqual(e5.metrics(empty_probability, empty_gt)["dice"], 1)
        self.assertEqual(
            e5.metrics(np.ones((2, 2), dtype=np.float32), empty_gt)["dice"], 0
        )

    def fixture(self, root):
        e0_run = root / "runs" / "e0"
        e4_run = root / "runs" / "e4"
        e4_evaluation = e4_run / "evaluation"
        e3_run = root / "runs" / "e3"
        e0_run.mkdir(parents=True)
        e4_evaluation.mkdir(parents=True)
        (e4_run / "checkpoints").mkdir()
        e3_run.mkdir(parents=True)

        checkpoint = e4_run / "checkpoints" / "final.pth"
        checkpoint.write_bytes(b"fixture checkpoint; never a real model")
        checkpoint_hash = e5.digest(checkpoint)

        manifest = []
        references = {"e0": [], "e4": [], "e3": []}
        source_directories = {
            "e0": e0_run,
            "e4": e4_evaluation,
            "e3": e3_run,
        }
        foreground_values = {"e0": 0.8, "e4": 0.6, "e3": 0.7}
        background_values = {"e0": 0.1, "e4": 0.2, "e3": 0.15}

        for dataset, count in e5.COUNTS.items():
            for index in range(count):
                sample_id = f"{dataset}/{index}"
                rgb = np.full((2, 3, 3), index % 255, dtype=np.uint8)
                ground_truth = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
                image_path = Path("data") / (sample_id + ".png")
                mask_path = Path("gt") / (sample_id + ".png")
                for path in (root / image_path, root / mask_path):
                    path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(root / image_path)
                Image.fromarray(ground_truth.astype(np.uint8) * 255).save(
                    root / mask_path
                )
                manifest.append(
                    {
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "image": image_path.as_posix(),
                        "mask": mask_path.as_posix(),
                        "height": 2,
                        "width": 3,
                        "gt_foreground_pixels": 2,
                        "image_sha256": e5.digest(root / image_path),
                        "mask_sha256": e5.digest(root / mask_path),
                        "decoded_rgb_sha256": e5.hashlib.sha256(
                            rgb.tobytes()
                        ).hexdigest(),
                    }
                )

                for model, directory in source_directories.items():
                    probability = np.where(
                        ground_truth,
                        foreground_values[model],
                        background_values[model],
                    ).astype(np.float32)
                    probability_path = Path("probabilities") / (sample_id + ".npz")
                    result_mask_path = Path("masks") / (sample_id + ".png")
                    for path in (
                        directory / probability_path,
                        directory / result_mask_path,
                    ):
                        path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        directory / probability_path, probability=probability
                    )
                    Image.fromarray(
                        (probability >= e5.THRESHOLD).astype(np.uint8) * 255
                    ).save(directory / result_mask_path)
                    result = e5.metrics(probability, ground_truth)
                    row = {
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "height": 2,
                        "width": 3,
                        "probability_path": probability_path.as_posix(),
                        "mask_path": result_mask_path.as_posix(),
                    }
                    if model == "e3":
                        row.update({f"fusion_{key}": value for key, value in result.items()})
                    else:
                        row.update(result)
                    references[model].append(row)

        e5.write_json(e0_run / "manifest.json", manifest)
        e5.write_json(e4_run / "test_manifest.json", manifest)
        e5.write_json(e3_run / "manifest.json", manifest)
        e5.write_csv(e0_run / "per_image.csv", references["e0"])
        e5.write_csv(e4_evaluation / "per_image.csv", references["e4"])
        e5.write_csv(e3_run / "per_image.csv", references["e3"])

        e5.write_json(
            e0_run / "run_info.json",
            {
                "status": "COMPLETED",
                "run_kind": "FULL_E0",
                "processed_count": 798,
                "threshold": 0.5,
                "protocol": {"id": "endolens-e0-probability-v1"},
                "manifest_sha256": e5.digest(e0_run / "manifest.json"),
                "checkpoint_sha256": "fixture-e0",
            },
        )
        e5.write_json(
            e4_run / "run_info.json",
            {
                "status": "TRAINED",
                "run_kind": "E2_FINAL",
                "completed_epochs": 76,
                "config": {"augmentation": "strong"},
                "final_checkpoint_sha256": checkpoint_hash,
            },
        )
        e5.write_json(e4_run / "e4_stage.json", {"stage": "E4_FINAL"})
        e5.write_json(
            e4_evaluation / "run_info.json",
            {
                "status": "COMPLETED",
                "processed_count": 798,
                "checkpoint_sha256": checkpoint_hash,
                "protocol": {
                    "id": "endolens-e2-probability-v1",
                    "threshold": 0.5,
                    "per_image_minmax": False,
                },
            },
        )
        e5.write_json(
            e3_run / "run_info.json",
            {
                "status": "COMPLETED",
                "run_kind": "FULL_V3",
                "processed_count": 798,
                "e0_checkpoint_sha256": "fixture-e0",
                "config": {
                    "weight_e0": 0.5,
                    "weight_e2": 0.5,
                    "threshold": 0.5,
                    "per_image_minmax": False,
                    "probability_precision": "float32",
                    "parameter_search": False,
                },
            },
        )
        return e0_run, e4_run, e4_evaluation, e3_run

    def arguments(self, root, sources, output):
        e0_run, e4_run, e4_evaluation, e3_run = sources
        return argparse.Namespace(
            project=str(root),
            e0_run=str(e0_run),
            e4_run=str(e4_run),
            e4_eval=str(e4_evaluation),
            e3_run=str(e3_run),
            output=output,
            smoke=True,
            audit_only=False,
        )

    def test_end_to_end_smoke_and_reject_corruption(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            sources = self.fixture(root)
            args = self.arguments(root, sources, "runs/smoke")
            with contextlib.redirect_stdout(io.StringIO()):
                e5.run(args)

            output = root / "runs" / "smoke"
            info = e5.read_json(output / "run_info.json")
            self.assertEqual(info["run_kind"], "SMOKE_ONLY")
            self.assertEqual(info["processed_count"], 5)
            summary = e5.read_rows(output / "summary.csv")
            self.assertEqual(float(summary[0]["e5_dice"]), 1.0)
            self.assertEqual(float(summary[0]["e3_dice"]), 1.0)
            self.assertTrue((output / "comparison.html").is_file())
            self.assertEqual(len(list((output / "probabilities").rglob("*.npz"))), 5)
            with self.assertRaises(FileExistsError):
                e5.run(args)

            # A source replacement must fail rather than silently alter E5.
            Image.fromarray(np.zeros((2, 3), dtype=np.uint8)).save(
                root / "gt" / "Kvasir" / "0.png"
            )
            args.output = "runs/bad"
            with self.assertRaises(ValueError):
                e5.run(args)
            self.assertEqual(
                e5.read_json(root / "runs" / "bad" / "run_info.json")["status"],
                "FAILED",
            )

    def test_reject_wrong_e4_or_e3_provenance(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            sources = self.fixture(root)
            e0_run, e4_run, e4_evaluation, e3_run = sources

            stage = e5.read_json(e4_run / "e4_stage.json")
            stage["stage"] = "NOT_E4_FINAL"
            e5.write_json(e4_run / "e4_stage.json", stage)
            with self.assertRaises(ValueError):
                e5.validate_sources(root, e0_run, e4_run, e4_evaluation, e3_run)

            stage["stage"] = "E4_FINAL"
            e5.write_json(e4_run / "e4_stage.json", stage)
            e3_info = e5.read_json(e3_run / "run_info.json")
            e3_info["config"]["weight_e0"] = 0.75
            e5.write_json(e3_run / "run_info.json", e3_info)
            with self.assertRaises(ValueError):
                e5.validate_sources(root, e0_run, e4_run, e4_evaluation, e3_run)

    def test_safe_paths_and_duplicate_ids(self):
        with self.assertRaises(ValueError):
            e5.by_id([{"sample_id": "x"}, {"sample_id": "x"}])
        root = Path.cwd().resolve()
        for path in ("../escape", "C:/escape", "sub\\file"):
            with self.assertRaises(ValueError):
                e5.under(root, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
