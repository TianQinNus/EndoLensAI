"""Synthetic CPU tests, never experiment training or holdout evaluation."""
import copy
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from common import ARMS, metric_summary, validation_score, read_json
from train_e8 import make_config, sampling_probabilities, per_image_loss, DrawDataset, train
from models import segmentation_loss
from engine import require_cuda
from prepare_data import fingerprint, near_matches, discover_single_frames
from download_polypgen import DownloadForm, safe_members
from run_e8 import development_gates, E0Adapter


class E8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def records(self):
        return [{"sample_id": str(i), "dataset": "original_training" if i < 10 else f"PolypGen-C{(i-10)//5+1}"}
                for i in range(30)]

    def test_matched_budget_and_same_initialization(self):
        configs = [make_config(arm) for arm in ARMS]
        for key in ("seed", "epochs", "samples_per_round", "input_size", "batch_size", "accumulation", "lr", "weight_decay", "encoder_weights", "loss", "augmentation"):
            self.assertEqual(len({str(c[key]) for c in configs}), 1)
        self.assertEqual(configs[0]["samples_per_round"] // 8, 145)

    def test_expansion_mass_and_no_zero_probability(self):
        records = self.records()
        p, boosted = sampling_probabilities(records, np.full(30, np.nan), make_config("B_expanded"), 1)
        self.assertAlmostEqual(p[:10].sum(), .5)
        for offset in (10, 15, 20, 25):
            self.assertAlmostEqual(p[offset:offset+5].sum(), .125)
        self.assertTrue(np.all(p > 0))
        self.assertEqual(boosted, 0)

    def test_hard_warmup_bounded_ratio_and_source_mass(self):
        records, losses = self.records(), np.arange(30, dtype=float)
        config = make_config("D_expanded_hard")
        warm, boosted = sampling_probabilities(records, losses, config, 20)
        self.assertEqual(boosted, 0)
        hard, boosted = sampling_probabilities(records, losses, config, 21)
        self.assertGreater(boosted, 0)
        self.assertAlmostEqual(hard[:10].sum(), .5)
        self.assertAlmostEqual(hard[:10].max()/hard[:10].min(), 2)
        self.assertFalse(np.array_equal(warm, hard))

    def test_unobserved_difficulty_keeps_normal_probability(self):
        records = self.records()[:10]
        losses = np.full(10, np.nan)
        losses[0] = 9.0
        p, boosted = sampling_probabilities(records, losses, make_config("C_hard"), 30)
        self.assertEqual(boosted, 0)
        self.assertTrue(np.allclose(p, .1))

    def test_loss_exactly_matches_existing_bce_dice(self):
        torch.manual_seed(2)
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = (torch.rand_like(logits) > .8).float()
        vector, bce, dice = per_image_loss(logits, target)
        old = segmentation_loss(logits, target)
        for a, b in zip((vector.mean(), bce.mean(), dice.mean()), old):
            torch.testing.assert_close(a, b)
        vector.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_macro_validation_not_image_weighted(self):
        rows = [{"dataset": "original_validation", "dice": .8, "iou": .5, "mae": .1}] * 10
        rows += [{"dataset": "PolypGen-C5", "dice": .4, "iou": .3, "mae": .2}]
        self.assertAlmostEqual(validation_score(rows), .6)
        self.assertNotAlmostEqual(metric_summary(rows)["dice"], .6)
        with self.assertRaises(ValueError):
            validation_score(rows[:10])

    def test_cuda_missing_never_falls_back(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                require_cuda()

    def test_development_gates_block_mean_only_improvement(self):
        base = [{"dataset": s, "dice": .8, "iou": .7, "mae": .02, "gt_area_fraction": .1}
                for s in ("original_validation", "PolypGen-C5")]
        rows = copy.deepcopy(base)
        for r in rows:
            r["dice"] += .01
        gain, gates = development_gates({"macro_dice": .81}, {"macro_dice": .8}, rows, base)
        self.assertTrue(all(gates.values()))
        rows[1]["mae"] = .03
        _, gates = development_gates({"macro_dice": .81}, {"macro_dice": .8}, rows, base)
        self.assertFalse(gates["PolypGen-C5_mae"])
        rows[1]["dice"] = .49
        _, gates = development_gates({"macro_dice": .81}, {"macro_dice": .8}, rows, base)
        self.assertFalse(gates["PolypGen-C5_low_dice"])

    def test_e0_adapter_uses_sum_of_logits_not_probability_average(self):
        class Stub(torch.nn.Module):
            def forward(self, x):
                return x[:, :1] * 2, x[:, :1] * 3
        x = torch.ones(1, 3, 4, 4)
        torch.testing.assert_close(E0Adapter(Stub())(x), torch.full((1,1,4,4), 5.0))

    def test_download_form_is_public_confirmation_not_auth(self):
        parser = DownloadForm()
        parser.feed('<form id="download-form" action="https://drive.usercontent.google.com/download"><input type="hidden" name="id" value="abc"></form>')
        self.assertEqual(parser.fields, {"id": "abc"})

    def test_zip_traversal_and_symlink_blocked(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="_test_") as tmp:
            root = Path(tmp)
            for name in ("../escape.txt", "C:/outside.txt", "/absolute.txt"):
                with zipfile.ZipFile(root / "test.zip", "w") as z:
                    z.writestr(name, b"no")
                with zipfile.ZipFile(root / "test.zip") as z, self.assertRaises(ValueError):
                    list(safe_members(z, root / "extract"))

    def test_near_duplicate_requires_rgb_confirmation(self):
        r = {"sample_id": "x", "dhash64": "0"*16, "width": 16, "height": 16, "decoded_rgb_sha256": "first"}
        other = dict(r, sample_id="y", decoded_rgb_sha256="second")
        thumb = np.zeros((16,16,3), np.uint8)
        self.assertEqual(near_matches(r, thumb, [other], [thumb]), [("y", 0.0)])
        self.assertEqual(near_matches(r, thumb, [other], [np.full_like(thumb, 255)]), [])

    def test_official_pairing_typo_and_missing_masks(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="_test_") as tmp:
            root = Path(tmp)
            for c in range(1,7):
                image = root / f"data_C{c}/images_C{c}"
                mask = root / f"data_C{c}/masks_C{c}"
                image.mkdir(parents=True)
                mask.mkdir()
                name = "C3_EndoCV2021_00489]" if c == 3 else "sample"
                (image / (name + ".jpg")).touch()
                (mask / (("C3_EndoCV2021_00489" if c == 3 else "sample") + "_mask.jpg")).touch()
            self.assertEqual(len(discover_single_frames(root)), 6)
            (root / "data_C1/images_C1/missing.jpg").touch()
            with self.assertRaises(ValueError):
                discover_single_frames(root)

    def test_cpu_integration_resume_restores_hard_scores_and_committed_best(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="_test_") as tmp:
            root = Path(tmp)
            a = np.zeros((32,32), np.uint8)
            a[6:16, 8:20] = 255
            Image.fromarray(np.stack([a]*3, axis=-1)).save(root / "x.png")
            Image.fromarray(a).save(root / "y.png")
            def record(i, source):
                return {"sample_id": str(i), "dataset": source, "image": "x.png", "mask": "y.png"}
            manifests = {"original_train": [record(i, "original_training") for i in range(6)],
                         "new_train": [record(10+i, "PolypGen-C1") for i in range(6)],
                         "validation": [record(20, "original_validation"), record(21, "PolypGen-C5")]}
            config = dict(make_config("D_expanded_hard", smoke=True), input_size=32, epochs=3, samples_per_round=16,
                          hard_warmup_rounds=0)
            # Small synthetic network tests checkpoint/control flow, not model performance.
            def tiny_model(*args, **kwargs):
                return torch.nn.Conv2d(3, 1, 1)
            bundle = {"bundle_signature": "synthetic-only"}
            with patch("train_e8.PROJECT", root), patch("train_e8.code_hashes", return_value={}), patch("train_e8.make_model", side_effect=tiny_model):
                with self.assertRaises(KeyboardInterrupt):
                    train(root / "run", config, bundle, manifests, workers=0, _test_device=torch.device("cpu"), _test_interrupt_epoch=1)
                cp = torch.load(root / "run/checkpoints/last.pth", weights_only=True)
                self.assertTrue(torch.isfinite(cp["loss_ema"]).any())
                self.assertEqual(cp["epoch"], 1)
                train(root / "run", config, bundle, manifests, workers=0, resume=True, _test_device=torch.device("cpu"))
                info = read_json(root / "run/run_info.json")
                self.assertEqual(info["status"], "TRAINED")
                self.assertEqual(info["completed_epochs"], 3)
                cp = torch.load(root / "run/checkpoints/last.pth", weights_only=True)
                self.assertEqual([r["round"] for r in cp["history"]], [1,2,3])
                self.assertGreater(cp["history"][-1]["hard_boosted_images"], 0)
                self.assertEqual(info["formal_holdout_evaluations"], 0)
                train(root / "uninterrupted", config, bundle, manifests, workers=0, _test_device=torch.device("cpu"))
                reference = torch.load(root / "uninterrupted/checkpoints/last.pth", weights_only=True)
                for name in cp["model_state"]:
                    torch.testing.assert_close(cp["model_state"][name], reference["model_state"][name], rtol=0, atol=0)
                torch.testing.assert_close(cp["loss_ema"], reference["loss_ema"], equal_nan=True, rtol=0, atol=0)
                self.assertEqual([r["sample_indices_sha256"] for r in cp["history"]],
                                 [r["sample_indices_sha256"] for r in reference["history"]])
                with self.assertRaises(ValueError):
                    train(root / "run", config, bundle, manifests, workers=0, resume=True, _test_device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
