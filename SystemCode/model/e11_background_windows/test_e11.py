"""Synthetic validation, loss-gradient and interruption tests; no real long training."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import run_e11 as e
from background_loss import loss_parts, enrich_rows


class BackgroundTests(unittest.TestCase):
    def test_only_loss_changes(self):
        for seed in e.SEEDS:
            new, old = e.make_config(seed), e.trainer.make_config("B_expanded", seed)
            for key in ("stage", "background_strength", "background_power", "background_detach", "background_weight_max"):
                new.pop(key)
            new["loss"] = old["loss"]
            self.assertEqual(new, old)
        with self.assertRaises(ValueError): e.make_config(99)

    def test_lambda_zero_exact_loss_and_gradient_identity(self):
        z = torch.tensor([[[[-3., 0., 2., 5.]]]], requires_grad=True)
        y = torch.tensor([[[[0., 1., 0., 1.]]]])
        old = e.trainer.per_image_loss(z, y)
        new = loss_parts(z, y, 0.)
        for a, b in zip(old, new): torch.testing.assert_close(a, b, rtol=0, atol=0)
        ga = torch.autograd.grad(old[0].sum(), z, retain_graph=True)[0]
        gb = torch.autograd.grad(new[0].sum(), z)[0]
        torch.testing.assert_close(ga, gb, rtol=0, atol=0)

    def test_weight_bound_detach_and_foreground_unchanged(self):
        z = torch.tensor([[[[-2., 2., -2., 2.]]]], requires_grad=True)
        y = torch.tensor([[[[0., 0., 1., 1.]]]])
        total, weighted, dice, base, extra = loss_parts(z, y)
        gradient = torch.autograd.grad(weighted.sum(), z)[0]
        p = z.detach().sigmoid()
        weights = 1+(1-y)*p.square()
        torch.testing.assert_close(gradient, weights*(p-y)/4, rtol=1e-6, atol=1e-7)
        self.assertTrue(bool(((weights >= 1)&(weights <= 2)).all()))
        torch.testing.assert_close(weights[y == 1], torch.ones(2))
        self.assertGreater(float(weights[0,0,0,1]), float(weights[0,0,0,0]))

    def test_extreme_and_empty_targets_finite_fp32(self):
        for target in (torch.zeros(1,1,2,2), torch.ones(1,1,2,2)):
            z = torch.tensor([[[[-100., 100.], [-20., 20.]]]], dtype=torch.float16, requires_grad=True)
            parts = loss_parts(z, target)
            self.assertEqual(parts[0].dtype, torch.float32)
            self.assertTrue(all(bool(torch.isfinite(x).all()) for x in parts))
            parts[0].sum().backward()
            self.assertTrue(bool(torch.isfinite(z.grad).all()))
        with self.assertRaises(ValueError): loss_parts(torch.zeros(1,1,2,2), torch.full((1,1,2,2), .3))
        with self.assertRaises(ValueError): loss_parts(torch.zeros(1,1,2,2), torch.zeros(1,1,2,2), 2.)

    def test_sampler_draws_and_difficulty_unchanged(self):
        records = [{"dataset": source} for source in ("original_training", "PolypGen-C1", "PolypGen-C2", "PolypGen-C3", "PolypGen-C4") for _ in range(3)]
        for seed in e.SEEDS:
            new = e.make_config(seed)
            old = e.trainer.make_config("B_expanded", seed)
            p, boost = e.trainer.sampling_probabilities(records, np.arange(15), new, 80)
            q, _ = e.trainer.sampling_probabilities(records, np.full(15, np.nan), old, 80)
            np.testing.assert_array_equal(p, q)
            self.assertEqual(boost, 0)
            rng1 = np.random.default_rng(np.random.SeedSequence([seed,80,808]))
            rng2 = np.random.default_rng(np.random.SeedSequence([seed,80,808]))
            np.testing.assert_array_equal(rng1.choice(15,1160,p=p), rng2.choice(15,1160,p=q))

    def test_adapter_restores_on_error(self):
        original = (e.trainer.per_image_loss, e.trainer.code_hashes, e.trainer.write_json, e.trainer.evaluate_records)
        with self.assertRaises(RuntimeError):
            with e.training_adapter():
                self.assertIsNot(e.trainer.per_image_loss, original[0])
                raise RuntimeError("synthetic")
        self.assertEqual(original, (e.trainer.per_image_loss, e.trainer.code_hashes, e.trainer.write_json, e.trainer.evaluate_records))

    def rows(self, better=False):
        result=[]
        for i, source in enumerate(e.SOURCES):
            tp, fp, fn = (90, 5, 10) if better else (90, 15, 10)
            result.append({"sample_id": str(i), "dataset": source, "height": 100, "width": 100,
                           "tp": tp, "fp": fp, "fn": fn, "dice": 2*tp/(2*tp+fp+fn),
                           "iou": tp/(tp+fp+fn), "mae": .01, "gt_area_fraction": .005})
        return result

    def test_validation_counts_and_empty_prediction(self):
        rows = self.rows()
        enriched = e.validate_rows(rows, rows)
        self.assertEqual(enriched[0]["recall"], .9)
        self.assertEqual(enriched[0]["fp_image_fraction"], .0015)
        empty = dict(rows[0], tp=0, fp=0, fn=100)
        self.assertEqual(enrich_rows([empty])[0]["precision"], 0.)
        self.assertEqual(enrich_rows([empty])[0]["recall"], 0.)
        for bad in (rows[:1], rows+[rows[0]]):
            with self.assertRaises(ValueError): e.validate_rows(bad, rows)
        bad = copy.deepcopy(rows); bad[0]["dice"] = .99
        with self.assertRaises(ValueError): e.validate_rows(bad, rows)
        with self.assertRaises(ValueError): enrich_rows([dict(rows[0], tp=0, fn=0)])

    def test_gates_require_dice_and_fp_not_safety_alone(self):
        old = self.rows()
        gates, _ = e.development_gates(old, old)
        self.assertFalse(gates["macro_dice_gain"])
        self.assertFalse(gates["macro_fp_reduced_5pct"])
        gates, _ = e.development_gates(self.rows(True), old)
        self.assertTrue(all(gates.values()))

    def test_recall_and_source_failure_protected(self):
        old = self.rows()
        new = self.rows(True)
        new[0].update(tp=80, fn=20, dice=160/185, iou=80/105)
        gates, _ = e.development_gates(new, old)
        self.assertFalse(gates["original_validation_recall"])
        self.assertFalse(gates["small_target_recall_protected"])
        new = self.rows(True)
        new[0].update(tp=0, fn=100, fp=0, dice=0., iou=0.)
        gates, _ = e.development_gates(new, old)
        self.assertFalse(gates["original_validation_zero_dice"])

    def test_all_seeds_no_cherry_pick_and_no_auto_test(self):
        with self.assertRaises(ValueError): e.complete_decision([{"seed":42,"gates":{"ok":True}}])
        rows = [{"seed":s,"gates":{"ok":True}} for s in e.SEEDS]
        self.assertEqual(e.complete_decision(rows)["selected_recipe"], e.ARM)
        rows[1]["gates"]["ok"] = False
        decision = e.complete_decision(rows)
        self.assertEqual(decision["selected_recipe"], "B_expanded")
        self.assertFalse(decision["formal_test_allowed"])

    def test_continuation_does_not_overwrite_active_run(self):
        with tempfile.TemporaryDirectory(dir=e.PACKAGE,prefix="_test_") as tmp:
            path = Path(tmp)/"run"
            self.assertEqual(e.continuation_action(path,{}, {}, "bundle"), "train")
            path.mkdir()
            info={"config":{},"source_hashes":{},"bundle_signature":"bundle","status":"TRAINING"}
            e.write_json(path/"run_info.json",info)
            with self.assertRaises(ValueError): e.continuation_action(path,{}, {}, "bundle")
            info["status"]="INTERRUPTED"; e.write_json(path/"run_info.json",info)
            with self.assertRaises(ValueError): e.continuation_action(path,{}, {}, "bundle")
            (path/"checkpoints").mkdir(); (path/"checkpoints/last.pth").touch()
            self.assertEqual(e.continuation_action(path,{}, {}, "bundle"), "resume")
            with self.assertRaises(ValueError): e.continuation_action(path,{"changed":True}, {}, "bundle")

    def test_band_diagnosis_not_near_boundary_only(self):
        from diagnose_errors import band_diagnosis
        gt = np.zeros((352,352),bool); gt[150:170,150:170]=True
        pred = gt.copy(); pred[20:40,20:40]=True
        self.assertEqual(band_diagnosis(pred,gt)["far_fp_fraction_at352"],1.)

    def test_cpu_integration_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory(dir=e.PACKAGE,prefix="_test_") as tmp:
            root=Path(tmp)
            mask=np.zeros((32,32),np.uint8);mask[6:16,8:20]=255
            Image.fromarray(np.stack([mask]*3,axis=-1)).save(root/"x.png")
            Image.fromarray(mask).save(root/"y.png")
            def record(i,s):return {"sample_id":str(i),"dataset":s,"image":"x.png","mask":"y.png"}
            manifests={"original_train":[record(i,"original_training") for i in range(3)],
                       "new_train":[record(10+i,f"PolypGen-C{i+1}") for i in range(4)],
                       "validation":[record(20,"original_validation"),record(21,"PolypGen-C5")]}
            config=dict(e.make_config(42,True),input_size=32,epochs=3,samples_per_round=16)
            bundle={"bundle_signature":"synthetic"}
            def tiny(*args,**kwargs):return torch.nn.Conv2d(3,1,1)
            with patch.object(e.trainer,"PROJECT",root), patch.object(e.trainer,"make_model",side_effect=tiny), patch.object(e,"stage_hashes",return_value={}):
                with e.training_adapter():
                    with self.assertRaises(KeyboardInterrupt):
                        e.trainer.train(root/"interrupted",config,bundle,manifests,workers=0,_test_device=torch.device("cpu"),_test_interrupt_epoch=1)
                    e.trainer.train(root/"interrupted",config,bundle,manifests,workers=0,resume=True,_test_device=torch.device("cpu"))
                with e.training_adapter():
                    e.trainer.train(root/"full",config,bundle,manifests,workers=0,_test_device=torch.device("cpu"))
            a=torch.load(root/"interrupted/checkpoints/last.pth",weights_only=True)
            b=torch.load(root/"full/checkpoints/last.pth",weights_only=True)
            for key in a["model_state"]:torch.testing.assert_close(a["model_state"][key],b["model_state"][key],rtol=0,atol=0)
            self.assertEqual([r["sample_indices_sha256"] for r in a["history"]],[r["sample_indices_sha256"] for r in b["history"]])
            info=e.read_json(root/"interrupted/run_info.json")
            self.assertEqual(info["run_kind"],"E11_GPU_SMOKE")
            self.assertEqual(info["loss_diagnostics_last_round"]["samples"],16)
            self.assertGreater(info["loss_diagnostics_last_round"]["background_extra"],0.)
            rows=e.repeat.read_csv(root/"interrupted/validation_best_per_image.csv")
            self.assertIn("recall",rows[0]);self.assertEqual(info["formal_holdout_evaluations"],0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
