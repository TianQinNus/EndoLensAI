"""Synthetic workflow tests. No real model training or C6 metrics are read."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_e8_repeat as repeat


class RepeatTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for seed in repeat.ALL_SEEDS:
            for index,arm in enumerate(repeat.ARMS):
                row={"arm":arm,"seed":seed}
                row.update({k:float(index)+seed/10000 for k in repeat.METRICS})
                rows.append(row)
        return rows

    def test_sample_sd_not_population_sd(self):
        mean,sd=repeat.mean_sd([1,2,3])
        self.assertEqual(mean,2)
        self.assertEqual(sd,1)

    def test_nonfinite_and_single_seed_blocked(self):
        for values in ([1],[1,float("nan")],[1,float("inf")]):
            with self.assertRaises(ValueError):repeat.mean_sd(values)

    def test_exact_12_runs_and_no_partial_summary(self):
        rows=self.fixture()
        summary=repeat.aggregate_seed_results(rows)
        self.assertEqual(len(summary),4)
        self.assertTrue(all(r["n_seeds"]==3 for r in summary))
        with self.assertRaises(ValueError):repeat.aggregate_seed_results(rows[:-1])
        with self.assertRaises(ValueError):repeat.aggregate_seed_results(rows+[rows[0]])
        wrong=copy.deepcopy(rows);wrong[0]["seed"]=99
        with self.assertRaises(ValueError):repeat.aggregate_seed_results(wrong)

    def test_paired_contrasts_use_same_seed(self):
        contrasts=repeat.paired_seed_contrasts(self.fixture())
        self.assertEqual(len(contrasts),12)
        b=[r for r in contrasts if r["contrast"]=="B_expanded-A_baseline"]
        self.assertEqual({r["seed"] for r in b},set(repeat.ALL_SEEDS))
        for r in b:self.assertAlmostEqual(r["macro_dice_delta"],1)

    def test_seed_changes_not_data_or_training_parameters(self):
        for arm in repeat.ARMS:
            a=repeat.e8.make_config(arm,42)
            b=repeat.e8.make_config(arm,17)
            c=repeat.e8.make_config(arm,2024)
            for row in (a,b,c):row.pop("seed")
            self.assertEqual(a,b);self.assertEqual(b,c)

    def test_completed_run_is_skipped_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,prefix="_test_") as tmp:
            root=Path(tmp)
            info={"config":{"seed":17},"source_hashes":{"x":"h"},"bundle_signature":"data", "run_kind":"E8_DEVELOPMENT","status":"TRAINED"}
            (root/"run_info.json").write_text(json.dumps(info),encoding="utf-8")
            self.assertEqual(repeat.continuation_action(root,info["config"],info["source_hashes"],"data"),"skip_verified")
            with self.assertRaises(ValueError):repeat.continuation_action(root,{"seed":2024},info["source_hashes"],"data")

    def test_interrupted_run_needs_last_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,prefix="_test_") as tmp:
            root=Path(tmp)
            info={"config":{},"source_hashes":{},"bundle_signature":"data","run_kind":"E8_DEVELOPMENT","status":"INTERRUPTED"}
            (root/"run_info.json").write_text(json.dumps(info),encoding="utf-8")
            with self.assertRaises(ValueError):repeat.continuation_action(root,{}, {},"data")
            (root/"checkpoints").mkdir();(root/"checkpoints/last.pth").touch()
            self.assertEqual(repeat.continuation_action(root,{}, {},"data"),"resume")

    def test_active_run_never_launches_second_process(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,prefix="_test_") as tmp:
            root=Path(tmp)
            info={"config":{},"source_hashes":{},"bundle_signature":"data","run_kind":"E8_DEVELOPMENT","status":"TRAINING"}
            (root/"run_info.json").write_text(json.dumps(info),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"may still be active"):
                repeat.continuation_action(root,{}, {},"data")

    def test_seed42_is_blocked_before_launch(self):
        with patch("run_e8_repeat.subprocess.run",side_effect=AssertionError("must not train")):
            with self.assertRaises(ValueError):repeat.start_one("B_expanded",42,2)

    def test_training_command_routes_to_unchanged_e8(self):
        info={"bundle_signature":"data"}
        config={"seed":17}
        protocol={"configs":{"B_expanded/seed17":config},"source_hashes":{}}
        with patch("run_e8_repeat.check",return_value=({},info,{},protocol)), \
             patch("run_e8_repeat.continuation_action",return_value="resume"), \
             patch("run_e8_repeat.subprocess.run") as command, \
             patch("run_e8_repeat.verified_run"):
            repeat.start_one_locked("B_expanded",17,0)
            args=command.call_args.args[0]
            self.assertIn("resume",args)
            self.assertIn("17",args)
            self.assertIn("B_expanded",args)
            self.assertNotIn("evaluate-holdout",args)
            self.assertTrue(any("e8_windows" in x for x in args))

    def test_serial_lock_releases_and_rejects_concurrent_launcher(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,prefix="_test_") as tmp:
            with patch("run_e8_repeat.PROJECT",Path(tmp)):
                with repeat.serial_training_lock():
                    with self.assertRaises((RuntimeError,OSError)):
                        with repeat.serial_training_lock():pass
                with repeat.serial_training_lock():pass

    def test_full_summary_and_audit_without_training_or_real_images(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,prefix="_test_") as tmp:
            root=Path(tmp)
            plan=root/"plan.json";plan.write_text("synthetic-only",encoding="utf-8")
            output=root/"summary"
            fixtures={}
            for seed in repeat.ALL_SEEDS:
                for index,arm in enumerate(repeat.ARMS):
                    path=root/f"{arm}_{seed}";(path/"checkpoints").mkdir(parents=True)
                    for name in ("run_info.json","checkpoints/best.pth","validation_best_per_image.csv","train_history.csv"):
                        (path/name).write_text("synthetic-only",encoding="utf-8")
                    rows=[dict(sample_id=f"{source}/x",dataset=source,dice=.6+index*.02,iou=.5+index*.01,mae=.03-index*.002,gt_area_fraction=.005)
                          for source in repeat.SOURCES]
                    metrics=repeat.metric_summary(rows)
                    record=dict(arm=arm,seed=seed,best_round=40,macro_dice=repeat.validation_score(rows),**metrics)
                    fixtures[arm,seed]=(path,{},rows,record)
            with patch("run_e8_repeat.PROJECT",root),patch("run_e8_repeat.PLAN_PATH",plan),patch("run_e8_repeat.OUTPUT",output), \
                 patch("run_e8_repeat.check",return_value=({}, {}, {}, {})), \
                 patch("run_e8_repeat.verified_run",side_effect=lambda a,s,*args:fixtures[a,s]), \
                 patch("run_e8_repeat.subprocess.run",side_effect=AssertionError("No training")):
                repeat.summarize()
                result=repeat.read_json(output/"stability.json")
                self.assertEqual(result["status"],"B_GATES_REPRODUCED_ALL_SEEDS")
                self.assertEqual(result["C6_evaluations"],0)
                self.assertFalse(result["best_seed_selection"])
                self.assertEqual(len(result["B_source_tail_details"]),6)
                self.assertEqual(len(repeat.read_csv(output/"per_seed.csv")),12)
                self.assertNotIn("|---|\n\n|",(output/"report.md").read_text(encoding="utf-8"))
                repeat.verify()
                with self.assertRaises(FileExistsError):repeat.summarize()
                (output/"mean_sd.csv").write_text("tampered",encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"Summary output changed"):repeat.verify()


if __name__=="__main__":
    unittest.main(verbosity=2)
