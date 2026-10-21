"""Synthetic tests; no real outcomes, GPU inference or protocol writes."""
import unittest
import numpy as np
from rescue import (FEATURES, BASE_FEATURES, components, candidate_regions, region_features,
                    supplement, isolated_label, fit_gate, score_gate, discrimination, component_weights)


class RescueTests(unittest.TestCase):
    def test_eight_connected_and_empty(self):
        mask = np.eye(4, dtype=bool)
        self.assertEqual(len(components(mask)), 1)
        self.assertEqual(components(np.zeros((3, 5), bool)), [])

    def test_connected_components_against_flood_fill(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            mask = rng.random((11, 13)) < .25
            seen, expected = set(), []
            for y, x in zip(*np.nonzero(mask)):
                if (y, x) in seen:
                    continue
                stack, part = [(y, x)], set()
                seen.add((y, x))
                while stack:
                    a, b = stack.pop()
                    part.add(a * mask.shape[1] + b)
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            q = (a + dy, b + dx)
                            if 0 <= q[0] < 11 and 0 <= q[1] < 13 and mask[q] and q not in seen:
                                seen.add(q)
                                stack.append(q)
                expected.append(part)
            actual = [set(a.tolist()) for a in components(mask)]
            self.assertEqual({frozenset(a) for a in actual}, {frozenset(a) for a in expected})

    def test_non_erasing_and_identity(self):
        pb = np.array([[.9, .4, .1], [.2, .7, .1]], np.float32)
        p0 = np.array([[.01, .8, .7], [.6, .2, .1]], np.float32)
        regions = candidate_regions(pb, p0)
        q = supplement(pb, p0, regions)
        self.assertTrue(np.all(q >= pb))
        self.assertTrue(np.all(q[pb >= .5] >= .5))
        np.testing.assert_array_equal(supplement(pb, p0, []), pb)
        np.testing.assert_array_equal(q >= .5, (pb >= .5) | (p0 >= .5))

    def test_feature_no_gt_and_flip_alignment(self):
        pb = np.full((3, 5), .1, np.float32)
        p0 = np.zeros_like(pb)
        p0[0, 1:3] = .9
        recovered = np.flip(np.flip(p0, axis=1), axis=1).copy()
        feature = region_features(pb, p0, [pb, pb], [recovered, p0], np.array([1, 2]))
        self.assertEqual(set(feature), set(FEATURES))
        self.assertEqual(feature["e0_flip_retention_min"], 1.)

    def test_label_and_mae_use_same_probability(self):
        pb = np.array([[.9, .1, .1, .1]], np.float32)
        p0 = np.array([[.1, .9, .1, .9]], np.float32)
        gt = np.array([[True, True, False, False]])
        useful = isolated_label(pb, p0, gt, np.array([1]))
        harmful = isolated_label(pb, p0, gt, np.array([3]))
        self.assertEqual(useful["helpful"], 1)
        self.assertEqual(harmful["helpful"], 0)
        q = supplement(pb, p0, [np.array([1])])
        delta = np.abs(q.astype(np.float64) - gt).mean() - np.abs(pb.astype(np.float64) - gt).mean()
        self.assertAlmostEqual(delta, useful["isolated_mae_delta"], places=12)

    def test_positive_nonzero_preserved_but_dice_can_worsen(self):
        pb = np.array([[.9, .1, .1, .1]], np.float32)
        p0 = np.array([[0., .9, .9, .9]], np.float32)
        gt = np.array([[True, False, False, False]])
        q = supplement(pb, p0, candidate_regions(pb, p0))
        self.assertEqual(int(((q >= .5) & gt).sum()), 1)
        self.assertGreater(int(((q >= .5) & ~gt).sum()), 0)

    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            candidate_regions(np.zeros((2, 2)), np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            candidate_regions(np.zeros((2, 2), np.float32), np.full((2, 2), np.nan, np.float32))

    def synthetic(self):
        rng = np.random.default_rng(7)
        rows = []
        for i in range(80):
            values = rng.normal(size=len(FEATURES))
            rows.append({"sample_id": str(i), "group_id": str(i // 2), "helpful": int(values[0] > 0),
                         **dict(zip(FEATURES, values))})
        return rows

    def test_fixed_fit_deterministic_and_ablation(self):
        rows = self.synthetic()
        full = fit_gate(rows)
        self.assertEqual(full, fit_gate(rows))
        self.assertGreater(np.mean((score_gate(rows, full) >= .5) == [r["helpful"] for r in rows]), .9)
        self.assertEqual(fit_gate(rows, BASE_FEATURES)["features"], list(BASE_FEATURES))

    def test_heldout_gt_cannot_affect_score(self):
        rows = self.synthetic()
        fitted = fit_gate(rows[:60])
        altered = [dict(r, helpful=1-r["helpful"], added_tp=999999, isolated_dice_delta=100.) for r in rows[60:]]
        np.testing.assert_array_equal(score_gate(rows[60:], fitted), score_gate(altered, fitted))
        # Held-out feature changes do not alter frozen train-only normalization.
        self.assertEqual(fitted, fit_gate(rows[:60]))

    def test_component_weights_equal_patients_images(self):
        rows = [{"sample_id": "a", "group_id": "p"}] * 4 + [{"sample_id": "b", "group_id": "p"}, {"sample_id": "c", "group_id": "q"}]
        weights = component_weights(rows)
        self.assertAlmostEqual(weights[:5].sum(), .5)
        self.assertAlmostEqual(weights[:4].sum(), .25)
        self.assertAlmostEqual(weights[5], .5)

    def test_empty_singleclass_and_auc_ties(self):
        self.assertEqual(fit_gate([])["constant"], 0.)
        rows = self.synthetic()
        same = [dict(r, helpful=0) for r in rows]
        self.assertLess(fit_gate(same)["constant"], .5)
        self.assertEqual(discrimination([], "score")["weighted_auc"], None)
        scored = [dict(r, score=.5) for r in rows]
        self.assertAlmostEqual(discrimination(scored, "score")["weighted_auc"], .5)

    def test_patient_folds_and_acceptance_not_tuned(self):
        import run_e10d as d
        records = [{"sample_id": str(i), "group_id": str(i // 2), "cv_fold": (i // 2) % 5} for i in range(80)]
        rows = [dict(a, cv_fold=b["cv_fold"]) for a, b in zip(self.synthetic(), records)]
        oof, fits = d.crossfit(rows, records)
        self.assertEqual(len(oof), 80)
        for fit in fits:
            self.assertFalse(set(fit["training_groups"]) & set(fit["heldout_groups"]))
            self.assertFalse(set(fit["training_ids"]) & set(fit["heldout_ids"]))
        self.assertEqual(d.POLICY["accept_score"], .8)
        self.assertEqual(d.POLICY["ridge_mean_nll"], .01)

    def test_no_admission_is_not_fusion_pass(self):
        import run_e10d as d
        rows = [{"model": label, "sample_id": str(fold), "cv_fold": fold,
                 "dice": .8, "iou": .7, "mae": .02} for fold in range(5) for label in d.LABELS]
        _, _, decision = d.assess(rows, [])
        self.assertEqual(decision["status"], "NO_FEASIBILITY_PASS_RETAIN_B")
        self.assertFalse(decision["selection_allowed"])
        self.assertFalse(decision["full_data_refit_allowed"])

    def test_replay_duplicate_and_metric_tamper_rejected(self):
        import run_e10d as d
        row = {"model": "x", "sample_id": "a", "dice": .8, "group_id": "p"}
        d.compare_rows([dict(row, dice="0.8")], [row])
        with self.assertRaises(ValueError):
            d.compare_rows([row, row], [row, row])
        with self.assertRaises(ValueError):
            d.compare_rows([dict(row, dice=.9)], [row])

    def test_per_model_backend_matches_historical_producer(self):
        import run_e10d as d
        import torch
        original = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            self.assertFalse(d.configure_backend(torch, d.c.b.B)["cudnn_allow_tf32"])
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertTrue(d.configure_backend(torch, d.c.b.E0)["cudnn_allow_tf32"])
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        finally:
            (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
