"""Synthetic tests; no real C6 probabilities or labels are read."""
import unittest
import numpy as np
from analyze_holdout import failure_category, pair_summary, paired_contrast


class HoldoutAnalysisTests(unittest.TestCase):
    def test_categories_and_boundary(self):
        self.assertEqual(failure_category(.5, .5), "both_ge05")
        self.assertEqual(failure_category(.49, .5), "B_rescues_E0_low")
        self.assertEqual(failure_category(.5, .49), "E0_rescues_B_low")
        self.assertEqual(failure_category(0, 0), "both_lt05")

    def test_paired_counts_and_delta(self):
        rows = [dict(e0_dice=a,b_dice=b,e0_mae=.02,b_mae=m) for a,b,m in
                [(0,.8,.01),(.8,0,.03),(.9,.9,.02),(.1,.2,.04)]]
        r = pair_summary(rows)
        self.assertEqual((r["dice_wins_B"],r["dice_losses_B"],r["dice_ties"]),(2,1,1))
        self.assertEqual((r["B_rescues_E0_low"],r["E0_rescues_B_low"],r["both_ge05"],r["both_lt05"]),(1,1,1,1))
        self.assertEqual((r["zero_rescued_by_B"],r["zero_new_in_B"]),(1,1))
        self.assertEqual((r["B_dice_up_mae_not_worse"],r["B_dice_up_mae_worse"]),(1,1))
        self.assertAlmostEqual(r["dice_delta_B_minus_E0"],.025)

    def test_constant_errors_have_no_fake_correlation(self):
        r = pair_summary([dict(e0_dice=.8,b_dice=.9,e0_mae=.02,b_mae=.01)]*3)
        self.assertIsNone(r["error_pearson_correlation_descriptive"])

    def test_pair_membership_is_strict(self):
        a = [dict(sample_id="a",dice=.2,mae=.01)]
        b = [dict(sample_id="a",dice=.8,mae=.02)]
        r = paired_contrast(a,b,"test")
        self.assertEqual(r["low05_delta"],-1)
        self.assertAlmostEqual(r["dice_delta"],.6)
        with self.assertRaises(ValueError):
            paired_contrast(a,[dict(sample_id="b",dice=.8,mae=.02)],"bad")


if __name__ == "__main__":
    unittest.main(verbosity=2)
