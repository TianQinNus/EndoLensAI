# E11 bounded background pixel training

Snapshot date22 October2026. **Incomplete and not promoted.** The latest local saved seed42 metadata says `TRAINING`,56/100 completed rounds, last update14 October2026. This saved status does not establish a currently active process. Seeds17/2024 and the final matched-seed decision are not complete in the inspected snapshot. The Web model remains the original E8 B.

## Error evidence and rationale

In the30 FAR calibration images where B Dice<0.5,25 overlapping cases had FP pixel count at least FN. A352-scale descriptive check found most FP beyond the GT's outer5-pixel Chebyshev band in29/30 images; pooled distant-FP fraction was0.960683. These findings point beyond slight boundary thickness alone, but do not identify tissue type, diagnose annotation errors or prove the underlying medical cause.

The review used existing cached B predictions, without new inference or exporting FAR regions into training. FAR images, masks and error labels remain out of E11 training. We did not remove difficult examples or rewrite GT after seeing predictions.

## Single controlled training change

The original B uses FP32 BCE plus per-image soft Dice. E11 adds:

```text
mean_all_pixels((1-y) * stop_gradient(sigmoid(logits))**2 * BCE(logits,y))
```

Background BCE weight becomes `1+stop_gradient(p)**2`, bounded[1,2]; foreground BCE weight stays1 and Dice is unchanged. Strength1 and exponent2 were frozen. This softly emphasizes currently confident background mistakes using the training image's existing GT only; it is not image-level hard sampling, a FAR-trained gate or an originality claim for focal-style weighting.

Data membership2,269/494,50/50 source mass, strong augmentation, initialization, optimizer,100x1,160 draws and checkpoint rule are unchanged. Each run restarts from the same ImageNet encoder initialization and a random decoder, not B best with extra budget. Matched seeds42/17/2024 are required.

## Declared development gates

Every seed must meet all gates against its matched B: macro Dice gain>=0.003; macro FP/image-area reduction>=5%, with no source increase; source Dice loss<=0.003 and MAE increase<=0.003; source recall and small-target recall loss<=0.01; no source increase in low, near-zero or zero Dice. FP/image-area on positive images is not negative-frame specificity.

If any seed/gate fails, retain original B without tuning strength, exponent, threshold, seed or checkpoint on the same development result. Passing would only nominate a developmental candidate for a separately frozen independent confirmation.

## Recorded progress and engineering checks

The preparation record reports13 synthetic checks and a real CUDA one-round smoke, not a performance gain. The latest formal seed42 snapshot has best round54, validation macro Dice0.791550 and56 completed rounds. This partial best cannot be compared as a completed100-round result or a three-seed conclusion. Original B seed42 validation macro Dice is0.804322.

Training history and the exact saved metadata are in [evidence](../Miscellaneous/model_results/2026-10-22/e11_B_background_seed42/). Source on `model_exploration`: `e11_background_windows/background_loss.py`, `run_e11.py`, `test_e11.py`, `diagnose_errors.py`. Do not resume an apparently active/stale run without checking the actual process and retaining its matching original source, data and recovery state.

## Next steps

1. Inspect the original training terminal/process and establish safe continuation; do not launch a duplicate or delete locks.
2. Complete the declared matched seeds under the original frozen source, without changing gates midway.
3. Summarize all source/tail/recall metrics and apply the declared decision once.
4. Keep B as the application baseline until a completed revision qualifies; use new audited data for independent confirmation.

The English GitHub documentation does not modify the original local source or runs. Historical source byte identity matters for recovery; see the research reproduction guide on `model_exploration`.
