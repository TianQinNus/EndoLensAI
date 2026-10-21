# E9 source balance revision

Updated 22 October 2026. Final status: `NO_PASS_RETAIN_B`.

## Why change only the sampling ratio

E8 expansion improved average performance but did not reproduce all original source-wise gates across seeds. E9 asked whether giving the original training source more exposure could improve that balance without changing the architecture or collecting a different validation split.

The sole revision was original/additional sampling mass from50/50 to70/30. Dataset membership, 100 rounds x1,160 draws, augmentation, loss, initialization, optimizer and validation checkpoint selection remained unchanged. Three matched seeds were required; a favorable seed could not replace the declared policy.

## Results on the reused development validation

| Recipe | Macro Dice mean +/- sample SD | Image-mean Dice | MAE | Mean Dice<0.5 | Mean zero Dice |
|---|---|---:|---:|---:|---:|
| A original | 0.750980 +/- 0.005561 | 0.779518 | 0.032534 | 84.33 | 26.00 |
| B expanded50/50 | 0.805842 +/- 0.001631 | 0.825501 | 0.025100 | 55.67 | 12.33 |
| B balance70/30 | 0.800373 +/- 0.003597 | 0.820485 | 0.024867 | 56.33 | 17.67 |

The revision slightly lowered MAE but reduced macro Dice by about0.005469 and increased average zero-Dice count by about5.33 relative to B. It did not satisfy all-seed protection requirements. We retained B50/50, did not retune the ratio, and did not rerun C6 to search for a favorable outcome.

## Operational lesson

A recorded `TRAINING` state after interruption is not sufficient evidence that it is safe to resume. The continuation guard deliberately refuses a second potentially active process. Inspect the actual terminal/process and checkpoints before recovery; do not delete locks or bypass provenance checks. Recovery tooling and experiment comparisons must remain distinct.

Exact gates and paired contrasts are preserved in [dated evidence](../Miscellaneous/model_results/2026-10-22/e9_balance_summary/). The next research question was whether the retained B's residual errors were complementary to E0, not whether another ratio could improve an already reused mean.
