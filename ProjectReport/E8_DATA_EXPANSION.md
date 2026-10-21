# E8 data expansion and hard sample controls

Updated 22 October 2026. Expanded-data arm B is the retained pretrained U-Net baseline and the Web integration model. Data expansion yielded consistent average gains over the matched original-data arm, but the original source-wise gates did not reproduce in every seed. We retain that qualification rather than relabeling the experiment a universal success.

## Hypothesis and data audit

After E5-E7 fusion failures, we asked whether improving the weaker trainable branch was more useful than adding fusion rules. E0 remained frozen. Additional PolypGen data could broaden training appearance/target size; training-only hard sampling could emphasize poor fits without leaking validation or FAR labels.

| Role | Images | Composition |
|---|---:|---|
| Original training | 1,151 | Original 1,160 development entries minus 9 near-duplicate candidates |
| Additional training | 1,118 | PolypGen C1 251, C2 267, C3 456, C4 144 |
| Validation | 494 | Original 290 plus PolypGen C5 204 |
| Historical C6 holdout | 83 | Positive single frames from C6 |

Empty-mask frames and quarantined near duplicates were excluded according to the frozen protocol, without deleting or relabeling original data. The image-only heuristic used dHash distance <=2, 16x16 RGB MAE/255 <=0.01 and aspect-ratio difference <=0.01; it cannot rule out all related frames. Reliable patient identity is unavailable for part of the original data.

PolypGen source: [author repository](https://github.com/DebeshJha/PolypGen), [data identifier](https://www.synapse.org/Synapse:syn45200214), [paper](https://doi.org/10.1038/s41597-023-01981-y). The downloaded archive's license says CC BY-NC-SA 4.0 while README/paper say CC BY 4.0. This discrepancy remains unresolved; research use is not a blanket redistribution/commercial license.

## Matched controls and reasons

| Arm | Training data | Image hard sampling | Intended contrast |
|---|---|---|---|
| A | Original | No | Matched training baseline |
| B | Original + additional | No | Expansion with fixed source balance |
| C | Original | Yes | Training-image difficulty alone |
| D | Original + additional | Yes | Difficulty added to expansion |

All arms use the same ImageNet-pretrained ResNet34-U-Net, 352 input, strong augmentation, equal BCE + per-image soft Dice, AdamW 1e-4, weight decay 1e-4, batch2/accumulation4, AMP training and FP32 validation. Each round draws 1,160 images with replacement; all run 100 rounds. A round is not a full traversal of the expanded dataset. AMP-skipped updates are recorded, not hidden by a claim of identical realized update counts.

In B/D, original data receive 50% total sampling probability, additional data 50%, split equally among C1-C4. This controls source exposure instead of letting the largest center dominate. It is not a claim that the full dataset is discarded: every eligible image has nonzero sampling probability over training.

C/D use only training-loss EMA: after round20, the observed top quarter within each source receives relative weight2, keeping source mass fixed and other images eligible. No validation/FAR result enters the difficulty score. Best checkpoint selection maximizes `0.5*original_validation_Dice + 0.5*C5_Dice`, earliest tie.

## Seed42 development

| Arm | Best round | Validation macro Dice | Image-mean Dice | MAE | Dice<0.5 | Zero Dice |
|---|---:|---:|---:|---:|---:|---:|
| A | 37 | 0.744955 | 0.772845 | 0.036626 | 88 | 21 |
| B | 98 | 0.804322 | 0.824788 | 0.025114 | 58 | 11 |
| C | 74 | 0.756522 | 0.785149 | 0.030887 | 78 | 19 |
| D | 84 | 0.802293 | 0.823122 | 0.025765 | 55 | 21 |

B was selected by the declared development criteria, not by C6. D's lower low-Dice count alone did not override its zero-Dice behavior or the declared selection rule.

## E8-R three-seed repetition

| Arm | Validation macro Dice mean +/- sample SD | Image-mean Dice | MAE | Mean low-Dice count | Mean zero-Dice count |
|---|---|---:|---:|---:|---:|
| A | 0.750980 +/- 0.005561 | 0.779518 | 0.032534 | 84.33 | 26.00 |
| B | 0.805842 +/- 0.001631 | 0.825501 | 0.025100 | 55.67 | 12.33 |
| C | 0.750298 +/- 0.005455 | 0.779586 | 0.031848 | 79.33 | 25.00 |
| D | 0.803064 +/- 0.003608 | 0.823217 | 0.025040 | 56.67 | 17.00 |

Paired B-A macro Dice gains were +0.059368, +0.049723 and +0.055495 for seeds42/17/2024. Original gates passed in42/17 but not2024: `B_GATES_NOT_REPRODUCED_ALL_SEEDS`. Seed variability is not a patient confidence interval or proof of new-center generalization. We did not cherry-pick a replacement seed or ensemble the seeds.

## E8-H frozen C6 evaluation

The selected seed42 checkpoints and E0 were evaluated once under the frozen protocol. B remained the prior validation-selected model; C6 did not reselect it.

| Model | Dice | IoU | MAE | Dice<0.5 | Dice<0.1 | Zero Dice |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.770830 | 0.705678 | 0.038116 | 15 | 8 | 5 |
| B | 0.847092 | 0.781509 | 0.015525 | 7 | 2 | 2 |
| C | 0.791152 | 0.729065 | 0.023915 | 14 | 7 | 4 |
| D | 0.819525 | 0.762119 | 0.016479 | 11 | 5 | 4 |
| E0 | 0.830458 | 0.771551 | 0.019135 | 10 | 5 | 4 |

B exceeded E0 on these 83 positive images, Dice +0.016634, with lower MAE and severe-failure counts. This is limited one-center, one-seed evidence, not general architectural superiority. C6 has now been viewed and is not an unseen fusion test set.

## Decision and implementation

Retain original B seed42 round98, not D, a seed ensemble or a later incomplete checkpoint. Image hard sampling did not deliver a consistent advantage over plain expanded training. See [E9](E9_BALANCE.md) for the controlled source-ratio revision and [E10](E10_FUSION.md) for B/E0 comparison and fusion failures.

Critical implementation on `model_exploration`: `e8_windows/train_e8.py` contains sampling probabilities and per-image loss; `prepare_data.py` contains the audit and center roles; `run_e8.py` contains selection/provenance; `e8_repeat_windows/run_e8_repeat.py` aggregates matched seeds. Curated exact tables and decisions are in [the evidence snapshot](../Miscellaneous/model_results/2026-10-22/).
