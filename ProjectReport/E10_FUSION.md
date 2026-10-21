# E10 retained B comparison and constrained fusion

Updated 22 October 2026. B remains the integration baseline. E10-B/C/D did not qualify a fusion model for deployment, despite useful average Dice improvements in some controls.

## E10-A comparison before choosing a fusion intervention

Question: after data expansion, does B still need E0, and is there observable error complementarity? B seed42 round98 was run on all432 FAR positive images; E0's verified original probability caches were reused. No training, fusion or C6 reevaluation occurred.

| FAR partition | Model | n | Dice | MAE | Dice<0.5 | Dice<0.1 | Zero Dice |
|---|---|---:|---:|---:|---:|---:|---:|
| All | E0 | 432 | 0.765853 | 0.040069 | 68 | 22 | 10 |
| All | B | 432 | 0.774268 | 0.037177 | 62 | 12 | 4 |
| Calibration | E0 | 160 | 0.762254 | 0.043688 | 26 | 8 | 4 |
| Calibration | B | 160 | 0.753331 | 0.041411 | 30 | 6 | 2 |
| Selection | E0 | 272 | 0.767970 | 0.037940 | 42 | 14 | 6 |
| Selection | B | 272 | 0.786584 | 0.034686 | 32 | 6 | 2 |

B was better in the all-image mean but not calibration Dice. Patient-group bootstrap for the all-image B-E0 Dice difference gave a descriptive95% interval approximately[-0.011524,+0.027752], including zero. We do not claim significant or universal superiority. B rescued29 E0 low-Dice images; E0 rescued23 B low-Dice images;39 were low in both. GT-derived rescue identities are diagnostic, never deployment inputs.

FAR had already been used in E6/E7; E10-A also viewed both partitions. All subsequent FAR experiments are adaptive development, not restored blind testing.

## E10-B fixed probability averaging

To isolate the intervention, only E0 weights0.25/0.50/0.75 were allowed in `p=(1-alpha)*pB+alpha*pE0`; threshold0.5, no calibrator, routing or new inference. Calibration gates required Dice gain>=0.003 against both single models, MAE increase<=0.003 and no increase in low, near-zero or zero Dice. No passing candidate meant no selection fusion computation.

| Calibration scheme | Dice | MAE | Dice<0.5 | Dice<0.1 | Zero Dice |
|---|---:|---:|---:|---:|---:|
| B | 0.753331 | 0.041411 | 30 | 6 | 2 |
| E0 | 0.762254 | 0.043688 | 26 | 8 | 4 |
| E0 weight0.25 | 0.756499 | 0.041980 | 30 | 6 | 2 |
| E0 weight0.50 | 0.779279 | 0.042549 | 24 | 6 | 3 |
| E0 weight0.75 | 0.763951 | 0.043119 | 26 | 8 | 4 |

Equal fusion improved mean Dice substantially, but added two zero-Dice cases and rescued one relative to B, net+1. Very low E0 probabilities suppressed B's surviving target response below0.5. Highest mean did not override the failure gate. Status `NO_CALIBRATION_PASS_SELECTION_BLOCKED`; no selection fusion result exists.

## E10-C constrained temperature calibration

Question: could softening erroneous confidence retain averaging benefits without those catastrophic failures? Independent positive temperatures for B/E0 were fitted by NLL on training patients within five-fold calibration CV, bounded[0.25,4], using deterministic8,192 pixels per image and image-equal weighting. Temperatures were not searched by Dice; weights and threshold remained0.5.

| OOF scheme on160 images | Dice | MAE | Dice<0.5 | Dice<0.1 | Zero Dice |
|---|---:|---:|---:|---:|---:|
| B | 0.753331 | 0.041411 | 30 | 6 | 2 |
| Raw equal fusion | 0.779279 | 0.042549 | 24 | 6 | 3 |
| Temperature equal fusion | 0.787784 | 0.056249 | 24 | 6 | 3 |

NLL/Brier improved to0.104278/0.028248 from raw equal0.136274/0.030512. However MAE rose markedly, the two newly zero-overlap cases remained, and one fold worsened its near-zero count. E0 temperature hit the declared upper bound in all folds; we recorded this, not expanded the range. Status `NO_OOF_PASS_RETAIN_B`; no full-calibration fit, deployed temperature or selection result was produced.

## E10-D B-preserving local rescue

Question: could adding only useful E0-exclusive components avoid deleting B's remaining true positives? Candidates were8-connected components of `(pE0>=0.5)&(pB<0.5)`. A fixed ridge logistic rule used five prediction-only features, including restored horizontal/vertical flip stability. Five-fold patient OOF fitting, ridge0.01 and acceptance0.8 were fixed; a no-TTA rule was a declared ablation, not a fallback selected after failure.

The label for training-fold components required positive Dice gain without MAE increase when added alone. GT was not provided to inference scoring. Within accepted regions the output used `max(pB,pE0)`; elsewhere B stayed unchanged. This preserves existing TP but cannot remove B's false positives.

| Calibration scheme | Dice | MAE | Dice<0.5 | Dice<0.1 | Zero Dice |
|---|---:|---:|---:|---:|---:|
| B | 0.753331 | 0.041411 | 30 | 6 | 2 |
| Add all candidate regions | 0.730337 | 0.054299 | 35 | 7 | 1 |
| Fixed no-TTA gate | 0.753331 | 0.041411 | 30 | 6 | 2 |
| Fixed TTA OOF gate | 0.753331 | 0.041411 | 30 | 6 | 2 |

Of2,115 components,532 had the retrospective helpful label. Unrestricted addition introduced857,673 FP and169,759 TP pixels; approximately83.5% of added pixels were FP. The TTA rule's descriptive weighted AUC0.654143 exceeded the no-TTA0.545106, but its maximum score0.750989 was below0.8: no component/patient was accepted. Unchanged B is not a successful safe fusion. Status `NO_FEASIBILITY_PASS_RETAIN_B`; no full rule fit or selection evaluation followed.

An initial inference attempt was preserved after an E0 numerical compatibility failure. Before downstream label/gate analysis, a separate compatibility protocol restored E0's historical cuDNN TF32 setting without relaxing tolerances; B retained TF32 disabled. The final run exactly replayed both models' original probabilities on all160 images each and generated640 committed flip forwards. Earlier partial attempts are not silently counted as final data.

## Decision

Stop these frozen fusion searches rather than lowering acceptance, relaxing zero-Dice protection, expanding temperature ranges or choosing extra weights on FAR. Retain B and E0 as separate references. A new independent dataset is needed for confirmation, including explicit negative-frame roles if false-alarm claims are desired.

Further B training is justified only by a concrete residual error hypothesis. E10-D showed many B low-Dice cases have substantial FP, motivating the bounded training-background intervention in [E11](E11_BACKGROUND.md). This does not imply that all fusion methods or TTA are ineffective.

Exact summary tables, decisions and fold diagnostics: [dated evidence](../Miscellaneous/model_results/2026-10-22/). Key implementation on `model_exploration`: `e10a_windows/`, `e10b_windows/`, `e10c_windows/`, `e10d_windows/`.
