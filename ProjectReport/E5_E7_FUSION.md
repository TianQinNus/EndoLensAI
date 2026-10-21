# E5 to E7 fusion and calibration experiments

Updated 22 October 2026. The conclusion is negative under the declared gates: E0/E4 fusion produced local or average improvements, but no tested strategy qualified for promotion. Formal downstream tests remained blocked after failed development gates. Later B-based fusion is a separate series described in [E10](E10_FUSION.md).

## E5 fixed equal fusion

Question: could stronger E4 complement E0 better than E2 did? We used `p = 0.5*pE0 + 0.5*pE4`, native-resolution float32 probabilities, threshold 0.5, no fitting, retraining or threshold search.

| Historical test scheme | Images | Dice | IoU | MAE | Dice < 0.5 | Zero Dice |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 798 | 0.836783 | 0.762996 | 0.021855 | 68 | 14 |
| E4 | 798 | 0.760910 | 0.690311 | 0.045028 | 151 | 19 |
| E3, E0+E2 | 798 | 0.832898 | 0.760585 | 0.028209 | 72 | 17 |
| E5, E0+E4 | 798 | 0.837386 | 0.765085 | 0.033442 | 68 | 15 |

E5 improved mean Dice over E0 by only 0.000604, while MAE increased by 0.011587 and zero-Dice count by one. ColonDB Dice decreased by 0.004105; ETIS increased by 0.008313. This justified investigating confidence scales, not deploying E5 as a uniformly better model. These 798 entries were already reused, not a fresh blind test.

## E6 new development resource and temperature fusion

FAR-POLYP-SEG provided 432 positive image/mask pairs in 187 patient groups. The frozen split uses 160 images / 75 groups for calibration and 272 images / 112 groups for selection, with no shared groups. Exact/heuristic overlap audits cannot establish the complete training history of public E0 weights.

Temperatures were fitted on calibration; E4 weights 0.1 to 0.6 and thresholds 0.35 to 0.65 were searched on selection. Gates required mean Dice gain at least 0.003, worst-source Dice change at least -0.003, MAE increase at most 0.003 and no increase in Dice<0.5 count relative to E0. One source makes the worst-source criterion limited in scope.

The nearest recorded candidate had selection Dice 0.770947 versus E0 0.767970, but MAE 0.069605 versus 0.037940. It failed both the minimum Dice gain and MAE criteria. Status `NO_PASS`; recording nearest parameters is not an approved configuration.

## E7-A affine post-fusion calibration

Question: could slope/intercept calibration improve probability error after fusion? Each mixture's affine transform was fitted on calibration and evaluated under unchanged selection gates. The nearest candidate's Dice was 0.771198, MAE 0.046458 and low-Dice count 40 versus E0's 42. MAE exceeded the permitted increase. Status `NO_PASS`; formal testing blocked.

## E7-A2 constrained calibration and threshold equivalence

Question: could an explicit full-image calibration MAE bound avoid the previous loss of probability quality while preserving the raw-mask search? We evaluated 91 identity candidates and 91 constrained affine candidates. The calibrated threshold was mapped from the raw probability threshold, preserving the intended mask semantics.

The closest candidate used E4 weight 0.5, raw threshold 0.5 and calibration strength 0.9. Selection Dice was 0.777276, MAE 0.041635, low-Dice count 42 and zero-Dice count 5. Dice gain was 0.009306, but MAE increased by 0.003695, exceeding the 0.003 bound by 0.000695. All 182 candidates failed. A bound on calibration did not guarantee the same bound on selection.

## E7-B0 error diagnosis and patient-grouped CV

Question: did prediction-only routing have plausible headroom, and where did error arise? Patient-grouped calibration CV avoided fitting on held-out patients. Out-of-fold calibrated fusion Dice was 0.772596 versus E0 0.762254; MAE 0.047645 versus 0.043688 still failed. A safe GT-based oracle obtained Dice 0.772374 and MAE 0.035098, but it used held-out ground truth and was never deployable.

Decision: diagnostic headroom justified one constrained prediction-only gate experiment. An oracle score is not achievable performance evidence.

## E7-B1 learned safe gate

The primary logistic policy routed zero images and exactly reproduced E0: Dice 0.762254, no required gain. Status `OOF_NO_PASS`. A fixed quantile comparator gave out-of-fold Dice 0.773332 and MAE 0.041727, but was not automatically promoted after seeing the primary fail. This motivated a separately frozen shallow-rule protocol, not a changed primary endpoint.

## E7-B2 frozen shallow rule

Exactly one frozen candidate was applied to the reused selection partition. Dice was 0.770099 versus E0 0.767970, MAE 0.037841 versus 0.037940, and low-Dice count 43 versus 42. Gain 0.002128 was below 0.003 and the tail worsened. Status `NO_PASS`; no formal 798-entry test was unlocked.

## Interpretation and next question

Repeated FAR use made later selection adaptive development. Patient grouping protects current fitting boundaries, not historical hypothesis-selection independence. GT area, true Dice and oracle labels must not be inference inputs. We stopped stacking E0/E4 rules and shifted to a controlled training question: improve the pretrained U-Net through additional training data and training-only hard sampling while freezing E0. See [E8](E8_DATA_EXPANSION.md).

Compact configurations, OOF tables and exact decisions are in [the dated evidence snapshot](../Miscellaneous/model_results/2026-10-22/). Executed E5-E7 code is retained on the `model_exploration` branch under `SystemCode/model/`.
