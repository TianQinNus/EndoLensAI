# Experiment status

Snapshot22 October2026. Current integration model: original E8 B, seed42 round98. A higher partial epoch score, a favorable example, or the best mean among failing fusion candidates does not change this selection.

| Stage | Main intervention | Recorded outcome |
|---|---|---|
| E0 | Fixed pretrained Polyp-PVT | Historical798 Dice0.836783 |
| E1 | Random conventional U-Net | Historical798 Dice0.5707 |
| E2 | Initialization controls | Pretrained ResNet34 selected; final historical798 Dice0.754879 |
| E3/V3 | Equal E0+E2 probabilities | Dice0.832898; below E0 |
| E4 | Strong augmentation | Dice0.760910; subset/FP trade-offs |
| E5 | Equal E0+E4 | Dice0.837386; worse MAE/tail than E0 |
| E6 | Temperature/weight/threshold development | NO_PASS; formal test blocked |
| E7-A/A2 | Post-fusion affine calibration | NO_PASS; MAE constraint failed |
| E7-B0 | Grouped diagnostics | GT-oracle headroom only; not a deployable result |
| E7-B1 | Prediction-only learned gate | OOF_NO_PASS; primary routes nothing |
| E7-B2 | One frozen shallow rule | NO_PASS; gain too small and low-Dice count increases |
| E8 | Expansion vs image hard sampling | Retain expanded B, not hard-sampling D |
| E8-R | Three matched seeds | Average gains; full source gates not reproduced in every seed |
| E8-H | Prior selected seed42 on C6 | B Dice0.847092 vs E0 0.830458 on83 positive images |
| E9 | Source ratio70/30 | NO_PASS_RETAIN_B; keep50/50 |
| E10-A | B/E0 FAR comparison | B all-image Dice0.774268 vs E0 0.765853; partition disagreement |
| E10-B | Three fixed mixture weights | NO_CALIBRATION_PASS_SELECTION_BLOCKED |
| E10-C | NLL temperature calibration, patient CV | NO_OOF_PASS_RETAIN_B |
| E10-D | B-preserving local/TTA rescue | NO_FEASIBILITY_PASS_RETAIN_B; fixed gate accepts nothing |
| E11 | Bounded training-background BCE | Incomplete; saved seed42 state56/100, best round54 macro Dice0.791550 |

The E11 record was last updated14 October; process activity is not inferred from its saved TRAINING label. No completed three-seed E11 result exists in this snapshot.

[Full rationale/results](../../../ProjectReport/README.md), [next steps](../../../ProjectReport/ROADMAP.md), [exact evidence](../../../Miscellaneous/model_results/2026-10-22/README.md), [retained model card](../endolens_b/docs/MODEL_CARD.md).

FAR is adaptively reused development. C5 participates in checkpoint selection; C6 and the original798 are already viewed. Positive-only metrics do not establish negative-frame specificity, patient independence, clinical diagnostic performance or Web/video FPS. All NO_PASS statuses and original gates remain unchanged.
