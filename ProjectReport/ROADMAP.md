# Research decisions and next steps

Updated22 October2026. Current application baseline: original E8 B, seed42 round98. No fusion model has met the complete declared promotion criteria. E11 is incomplete. These are distinct decisions, not evidence that future improvements are impossible.

## Why the research direction changed

E1 established the random U-Net baseline. E2 showed a useful initialization effect. E4 tested stronger augmentation but exposed subset/FP trade-offs. E5-E7 showed that local complementarity, better average overlap and GT-oracle headroom do not ensure deployable fusion under probability-error and failure-tail constraints.

E8 therefore improved the trainable U-Net through additional data while keeping E0 fixed. The factorial controls distinguished expansion/source policy from training-image difficulty. Three-seed results favored B in mean metrics, with an explicit source-wise repeatability qualification. E9's ratio-only revision did not improve the overall safety/performance profile.

E10 revisited fusion only after establishing B/E0 residual complementarity. Fixed averaging improved means but suppressed some surviving target responses. Temperature calibration improved NLL/Brier but not MAE or the critical failures. Non-deleting local rescue avoided deleting TP yet introduced FP or accepted nothing. The evidence now favors a concrete FP-focused training test over another unconstrained fusion search.

## Parallel work without changing the scientific baseline

The Web team can integrate the frozen B now: standardized RGB input, native-resolution masks, asynchronous recorded-video processing, reproducible task configuration and separate human reviews. Synthetic integration tests establish software behavior only. They do not validate medical accuracy or real-time throughput.

The research team can complete E11 under its existing protocol. Integration must not silently swap a later checkpoint under the same model version. Any promoted weight gets a new model identifier, provenance, export and validation record.

## Independent confirmation still needed

Before a new confirmation experiment, freeze data source, rights, patient/video units, exact and near-duplicate audits, positive/negative roles, checkpoints, sampling, metrics, gates and stopping rules. FAR, C5, C6 and the original798 entries have been viewed and cannot be repackaged as new blind confirmation.

Report means together with severe-failure counts and source strata. If video behavior matters, assess temporal coverage and false alarms on appropriate videos rather than assuming single-frame Dice/FPS transfer. Lesion-level counting requires a separately validated tracking definition; candidate frames are not unique polyps.

Do not change thresholds, remove difficult images, pick favorable seeds or reinterpret a failed gate after seeing confirmation results. Negative experimental results are part of the project contribution and remain documented.
