# Experimental evidence snapshot

Snapshot22 October2026. These58 explicitly selected original files support the English E5-E11 reports. They are not new experiments or complete resumable runs. The source import manifest records original file hashes; metric values, statuses and protocol signatures are not rewritten to turn a failure into a pass.

| Directory | Main evidence |
|---|---|
| e5_fusion_equal_seed42 | Historical798-entry E0/E4/E3/E5 summary |
| e6_selection_seed42, e7a_selection_seed42, e7a2_selection_seed42 | Exact failed selection configurations/gates |
| e7b0_diagnostics_seed42, e7b1_gate_cv_seed42 | Patient-grouped OOF tables and decisions; oracle results are diagnostic |
| e7b2_rule_selection_seed42 | Frozen one-candidate selection and NO_PASS |
| e8_*_seed* | Selected run metadata for four arms x three seeds |
| e8_repeat_summary | Source-equal and image-equal means/SD, paired contrasts and repeatability qualification |
| e8_holdout_seed42, e8_holdout_analysis_seed42 | Historical C6 results, not a new blind set |
| e9_balance_summary |70/30 revision decision and comparisons |
| e10a_far_seed42 | B/E0 partition comparison and descriptive complementarity/bootstrap |
| e10b_fusion_seed42 | Calibration-only fixed mixture results and blocked selection |
| e10c_temperature_seed42 | Patient OOF temperature results; no full-calibration/selection deployment |
| e10d_rescue_compat_seed42 | Final compatibility-protocol local rescue decision; initial interrupted attempt is not substituted |
| e11_cached_error_review | Aggregate cached-error spatial diagnosis; no patient review images |
| e11_B_background_seed42 | Incomplete saved56/100-round state/history; last update14 October, activity unverified |

Do not infer that a recorded TRAINING state is currently active. Do not select an E11 checkpoint or claim completed improvement from partial best values. B seed42 round98 remains the integration model.

Original experiment paths in metadata are historical provenance, not teammate setup instructions. Preserving a checksum does not make this compact directory sufficient for training recovery. Dataset images, real masks, probability maps, optimizer/RNG state and complete source snapshots remain separately controlled. See [reports](../../../ProjectReport/README.md) for interpretation.
