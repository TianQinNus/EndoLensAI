# E6 calibrated development fusion

Updated22 October2026. Run from SystemCode/model, not this subdirectory.

## Question and controlled change

Audit FAR development data, freeze patient-grouped calibration/selection roles, fit temperatures only on calibration, and select weights/thresholds under declared protection gates.

## Recorded decision

NO_PASS. Nearest selection candidate Dice0.770947, MAE0.069605; formal test blocked.

See [the English experimental report](../../../ProjectReport/E5_E7_FUSION.md) and [exact compact evidence](../../../Miscellaneous/model_results/2026-10-22/README.md). Shared results are not complete runnable/resumable runs.

## Code and safe checks

- [test_e6.py](test_e6.py): run python e6_windows/test_e6.py for synthetic functional checks, not formal performance.

Critical source and all original CMD entry points are retained here. Read the [research guide](../docs/RESEARCH_GUIDE.md) before restoring assets or launching an experiment. Formal training/inference requires matching authorized data, weights, full source/cache metadata and stage-specific prerequisites; none is fabricated by the compact result snapshot.

## Historical execution entry points

The following scripts preserve the original order/names. They may launch formal work, freeze a protocol or stop after a failed gate; they are **not** a request to rerun every experiment now:

- [01_test_and_prepare.cmd](01_test_and_prepare.cmd)
- [02_smoke_inference.cmd](02_smoke_inference.cmd)
- [03_infer_development.cmd](03_infer_development.cmd)
- [04_calibrate_and_select.cmd](04_calibrate_and_select.cmd)
- [05_apply_locked_test_once.cmd](05_apply_locked_test_once.cmd)

The original executed source is archived atc00b15fae599ae00d788393f366016e4e9c9516b. Current English report-string changes are explicitly registered; original source hashes/gates are not rewritten. Use the archived source and full matching artifacts for historical recovery, and never bypass a failed preflight or duplicate a possibly active process.

FAR/C5/C6/original798 are already viewed. Synthetic tests do not prove patient independence, negative-frame specificity, clinical safety or real-time video throughput. Any new experiment needs a separately frozen hypothesis/output/confirmation plan.
