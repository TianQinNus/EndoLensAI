# E7 A affine post fusion calibration

Updated22 October2026. Run from SystemCode/model, not this subdirectory.

## Question and controlled change

Test slope/intercept calibration after E0/E4 fusion while retaining the E6 gates and separating fit/selection roles.

## Recorded decision

NO_PASS. Nearest selection Dice0.771198, MAE0.046458; probability-error gate failed.

See [the English experimental report](../../../ProjectReport/E5_E7_FUSION.md) and [exact compact evidence](../../../Miscellaneous/model_results/2026-10-22/README.md). Shared results are not complete runnable/resumable runs.

## Code and safe checks

- [test_e7a.py](test_e7a.py): run python e7a_windows/test_e7a.py for synthetic functional checks, not formal performance.

Critical source and all original CMD entry points are retained here. Read the [research guide](../docs/RESEARCH_GUIDE.md) before restoring assets or launching an experiment. Formal training/inference requires matching authorized data, weights, full source/cache metadata and stage-specific prerequisites; none is fabricated by the compact result snapshot.

## Historical execution entry points

The following scripts preserve the original order/names. They may launch formal work, freeze a protocol or stop after a failed gate; they are **not** a request to rerun every experiment now:

- [01_test_e7a.cmd](01_test_e7a.cmd)
- [02_select_e7a.cmd](02_select_e7a.cmd)
- [03_apply_locked_test_once.cmd](03_apply_locked_test_once.cmd)

The original executed source is archived atc00b15fae599ae00d788393f366016e4e9c9516b. Current English report-string changes are explicitly registered; original source hashes/gates are not rewritten. Use the archived source and full matching artifacts for historical recovery, and never bypass a failed preflight or duplicate a possibly active process.

FAR/C5/C6/original798 are already viewed. Synthetic tests do not prove patient independence, negative-frame specificity, clinical safety or real-time video throughput. Any new experiment needs a separately frozen hypothesis/output/confirmation plan.
