# E10 C constrained temperature fusion

Updated22 October2026. Run from SystemCode/model, not this subdirectory.

## Question and controlled change

Fit independent temperatures by NLL under bounded patient-grouped OOF fitting, retaining equal mixture and fixed threshold; do not optimize temperatures by Dice.

## Recorded decision

NO_OOF_PASS_RETAIN_B. OOF Dice0.787784 but MAE0.056249 and critical zero-Dice failures remain.

See [the English experimental report](../../../ProjectReport/E10_FUSION.md) and [exact compact evidence](../../../Miscellaneous/model_results/2026-10-22/README.md). Shared results are not complete runnable/resumable runs.

## Code and safe checks

- [test_e10c.py](test_e10c.py): run python e10c_windows/test_e10c.py for synthetic functional checks, not formal performance.

Critical source and all original CMD entry points are retained here. Read the [research guide](../docs/RESEARCH_GUIDE.md) before restoring assets or launching an experiment. Formal training/inference requires matching authorized data, weights, full source/cache metadata and stage-specific prerequisites; none is fabricated by the compact result snapshot.

## Historical execution entry points

The following scripts preserve the original order/names. They may launch formal work, freeze a protocol or stop after a failed gate; they are **not** a request to rerun every experiment now:

- [01_test_and_freeze.cmd](01_test_and_freeze.cmd)
- [02_patient_cv.cmd](02_patient_cv.cmd)
- [03_refit_if_pass.cmd](03_refit_if_pass.cmd)
- [04_locked_selection_if_pass.cmd](04_locked_selection_if_pass.cmd)
- [05_verify.cmd](05_verify.cmd)

The original executed source is archived atc00b15fae599ae00d788393f366016e4e9c9516b. Current English report-string changes are explicitly registered; original source hashes/gates are not rewritten. Use the archived source and full matching artifacts for historical recovery, and never bypass a failed preflight or duplicate a possibly active process.

FAR/C5/C6/original798 are already viewed. Synthetic tests do not prove patient independence, negative-frame specificity, clinical safety or real-time video throughput. Any new experiment needs a separately frozen hypothesis/output/confirmation plan.
