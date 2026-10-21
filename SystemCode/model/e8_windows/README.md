# E8 data expansion and training controls

Updated22 October2026. Run from SystemCode/model, not this subdirectory.

## Question and controlled change

Audit original/PolypGen roles and compare A original, B expanded, C hard sampling and D expanded+hard under matched image-draw budgets and the same pretrained encoder.

## Recorded decision

Retain original B seed42 round98. C6 historical Dice0.847092 versus E0 0.830458; positive one-center evidence only.

See [the English experimental report](../../../ProjectReport/E8_DATA_EXPANSION.md) and [exact compact evidence](../../../Miscellaneous/model_results/2026-10-22/README.md). Shared results are not complete runnable/resumable runs.

## Code and safe checks

- [test_e8.py](test_e8.py): run python e8_windows/test_e8.py for synthetic functional checks, not formal performance.
- [test_holdout_analysis.py](test_holdout_analysis.py): run python e8_windows/test_holdout_analysis.py for synthetic functional checks, not formal performance.

Critical source and all original CMD entry points are retained here. Read the [research guide](../docs/RESEARCH_GUIDE.md) before restoring assets or launching an experiment. Formal training/inference requires matching authorized data, weights, full source/cache metadata and stage-specific prerequisites; none is fabricated by the compact result snapshot.

## Historical execution entry points

The following scripts preserve the original order/names. They may launch formal work, freeze a protocol or stop after a failed gate; they are **not** a request to rerun every experiment now:

- [00_download_and_prepare.cmd](00_download_and_prepare.cmd)
- [01_check_and_smoke.cmd](01_check_and_smoke.cmd)
- [02_train_baseline.cmd](02_train_baseline.cmd)
- [03_train_expanded.cmd](03_train_expanded.cmd)
- [04_train_hard_controls.cmd](04_train_hard_controls.cmd)
- [05_compare_validation.cmd](05_compare_validation.cmd)
- [06_lock_holdout.cmd](06_lock_holdout.cmd)
- [07_evaluate_locked_holdout.cmd](07_evaluate_locked_holdout.cmd)
- [08_analyze_holdout.cmd](08_analyze_holdout.cmd)

The original executed source is archived atc00b15fae599ae00d788393f366016e4e9c9516b. Current English report-string changes are explicitly registered; original source hashes/gates are not rewritten. Use the archived source and full matching artifacts for historical recovery, and never bypass a failed preflight or duplicate a possibly active process.

FAR/C5/C6/original798 are already viewed. Synthetic tests do not prove patient independence, negative-frame specificity, clinical safety or real-time video throughput. Any new experiment needs a separately frozen hypothesis/output/confirmation plan.
