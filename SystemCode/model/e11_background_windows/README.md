# E11 bounded background pixel training

Updated22 October2026. Run from SystemCode/model, not this subdirectory.

## Question and controlled change

Add only stopped-gradient probability-squared weighting to background BCE, bounded total weight1-2, under the original B data/exposure/augmentation/selection protocol.

## Recorded decision

INCOMPLETE. Saved seed42 state56/100, best round54 macro Dice0.791550; last update14 October, current process activity unverified. No promoted revision.

See [the English experimental report](../../../ProjectReport/E11_BACKGROUND.md) and [exact compact evidence](../../../Miscellaneous/model_results/2026-10-22/README.md). Shared results are not complete runnable/resumable runs.

## Code and safe checks

- [test_e11.py](test_e11.py): run python e11_background_windows/test_e11.py for synthetic functional checks, not formal performance.

Critical source and all original CMD entry points are retained here. Read the [research guide](../docs/RESEARCH_GUIDE.md) before restoring assets or launching an experiment. Formal training/inference requires matching authorized data, weights, full source/cache metadata and stage-specific prerequisites; none is fabricated by the compact result snapshot.

## Historical execution entry points

The following scripts preserve the original order/names. They may launch formal work, freeze a protocol or stop after a failed gate; they are **not** a request to rerun every experiment now:

- [00_review_cached_errors.cmd](00_review_cached_errors.cmd)
- [01_test_and_prepare.cmd](01_test_and_prepare.cmd)
- [02_smoke.cmd](02_smoke.cmd)
- [03_train_all.cmd](03_train_all.cmd)
- [04_summarize.cmd](04_summarize.cmd)
- [05_status.cmd](05_status.cmd)
- [06_verify_summary.cmd](06_verify_summary.cmd)

The original executed source is archived atc00b15fae599ae00d788393f366016e4e9c9516b. Current English report-string changes are explicitly registered; original source hashes/gates are not rewritten. Use the archived source and full matching artifacts for historical recovery, and never bypass a failed preflight or duplicate a possibly active process.

FAR/C5/C6/original798 are already viewed. Synthetic tests do not prove patient independence, negative-frame specificity, clinical safety or real-time video throughput. Any new experiment needs a separately frozen hypothesis/output/confirmation plan.
