# E7 B2 frozen shallow rule

Updated22 October2026. Run from SystemCode/model, not this subdirectory.

## Question and controlled change

Freeze exactly one shallow rule before applying it once to the adaptively reused selection partition.

## Recorded decision

NO_PASS. Dice gain0.002128 below0.003 and low-Dice count43 versus42; formal test blocked.

See [the English experimental report](../../../ProjectReport/E5_E7_FUSION.md) and [exact compact evidence](../../../Miscellaneous/model_results/2026-10-22/README.md). Shared results are not complete runnable/resumable runs.

## Code and safe checks

- [test_e7b2.py](test_e7b2.py): run python e7b2_windows/test_e7b2.py for synthetic functional checks, not formal performance.

Critical source and all original CMD entry points are retained here. Read the [research guide](../docs/RESEARCH_GUIDE.md) before restoring assets or launching an experiment. Formal training/inference requires matching authorized data, weights, full source/cache metadata and stage-specific prerequisites; none is fabricated by the compact result snapshot.

## Historical execution entry points

The following scripts preserve the original order/names. They may launch formal work, freeze a protocol or stop after a failed gate; they are **not** a request to rerun every experiment now:

- [01_test_e7b2.cmd](01_test_e7b2.cmd)
- [02_freeze_rule.cmd](02_freeze_rule.cmd)
- [03_evaluate_selection_once.cmd](03_evaluate_selection_once.cmd)
- [04_verify_outputs.cmd](04_verify_outputs.cmd)

The original executed source is archived atc00b15fae599ae00d788393f366016e4e9c9516b. Current English report-string changes are explicitly registered; original source hashes/gates are not rewritten. Use the archived source and full matching artifacts for historical recovery, and never bypass a failed preflight or duplicate a possibly active process.

FAR/C5/C6/original798 are already viewed. Synthetic tests do not prove patient independence, negative-frame specificity, clinical safety or real-time video throughput. Any new experiment needs a separately frozen hypothesis/output/confirmation plan.
