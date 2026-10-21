# E8 R matched seed repetition

Updated22 October2026. Run from SystemCode/model, not this subdirectory.

## Question and controlled change

Repeat all four arms with declared seeds42/17/2024 on the same fixed split; report paired/source/tail behavior without selecting a favorable seed.

## Recorded decision

B mean macro Dice0.805842+/-0.001631; B_GATES_NOT_REPRODUCED_ALL_SEEDS because seed2024 failed original source gates.

See [the English experimental report](../../../ProjectReport/E8_DATA_EXPANSION.md) and [exact compact evidence](../../../Miscellaneous/model_results/2026-10-22/README.md). Shared results are not complete runnable/resumable runs.

## Code and safe checks

- [test_e8_repeat.py](test_e8_repeat.py): run python e8_repeat_windows/test_e8_repeat.py for synthetic functional checks, not formal performance.

Critical source and all original CMD entry points are retained here. Read the [research guide](../docs/RESEARCH_GUIDE.md) before restoring assets or launching an experiment. Formal training/inference requires matching authorized data, weights, full source/cache metadata and stage-specific prerequisites; none is fabricated by the compact result snapshot.

## Historical execution entry points

The following scripts preserve the original order/names. They may launch formal work, freeze a protocol or stop after a failed gate; they are **not** a request to rerun every experiment now:

- [01_check_and_prepare.cmd](01_check_and_prepare.cmd)
- [02_train_seed17.cmd](02_train_seed17.cmd)
- [03_train_seed2024.cmd](03_train_seed2024.cmd)
- [04_summarize_seeds.cmd](04_summarize_seeds.cmd)
- [05_status.cmd](05_status.cmd)

The original executed source is archived atc00b15fae599ae00d788393f366016e4e9c9516b. Current English report-string changes are explicitly registered; original source hashes/gates are not rewritten. Use the archived source and full matching artifacts for historical recovery, and never bypass a failed preflight or duplicate a possibly active process.

FAR/C5/C6/original798 are already viewed. Synthetic tests do not prove patient independence, negative-frame specificity, clinical safety or real-time video throughput. Any new experiment needs a separately frozen hypothesis/output/confirmation plan.
