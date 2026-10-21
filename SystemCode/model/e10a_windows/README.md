# E10 A frozen B and E0 comparison

Updated22 October2026. Run from SystemCode/model, not this subdirectory.

## Question and controlled change

Check B/E0 FAR performance and bidirectional residual complementarity before choosing a tightly scoped fusion intervention; no fusion or training here.

## Recorded decision

All-FAR B Dice0.774268 versus E0 0.765853, but calibration favors E0 and descriptive patient interval includes zero.

See [the English experimental report](../../../ProjectReport/E10_FUSION.md) and [exact compact evidence](../../../Miscellaneous/model_results/2026-10-22/README.md). Shared results are not complete runnable/resumable runs.

## Code and safe checks

- [test_e10a.py](test_e10a.py): run python e10a_windows/test_e10a.py for synthetic functional checks, not formal performance.

Critical source and all original CMD entry points are retained here. Read the [research guide](../docs/RESEARCH_GUIDE.md) before restoring assets or launching an experiment. Formal training/inference requires matching authorized data, weights, full source/cache metadata and stage-specific prerequisites; none is fabricated by the compact result snapshot.

## Historical execution entry points

The following scripts preserve the original order/names. They may launch formal work, freeze a protocol or stop after a failed gate; they are **not** a request to rerun every experiment now:

- [01_check_and_freeze.cmd](01_check_and_freeze.cmd)
- [02_compare_frozen_models.cmd](02_compare_frozen_models.cmd)
- [03_verify.cmd](03_verify.cmd)

The original executed source is archived atc00b15fae599ae00d788393f366016e4e9c9516b. Current English report-string changes are explicitly registered; original source hashes/gates are not rewritten. Use the archived source and full matching artifacts for historical recovery, and never bypass a failed preflight or duplicate a possibly active process.

FAR/C5/C6/original798 are already viewed. Synthetic tests do not prove patient independence, negative-frame specificity, clinical safety or real-time video throughput. Any new experiment needs a separately frozen hypothesis/output/confirmation plan.
