# E5 fixed E0 and E4 fusion

Updated22 October2026. Run from SystemCode/model, not this subdirectory.

## Question and controlled change

Test whether strong-augmentation E4 adds useful complementary probabilities to E0, without fitting a new model. Fixed equal weights and threshold0.5 isolate averaging.

## Recorded decision

Historical798 Dice0.837386 versus E0 0.836783, but MAE0.033442 versus0.021855 and zero Dice15 versus14. No broad promotion claim.

See [the English experimental report](../../../ProjectReport/E5_E7_FUSION.md) and [exact compact evidence](../../../Miscellaneous/model_results/2026-10-22/README.md). Shared results are not complete runnable/resumable runs.

## Code and safe checks

- [test_e5.py](test_e5.py): run python e5_windows/test_e5.py for synthetic functional checks, not formal performance.

Critical source and all original CMD entry points are retained here. Read the [research guide](../docs/RESEARCH_GUIDE.md) before restoring assets or launching an experiment. Formal training/inference requires matching authorized data, weights, full source/cache metadata and stage-specific prerequisites; none is fabricated by the compact result snapshot.

## Historical execution entry points

The following scripts preserve the original order/names. They may launch formal work, freeze a protocol or stop after a failed gate; they are **not** a request to rerun every experiment now:

- [01_check_and_smoke.cmd](01_check_and_smoke.cmd)
- [02_run_full.cmd](02_run_full.cmd)

The original executed source is archived atc00b15fae599ae00d788393f366016e4e9c9516b. Current English report-string changes are explicitly registered; original source hashes/gates are not rewritten. Use the archived source and full matching artifacts for historical recovery, and never bypass a failed preflight or duplicate a possibly active process.

FAR/C5/C6/original798 are already viewed. Synthetic tests do not prove patient independence, negative-frame specificity, clinical safety or real-time video throughput. Any new experiment needs a separately frozen hypothesis/output/confirmation plan.
