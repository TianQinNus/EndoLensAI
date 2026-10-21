# E10 D B preserving local rescue

Updated22 October2026. Run from SystemCode/model, not this subdirectory.

## Question and controlled change

Test E0-exclusive components using fixed prediction-only features and flip stability, while never deleting B regions. Preserve original interrupted and final compatibility protocols separately.

## Recorded decision

NO_FEASIBILITY_PASS_RETAIN_B. Unrestricted additions worsen FP; fixed acceptance0.8 accepts zero regions and reproduces B.

See [the English experimental report](../../../ProjectReport/E10_FUSION.md) and [exact compact evidence](../../../Miscellaneous/model_results/2026-10-22/README.md). Shared results are not complete runnable/resumable runs.

## Code and safe checks

- [test_e10d.py](test_e10d.py): run python e10d_windows/test_e10d.py for synthetic functional checks, not formal performance.

Critical source and all original CMD entry points are retained here. Read the [research guide](../docs/RESEARCH_GUIDE.md) before restoring assets or launching an experiment. Formal training/inference requires matching authorized data, weights, full source/cache metadata and stage-specific prerequisites; none is fabricated by the compact result snapshot.

## Historical execution entry points

The following scripts preserve the original order/names. They may launch formal work, freeze a protocol or stop after a failed gate; they are **not** a request to rerun every experiment now:

- [01_test_and_freeze.cmd](01_test_and_freeze.cmd)
- [02_infer_stability.cmd](02_infer_stability.cmd)
- [03_diagnose_patient_cv.cmd](03_diagnose_patient_cv.cmd)
- [04_verify.cmd](04_verify.cmd)

The original executed source is archived atc00b15fae599ae00d788393f366016e4e9c9516b. Current English report-string changes are explicitly registered; original source hashes/gates are not rewritten. Use the archived source and full matching artifacts for historical recovery, and never bypass a failed preflight or duplicate a possibly active process.

FAR/C5/C6/original798 are already viewed. Synthetic tests do not prove patient independence, negative-frame specificity, clinical safety or real-time video throughput. Any new experiment needs a separately frozen hypothesis/output/confirmation plan.
