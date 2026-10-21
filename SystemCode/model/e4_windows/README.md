# E4 strong augmentation control

Run from SystemCode/model. E4 changes only development augmentation from flips to strong relative to E2-C: paired flips, paired rotations within15 degrees and image-only brightness/contrast0.85-1.15. It reinitializes from the same ImageNet encoder, not a trained E2 checkpoint.

Restore the fixed1160/290 bundle, full E2-C best/history/validation records, initializer and authorized dataset. The frozen core is retained byte-for-byte and keeps E2 compatibility labels; strong configuration, output directory and e4_stage distinguish E4.

~~~bat
python e4_windows/test_e4.py
python e4_windows/core/test_e2.py
python e4_windows/run_e4.py check-gpu
python e4_windows/run_e4.py preflight
python e4_windows/run_e4.py smoke
python e4_windows/run_e4.py train
python e4_windows/run_e4.py compare
~~~

Only after the validation decision use train-final and evaluate. Historical E4 selected epoch76 and reinitialized on all1450 entries; E2 selected67, so final comparison is between complete selected recipes. Existing outputs are never overwritten. Resume requires the exact original run and configuration; workers0 diagnoses Windows loading without CPU model fallback.

The completed result, not the old preparation status, is documented in the [English baseline report](../../../ProjectReport/E0_E4_BASELINES.md). See the [team guide](../docs/TEAM_GUIDE.md) for environment and migration requirements. Later E5 uses a separate audited implementation on model_exploration.
