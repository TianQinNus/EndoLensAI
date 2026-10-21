# E3 V3 fixed equal probability fusion

This historical stage averages **E0 and E2**, not E4 or expanded B:

~~~text
p = 0.5*pE0 + 0.5*pE2
mask = p >= 0.5
~~~

It uses native float32 sigmoid probability caches, not PNG masks, per-image min-max maps or averaged weights. Both branches stay frozen. There is no training/new forward and no CUDA option; NumPy CPU arithmetic is expected.

Restore complete E0 probability/mask/manifest/per-image metadata; E2 final checkpoint/provenance and evaluation_cuda_20261005_231641_102143 caches; merged798 test images/masks. Shared CSV summaries alone are insufficient. Run from SystemCode/model:

~~~bat
python v3_windows/test_v3.py
python v3_windows/run_v3.py --audit-only --output runs/v3_audit_seed42
python v3_windows/run_v3.py --smoke --output runs/v3_smoke_seed42
python v3_windows/run_v3.py --output runs/v3_fusion_equal_seed42
~~~

Each output must be new, including audit/smoke directories. The smoke covers five entries and is not a full experiment. The real FULL_V3 result requires798 entries and completed status; summary, per-image metrics, probabilities, masks and provenance remain together.

Historical Dice0.832898 did not exceed E0 0.836783. Cache arithmetic time is not model/video FPS. See the [baseline report](../../../ProjectReport/E0_E4_BASELINES.md) and [team guide](../docs/TEAM_GUIDE.md).
