# E0 fixed reference on Mac

Historical evaluation only: pretrained Polyp-PVT, no training/fine-tuning. The project root is SystemCode/model, containing lib/, dataset/, model_pth/PolypPVT.pth and pretrained_pth/pvt_v2_b2.pth. An alternate root can be supplied with --project.

The original validated CPU environment was Python3.10.21, torch2.2.2, torchvision0.17.2, timm0.4.12, NumPy1.26.4, Pillow10.2.0. Preserve it for historical replay rather than upgrading an active environment.

~~~bash
python -m pip install -r e0_mac/requirements-e0-mac.txt
python e0_mac/test_e0.py
python e0_mac/run_e0.py --audit-only
python e0_mac/run_e0.py --device cpu --smoke
python e0_mac/run_e0.py --device cpu
~~~

Output directories must be new. Smoke uses one image per subset, not the full798. Formal completion requires FULL_E0, COMPLETED and798 processed entries. Failed/interrupted partial outputs are retained; this version does not resume an incomplete evaluation.

The audit checks100/62/380/196/60 named-subset pairs and their unambiguous mapping to merged test entries, decoded RGB and binary masks. It does not establish patient independence. The798 entries contain701 distinct RGB images and76 duplicate groups;58 groups cross CVC-300/ColonDB. Entry weighting is retained.

Preprocess RGB using Pillow bilinear352, /255 and ImageNet normalization; sum both logits, native-size bilinear restore, sigmoid, threshold0.5. Save float32 NPZ key probability and binary0/255 PNG. Legacy per-image min-max outputs are separate and must never be used as calibrated probabilities for fusion.

CPU is the historical reference. Mac auto deliberately uses CPU; MPS was an optional, less-validated path, and CPU fallback timing is not pure GPU timing. Forward time excludes decoding/preprocessing/postprocessing/metrics/storage; loop time has a different scope. No video FPS follows.

Use the [current team guide](../docs/TEAM_GUIDE.md), [metric protocol](../docs/MODEL_PROTOCOL.md) and [English baseline report](../../../ProjectReport/E0_E4_BASELINES.md). Frozen Python bytes are unchanged by this documentation migration.
