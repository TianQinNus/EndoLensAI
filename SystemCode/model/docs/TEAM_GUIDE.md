# Team installation and reproduction guide

Updated22 October2026. Use master for the frozen B SDK/reference API; use model_exploration for E5-E11 experiment source and original-source recovery guidance. Preserve the separate frontend branch and teammate model work.

## Web integration

Clone normally, then enter SystemCode/model/endolens_b and follow its [README](../endolens_b/README.md). The selected trained weight is the sole explicit exception to the repository's general weight ignore policy. CPU is sufficient for a synthetic integration smoke. Install a separate environment; do not change an active research environment.

## Historical E0-E4 setup

From SystemCode/model:

~~~bash
python scripts/bootstrap_artifacts.py
python scripts/verify_imported_code.py
python scripts/run_checks.py
~~~

Bootstrap only restores tracked E2 split metadata and lightweight E0 reference records. It downloads no images/weights, launches no training, preserves matching files and refuses different local artifacts. Shared results are not resumable runs.

Windows historical training used Python3.11, torch2.9.1+cu128, torchvision0.24.1+cu128, NumPy1.26.4 and Pillow10.2.0. Install matching official PyTorch wheels separately from common requirements:

~~~bat
conda create -n endolens-e2-win python=3.11 -y
conda activate endolens-e2-win
python -m pip install -r e2_windows/requirements-windows.txt --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r e2_windows/requirements-common.txt
python e2_windows/check_device.py --device cuda --output runs/e2_gpu_device_proof.json
python e2_windows/run_e2.py download-weights
~~~

Synthetic tests intentionally use CPU. Formal E2/E4 training requires a real CUDA forward/backward/parameter-update probe and smoke; there is no CPU fallback. CPU image loading is normal. Set workers0 for Windows loader diagnostics without changing the model device. Do not infer device proof from Task Manager alone.

## Restore authorized experiment assets

Restore byte-identical images and masks without re-encoding or renaming:

~~~text
dataset/TrainDataset/images/ and masks/       1450 original pairs
dataset/TestDataset/<five subsets>/          100/62/380/196/60 pairs
dataset/TestDataset/test/images/ and masks/  merged798 entries
e2_weights/resnet34-b627a593.pth              encoder initializer
model_pth/PolypPVT.pth                       E0 reference if rerunning inference
pretrained_pth/pvt_v2_b2.pth                  E0 backbone initializer
runs/                                       complete authorized original runs
~~~

E8 additionally requires original PolypGen data, FAR development inputs and the exact audited manifests/protocols. Data access/rights must be checked separately. Do not regenerate partitions to make a result look better.

## E2 historical commands

Restore runs/e2_data_seed42 first. New output directories must not already exist:

~~~bat
python e2_windows/run_e2.py train --experiment pretrained --device cuda --smoke --output runs/e2_smoke_C_seed42
python e2_windows/run_e2.py train --experiment original --device cuda --output runs/e2_dev_A_original_seed42
python e2_windows/run_e2.py train --experiment random --device cuda --output runs/e2_dev_B_resnet_random_seed42
python e2_windows/run_e2.py train --experiment pretrained --device cuda --output runs/e2_dev_C_resnet_pretrained_seed42
python e2_windows/run_e2.py select --runs runs/e2_dev_A_original_seed42 runs/e2_dev_B_resnet_random_seed42 runs/e2_dev_C_resnet_pretrained_seed42
~~~

Inspect the validation-only recipe before train-final/evaluate. Do not retrain simply to read existing results. Recovery requires the complete original optimizer/RNG/checkpoint state, matching source/configuration/data and environment; never point a different recipe at an existing run.

## E4 historical commands

Restore the complete E2-C control, including best checkpoint, history and validation records, then:

~~~bat
python e4_windows/run_e4.py check-gpu
python e4_windows/run_e4.py preflight
python e4_windows/run_e4.py smoke
python e4_windows/run_e4.py train
python e4_windows/run_e4.py compare
~~~

Only after the declared validation decision, use train-final and evaluate. E4's frozen core retains E2 compatibility labels; strong config and e4_stage identify the E4 recipe. Full-data training is reinitialized, not continued from development best.

## Offline fusion and later research

V3 is E0+E2 only, not E0+E4. It needs complete native float32 probability caches, matching masks/manifests, E2 final provenance and merged test images. It performs NumPy CPU arithmetic, no new inference:

~~~bat
python v3_windows/run_v3.py --audit-only --output runs/v3_audit_seed42
python v3_windows/run_v3.py --smoke --output runs/v3_smoke_seed42
python v3_windows/run_v3.py --output runs/v3_fusion_equal_seed42
~~~

Later source is on model_exploration. Read its research guide before recovering a historical run: English-only report-string edits change source bytes and must not be hidden by updating old provenance. No formal test is authorized by a failed selection result. E11's stale TRAINING label requires actual process inspection before continuation.

## Mac and migration boundaries

The original E0 Mac environment used Python3.10/torch2.2.2/torchvision0.17.2/timm0.4.12. Preserve it for historical checks; do not install CUDA wheels on Mac. CPU replay and summary analysis can be useful, but Mac timing is not Windows GPU deployment timing.

Transfer complete run directories, not just summary.csv or final weights: checkpoint/optimizer/RNG, manifests, configurations, source snapshot, history and full evaluation outputs. Keep separate device/evaluation directories. Fixed seed does not guarantee bitwise equality across backends.

## Collaboration

Use explicit paths when staging, review the diff and run the relevant tests. Do not commit data, private uploads, environments, caches or extra weights. Never force-push master or relax provenance checks. Project-facing text is English. See [CONTRIBUTING](../../../CONTRIBUTING.md) and the [asset guide](../../../Miscellaneous/model_artifacts/README.md).
