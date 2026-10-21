# Historical E1 Windows transfer

Preserve the full e1_windows directory at SystemCode/model, the original1450 training pairs, merged/named798 test entries and matching E0 metadata. The U-Net source retains the original Mac architecture and31,031,745 parameters; this stage is random initialization, not expanded B.

The historical Windows environment used Python3.11, torch2.9.1+cu128, NumPy1.26.4 and Pillow10.2.0. Use a separate environment and the declared requirements. Check the actual CUDA device before a standard-width352 smoke; CPU synthetic checks cannot prove GPU execution.

~~~bat
python e1_windows/check_device.py --device cuda
python e1_windows/test_e1.py
python e1_windows/run_e1.py train --device cuda --smoke
python e1_windows/run_e1.py train --device cuda --e0-run runs/e0_20261004_221345_872444 --output runs/e1_unet_cuda_seed42 --train-only
~~~

The formal recipe is100 epochs, physical batch2/accumulation4, AdamW1e-4/weight decay1e-4, equal BCE and soft Dice, paired flips, seed42, FP32 and fixed final epoch. DataLoader workers on CPU do not mean model CPU fallback.

Retain the complete run/checkpoints/history/manifests for transfer. Evaluate in a separate device-specific output and keep original timing scopes. Do not resume across mismatched platforms/source/env or substitute CSV summaries for checkpoint/RNG state.

The completed E1 outcome and limitations are in the [baseline report](../../../ProjectReport/E0_E4_BASELINES.md). This English documentation change does not alter frozen Python.
