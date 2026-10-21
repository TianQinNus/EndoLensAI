# Independent U-Net from random initialization

This teammate research line is retained separately from the ImageNet-pretrained E8 B. It implements a four-level U-Net without importing E0-E4 model code or loading existing/pretrained weights. Reused split metadata do not imply reused model parameters.

Original run: runs/unet_scratch_20261016; log: runs/unet_scratch_20261016.log. Random seed42, base32,7,760,097 parameters; RGB/255 at256 square, nearest masks, paired flips. Train1160/validate290 with image/mask SHA and RGB overlap checks. AdamW1e-4/weight decay1e-4, batch8, BCE+soft Dice,100 epochs.

Best is selected by resized256 validation binary Dice at threshold0.5; this is not directly comparable with native-resolution E8 macro Dice or historical798 test metrics. The798 entries did not select this initial model. Region masks provide derived contours; there is no separate edge supervision. Weight format is independent of E1/unet_edges and B.

From SystemCode/model:

~~~bash
python3 unet_scratch/train.py --output runs/unet_scratch_20261016 --device mps
~~~

A new run needs a nonexistent output. Before --resume, establish that the old process stopped; require matching source/split/config and complete checkpoint/optimizer/random state. Never run two processes against one output or claim cross-platform bitwise GPU equality.

history.json holds per-epoch losses, Dice/IoU/times; config.json holds source/split hashes; status.json records saved progress. A saved state is not proof of an active process. Sleep prevention does not protect against shutdown/logout/lid closure.

The original environment was Python3.9, PyTorch2.8.0, NumPy2.0.2 and Pillow11.3.0. It is distinct from the B research environment. No missing weights/results were invented or changed in this publication.
