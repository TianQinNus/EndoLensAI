# Historical U-Net region and boundary inference

This teammate utility reuses frozen e1_mac/unet.py and run_e1.py, adding a single-image boundary entry point. The model predicts regions; the red boundary is binary region minus3x3 erosion, not an independently edge-supervised network.

Run from SystemCode/model with E1 dependencies. Original E1 uses1450 pairs, fixed100 epochs, random initialization,352 RGB/ImageNet normalization, paired flips, BCE+soft Dice and AdamW. Final epoch is fixed; test entries do not select it.

~~~bash
python3 scripts/bootstrap_artifacts.py
python3 e1_mac/run_e1.py audit --output runs/unet_data_audit
python3 e1_mac/run_e1.py train --smoke --input-size 128 --base-channels 16 --device cpu --output runs/unet_edge_smoke --log-every 1
python3 unet_edges/test_boundary.py
~~~

Outputs must be new. Eight-image/one-round/five-test-entry smoke checks the pipeline only, not segmentation quality. Formal Mac training requires actual device checks and matching assets, not an assumption from this utility. Restore complete checkpoint/source/config for recovery; do not launch duplicate processes.

~~~bash
python3 unet_edges/predict.py --checkpoint runs/unet_full/checkpoints/unet_final.pth --image dataset/TrainDataset/images/348.png --output runs/unet_edge_prediction --device auto
~~~

Outputs are probability.npz, mask.png, edge.png, overlay.png and prediction.json with source/data/weight hashes. This entry point accepts the matching E1 Mac format only, not independent scratch/U-Net++ or B portable checkpoints. It restores logits to native size before sigmoid/threshold0.5 and derives contours afterward. Preserving this tool does not select it as the current Web model.
