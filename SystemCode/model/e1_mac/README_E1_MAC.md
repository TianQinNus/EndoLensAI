# Historical E1 U-Net on Mac

E1 trains a random conventional U-Net on1450 original pairs for a fixed100 epochs. It uses the frozen E0 test manifest, not test-selected epochs. The standard model has31,031,745 parameters, no BatchNorm/dropout,352 RGB input, paired flips, FP32 BCE+soft Dice, AdamW1e-4, batch2/accumulation4. This is not the current pretrained B.

From SystemCode/model, in a separate appropriate Mac environment:

~~~bash
python -m pip install -r e1_mac/requirements-e1-mac.txt
python e1_mac/check_device.py --device mps
python e1_mac/test_e1.py
python e1_mac/run_e1.py train --device mps --smoke
~~~

MPS must be genuinely available and pass forward/backward/AdamW. The original preparation environment tested CPU, not full MPS training. A standard-width352 smoke remains necessary; smaller CPU synthetic checks are software checks only. Explicit unavailable devices fail; record the actual selected backend and do not label mixed CPU fallback as pure GPU.

Historical formal command:

~~~bash
python e1_mac/run_e1.py train --device mps --e0-run runs/e0_20261004_221345_872444 --output runs/e1_unet_mps_seed42 --train-only
python e1_mac/run_e1.py evaluate --run-dir runs/e1_unet_mps_seed42 --device cpu --threads 4
~~~

Do not reuse an existing output. A train-only run is TRAINED, not a completed test evaluation. The final epoch is fixed, not selected by test/lowest training loss. Resume requires the matching full checkpoint, optimizer, data, configuration, source and environment; unfinished epochs are replayed, not mixed across platforms.

The actual completed project E1 training was later performed on Windows CUDA and evaluated on Mac CPU, with Dice0.5707. See the [English report](../../../ProjectReport/E0_E4_BASELINES.md) and [current team guide](../docs/TEAM_GUIDE.md). Historical preparation checks are not new claims about current Mac hardware compatibility.
