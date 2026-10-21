# E2 initialization controls

Run from SystemCode/model. Formal training requires CUDA; CPU synthetic tests are expected. Use the fixed runs/e2_data_seed42 split rather than regenerating it.

Compare original U-Net, random ResNet34-U-Net and ImageNet-pretrained ResNet34-U-Net on1160 training/290 validation entries,100 development epochs. Select architecture/epoch by validation Dice, then reinitialize on1450 entries for the selected epoch budget. Keep the100-epoch schedule horizon; never select from test results.

See the [team guide](../docs/TEAM_GUIDE.md) for installation, asset restoration and full commands, and [E0-E4 report](../../../ProjectReport/E0_E4_BASELINES.md) for actual completed results. metadata_template.csv is optional and must not be populated with guessed patient identity.

The frozen Python implementation is unchanged. Download the official verified encoder initializer using run_e2.py download-weights; do not substitute a trained final checkpoint.
