# E2 operating guide

This English guide replaces the earlier non-English filename. Run all commands from SystemCode/model; the [team guide](../docs/TEAM_GUIDE.md) contains the environment, restoration, development, selection, final training and evaluation commands.

Use the original frozen1160/290 split, seed42,100 development epochs and selected recipe. Compare original U-Net, random ResNet34-U-Net and pretrained ResNet34-U-Net; do not infer initialization effects from models with different structure. Formal training requires CUDA. Synthetic CPU tests are intentional.

After validation selects a recipe/epoch, reinitialize for full1450-entry training rather than continue development best. Keep the100-epoch scheduler horizon. Do not tune on the798 test entries or replace final epoch with a test-favorable checkpoint. The completed selection was pretrained ResNet34,67 epochs; see [results](../../../ProjectReport/E0_E4_BASELINES.md).

Restore complete original runs for recovery, preserve matching hashes, and never overwrite an existing experiment. Source/data/version mismatches require investigation, not disabled guards.
