# E13 ResNet50 U-Net + region mean0.8 model card

## Identity and team decision

Version `e13-r50-joint-r94-r4-region080-v2`; source E13 seed42, best-positive
checkpoint at epoch94. The user selected this model and the E14 confidence-only
rule as the final project handover. This does not relabel E13 as having passed its
earlier validation tradeoff gate: E13 had no protected tradeoff checkpoint. The
user's final choice prioritizes lower false foreground and is documented explicitly.

Source checkpoint SHA256:
`1dadc0706f69069104f1395a3959ade36beca8be4be0b2c4b2cf821471a59088`.
Export weight SHA256 and file hashes are in model.json and MANIFEST.json.

## Architecture and training

ResNet50 encoder (channels64/256/512/1024/2048), five skip-connected U-Net decoder
stages256/128/64/32/16, double Conv3x3/GN8/ReLU blocks and a single1x1 head;
32,520,977 parameters. Bilinear upsampling. ImageNet V2 encoder initialization,
random decoder; both trainable, with encoder BN running statistics frozen.

E13 trained100 epochs from this initialization, with no positive-only segmentation
checkpoint. Input320,1160 replacement draws/epoch,50% positive/50% negative
probability; positive internal source mass original50%, each C1-C4 12.5%; negatives
image-uniform. AdamW1e-4 cosine to1e-5, weight decay1e-4, microbatch2/effective8,
FP32. Loss: original BCE + soft Dice +0.2 inner boundary Dice, smooth1, including
empty targets. Augmentation: H/V flips, right-angle rotations, brightness/contrast,
shrink+black-pad, saturation and gamma.

## Data roles

| Role | Positive | Negative | Total |
|---|---:|---:|---:|
| Training | 2269 | 1907 | 4176 |
| Validation | 494 | 733 | 1227 |
| Historical C6 exploration | 83 | 1760 | 1843 |

Positives: original1151 plus PolypGen C1-C4 1118 in training; original290+C5_204
validation; C6_83 evaluation. Negatives: verified empty single masks plus empty
author bboxes, or authored negativeOnly sequences. Training seq1-10/C1-C4,
validation seq11-13/C5, evaluation seq14-23/C6. No missing-mask negative inference.
Nine original quarantined near-duplicates were not restored; the clean E13
initialization has no inherited teammate segmentation exposure to them. Full
patient independence is not established by sequence IDs or similarity audits.

## Released inference and measured results

Input320, four-rotation mean logits, native interpolation before sigmoid, pixel0.5,
8-connected regions, delete regions with native mean score<0.8. No min-max,
minimum-area or temporal filtering. The exact rule is fixed in the SDK.

Reused C6 positive Dice0.85607267, IoU0.79299139;7/83 below Dice0.5 and3/83 zero
Dice. Negative any-foreground FPR369/1760=20.9659%; mean negative FP area0.33038%
of the image. E13 unfiltered FPR542/1760=30.7955%. See validation_results.json.

E14 explored150 postprocessing settings directly on this already-inspected C6
set at the user's request. These scores are exploratory and do not establish
independent generalization. Positive Dice must not be mixed with negative empty-mask
Dice to inflate a headline. Frame false foreground is not lesion-level detection
performance. The mean region score is not calibrated clinical confidence.

Intended use is team web integration and research demonstration on recorded
videos. Candidate counts count frames, not distinct polyps. Temporal behavior,
real-time throughput and independent clinical performance were not established
by this handover. No patient images or training data are included in the bundle.
