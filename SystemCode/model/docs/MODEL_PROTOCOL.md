# Model and evaluation protocol

## Data roles

Historical E0-E5 use the original1,450 training /798 test entries. E2/E4 development uses1,160/290, then reinitializes for full-data training at the validation-selected epoch count,67 and76 respectively. Test results do not select epochs, weights or thresholds. There are701 distinct decoded RGB test images; exact overlap audits do not establish patient independence.

E8 B uses1,151 original +1,118 PolypGen C1-C4 training images; validation is original290+C5 204. The1,160 image draws per round are with replacement, original/additional mass50/50, equal mass across new centers. A round is not a full-dataset epoch. C6 historical evaluation contains83 positive single frames. FAR432 uses160/272 images in75/112 disjoint patient groups, but has been repeatedly reused for development.

## Architecture and training

The retained B is a24,436,241-parameter ResNet34-U-Net. Its encoder was ImageNet-initialized; five decoder stages use bilinear upsampling, skip concatenation and two3x3 convolutions with GroupNorm8/ReLU, channels256/128/64/32/16, followed by a one-channel head. It is not Polyp-PVT or a fusion model.

B training uses strong paired flips/rotations within15 degrees and image-only brightness/contrast0.85-1.15. Labels use nearest interpolation. AdamW lr1e-4, weight decay1e-4, batch2/accumulation4, five-round warmup then the100-round cosine horizon, AMP forward and FP32 equal BCE+per-image soft Dice loss, smooth1. Encoder BatchNorm uses physical batch statistics; accumulation does not enlarge that batch.

B best selection maximizes0.5*original-validation mean Dice+0.5*C5 mean Dice, earliest tie. Seed42 round98 remains deployed; no favorable seed is substituted.

## Standardized inference

RGB uint8[H,W,3] -> Pillow bilinear352x352 square stretch -> /255 -> ImageNet mean[0.485,0.456,0.406], std[0.229,0.224,0.225] -> FP32 logits -> native-resolution bilinear resize, align_corners=False -> sigmoid float32[H,W]. No min-max, letterbox, TTA, region removal or AMP inference is silently added. Mask threshold is0.5.

E0 separately sums its two output logits. Historical cache compatibility includes backend numerical settings, not just tensor dtype; E10-D explicitly preserved distinct E0/B cuDNN TF32 settings. Do not relax replay tolerances to hide a mismatch.

## Metrics

Per image Dice=2TP/(2TP+FP+FN), IoU=TP/(TP+FP+FN). Both empty gives1; only one empty gives0. Group averages are image-equal; the original ALL is entry-weighted, not five-subset-equal. MAE compares continuous probability with binary GT.

Validation macro Dice in E8 is source-equal, not the494-image mean. Low/near-zero/zero Dice counts are reported alongside means. Pixel-pooled precision/recall is not image-equal recall or clinical lesion sensitivity. Seed sample SD is not a patient confidence interval. Cluster bootstrap is descriptive and does not undo adaptive selection.

## Fusion decisions

E3/E5 use fixed equal probabilities. E6-E7 and E10 have distinct frozen hypotheses/gates and failed promotion. E10-B equal fusion improves calibration Dice but adds zero-Dice cases relative to B. E10-C improves NLL/Brier but raises MAE and retains critical failures. E10-D can preserve B TP but add FP; accepting nothing is not a successful fused model. No failed candidate is a default Web option.

## Application contract

The [standalone SDK/API](../endolens_b/README.md) uses version unet-b-expanded-seed42-r98-v1. Keep original AI masks immutable and human revisions separate. Preserve real video PTS, source frame index, native dimensions and configuration hashes. Candidate frames are not unique lesions, foreground score is not calibrated diagnostic confidence, and failed/unsampled frames are unknown.

Only the segmentation threshold0.5 has the retained research protocol; the candidate area gate is unvalidated engineering behavior, default0 with empty masks explicitly false. No negative-frame specificity, clinical validity, authentication or real-time guarantee is implied. See the full [Web contract](../endolens_b/docs/WEB_INTEGRATION_GUIDE.md).
