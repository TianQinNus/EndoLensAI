# Retained B model card

English repository edition22 October2026. Intended use: research/teaching and recorded-video team integration, not clinical diagnosis, screening or a medical device.

## Identity

| Field | Value |
|---|---|
| Version | unet-b-expanded-seed42-r98-v1 |
| Research checkpoint | E8 B_expanded, seed42, validation-selected round98 |
| Architecture | ImageNet-initialized ResNet34 encoder and five-stage U-Net decoder |
| Parameters | 24,436,241 |
| Decoder | Bilinear upsampling, skips, two3x3 Conv/GroupNorm8/ReLU blocks; channels256/128/64/32/16; one-channel head |
| Original checkpoint SHA256 | 2ee7145fe5509d6cdbeefd90c8a2f0d87ca7f1d9daf78ce79e7edb4764ee7b5f |
| Portable weight SHA256 | 1e4531592942ac8683b3ea7ea7ac9af6e77e03b08a2e3481942cfbbb24f657bb |

Export removed optimizer/training-path dependencies and did not retrain. The file contains complete inference parameters; no parameter download occurs at initialization. No E0, ensemble, scratch U-Net++ or incomplete E11 weight is selected.

## Training and selection

2,269 positive training images:1,151 original Kvasir/ClinicDB development entries after near-duplicate quarantine and1,118 PolypGen C1-C4(251/267/456/144). Validation494=original290+C5 204; C6 did not train/select B. Image counts are not independent patient counts.

100 rounds with1,160 replacement draws per round; original/additional mass50/50, new centers equally weighted. This B has no image hard weighting. Strong augmentation,352 input, AdamW1e-4/weight decay1e-4, batch2/accumulation4, AMP training, FP32 equal BCE+per-image soft Dice smooth1. Best maximizes0.5*old-validation Dice+0.5*C5 Dice, earliest tie. Selected seed42 macro Dice0.8043223644.

## Inference

Strict RGB uint8[H,W,3], Pillow bilinear352 square stretch, /255, ImageNet normalization, FP32 forward, native logits bilinear restore with align_corners=False, sigmoid float32[H,W], threshold0.5. No min-max, letterbox, TTA, fusion or connected-component deletion.

Three local GPU cache replays were bitwise exact; CPU fixture checks use a numerical tolerance. This does not guarantee equality across hardware/software. Area-based candidate gating is a separate unvalidated engineering rule, default0 with empty mask false. Scores are not calibrated diagnostic probabilities.

## Existing performance, not new release evaluation

| Role | Images | Dice | MAE | Dice<0.5 / <0.1 / =0 |
|---|---:|---:|---:|---|
| Historical C6 positive-frame evaluation | 83 | 0.847092 | 0.015525 | 7 / 2 / 2 |
| Adaptively reused FAR development | 432 | 0.774268 | 0.037177 | 62 / 12 / 4 |

validation_results.json retains precise aggregate values. Three-seed validation macro Dice0.80584194 +/-0.00163119 sample SD is not an external/patient confidence interval. Source-wise original gates did not pass in every seed. FAR calibration Dice favors E0; aggregate B/E0 comparisons do not establish universal architectural superiority.

## Risks and missing validation

Large false-foreground regions, localization errors and small-target misses remain possible. Positive-only evaluations do not establish negative-frame specificity, lesion-level sensitivity, pathology, unique polyp counting, real-time performance or clinical safety. Original data partly lack reliable patient identity; heuristic deduplication is incomplete. FAR/C5/C6 and the original798 have been viewed and are not new blind confirmation sets.

No millimeter calibration, tracking, prospective multicenter confirmation or regulatory approval is provided. Use authorized deidentified data only; unresolved dataset licensing is documented in THIRD_PARTY_NOTICES.md.

verification/ covers software parity, CPU fixture and synthetic API/video behavior. Synthetic content has no true clinical lesion and cannot benchmark diagnostic accuracy/FPS. Teammate React integration and production security remain unverified.
