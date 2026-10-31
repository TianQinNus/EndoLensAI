# Final E13 model release

Published 31 October 2026. This record supersedes older statements identifying
ResNet34 B as the current integration model. Historical source snapshots and
NO_PASS outcomes remain unchanged.

## Frozen identities

| Item | Value |
|---|---|
| Model | `e13-r50-joint-r94-r4-region080-v2` |
| Output rule | `native-cc8-pixel050-mean080-v1` |
| Architecture | ImageNet ResNet50 encoder, five-stage GroupNorm U-Net decoder |
| Parameters | 32,520,977 |
| Selected checkpoint | Seed 42, epoch 94, best-positive validation fallback |
| Source checkpoint SHA256 | `1dadc0706f69069104f1395a3959ade36beca8be4be0b2c4b2cf821471a59088` |
| Portable weight SHA256 | `dc6c9092baf68cf9e66264f632fae76642ff621282aaa7f65dbf8a899f168e68` |
| Weight bytes | 130,413,723 |
| Download and installation | [SDK guide](../endolens_e13/README.md) |

## Architecture and training

The encoder produces 64/256/512/1,024/2,048-channel features. Decoder stages
produce 256/128/64/32/16 channels through bilinear upsampling, corresponding skip
concatenation where available, and two 3x3 convolution/GroupNorm(8)/ReLU blocks.
A 1x1 convolution returns one foreground logit. Inference loading uses no external
ImageNet download. ImageNet normalisation is embedded in the model.

E13 starts with ImageNet ResNet50 V2 encoder weights and a random decoder, not a
previous positive-only segmentation checkpoint. Both encoder and decoder train.
Encoder BatchNorm running statistics are fixed, with affine parameters trainable.

| Training choice | Configuration |
|---|---|
| Budget | 100 epochs, 1,160 replacement draws per epoch |
| Sampling | 50% positive, 50% negative from epoch 1 |
| Conditional positive sources | Original 50%; PolypGen C1/C2/C3/C4 each 12.5% |
| Conditional negative sampling | Uniform by training image |
| Input and precision | 320 x 320 RGB, FP32, no AMP |
| Optimiser | AdamW, weight decay 1e-4 |
| Learning rate | Cosine from 1e-4 to 1e-5 |
| Batch | Microbatch 2, accumulation 4, effective batch 8 |
| Loss | BCE + soft Dice + 0.2 inner-boundary Dice, smoothing 1 |
| Empty targets | All loss terms retained with smoothing, unlike E12 |
| Geometry | H/V flip p=0.5 each, right-angle rotations, shrink/pad p=0.5, scale 0.4-0.8 |
| Photometric | Brightness/contrast 0.8-1.2; saturation p=0.5, 0.5-1.5; gamma p=0.5, 0.7-1.5 |
| Validation | Identity inference, 320 input, labels resized to 256 |

Validation positive Dice/IoU were 0.79826/0.72943, with 34 zero-Dice images.
The supplied positive-only ResNet50 reference had 0.85363/0.79103 and 11 zero-Dice
images under that protocol. Negative validation frame FPR decreased from 97.68%
to 65.62%. No checkpoint passed every original positive-protection guard, so no
protected `best_tradeoff` checkpoint exists. The team's later final choice accepts
this trade-off explicitly. A matched fresh-start positive-only E13 control was
not completed, so the full difference cannot be attributed solely to negatives.

## Frozen data accounting

| Source | Train | Validation | C6 evaluation |
|---|---:|---:|---:|
| Original positive images | 1,151 | 290 | 0 |
| PolypGen positive images | 1,118 | 204 | 83 |
| Negative sequence frames | 1,789 | 731 | 1,755 |
| Verified empty single frames | 118 | 2 | 5 |
| All positive images | 2,269 | 494 | 83 |
| All negative images | 1,907 | 733 | 1,760 |
| All images | 4,176 | 1,227 | 1,843 |

Nine near-duplicate original training images were excluded. Positive PolypGen
centres C1-C4 train, C5 validates and C6 evaluates. Negative sequences 1-10 train,
11-13 validate and 14-23 evaluate. Empty single frames follow the centre split,
with both mask and bounding-box annotations verified empty. Missing annotations
are not negatives. Grouping reduces adjacent-frame leakage but does not prove
full patient independence across inherited datasets and checkpoints.

## Exact deployed inference

RGB uint8 input is Pillow-bilinear resized to 320 square and divided by 255.
ImageNet normalisation occurs internally once. Predict four right-angle rotations,
inverse-align and average logits. Bilinearly restore logits to native dimensions
with `align_corners=False`, then apply sigmoid once. Compute 8-connected components
of pixels >=0.5. Retain components whose unrounded native mean score is >=0.8.

The filtered uint8 0/1 `binary_mask` is canonical for all mask PNGs, contours,
regions and candidate status. An empty result is not a candidate. Raw
`probability_map` remains diagnostic only. The rule does not use an area filter,
largest-only selection, temporal confirmation or another model. Scores are not
calibrated disease probabilities.

## Matched development results

All models: 320 input, FP32, rotate4, native scoring, 83 positives and 1,760
negatives. These are the exact same comparison conditions, not identical training
histories. The positive-only comparators use pixel threshold 0.5 without the
selected E13 component filter.

| Model | Positive mean Dice | Positive mean IoU | Negative FP frames | Negative frame FPR |
|---|---:|---:|---:|---:|
| Polyp-PVT | 0.84356947 | 0.78978453 | 1,165 | 66.1932% |
| ResNet34 B | 0.84647033 | 0.78439558 | 986 | 56.0227% |
| Positive-only ResNet50 | 0.88388883 | 0.82727448 | 1,579 | 89.7159% |
| E13 raw | 0.85882717 | 0.79539512 | 542 | 30.7955% |
| Final E13 | 0.85607267 | 0.79299139 | 369 | 20.9659% |

Dice/IoU are image-equal means over positive images only. Empty negative images
do not inflate this mean. Negative-frame FPR is the fraction of negative images
with any retained foreground, not a pixel-level FPR. Final mean negative false
foreground area fraction is 0.00330383. Ten-sequence macro FPR is 0.20990372,
computed on the 1,755 sequence frames, excluding five negative singles.

The final rule removes 173 negative-frame predictions relative to raw E13, a
9.8295 percentage-point or 31.92% relative FPR reduction. Positive Dice falls
0.00275450 and IoU 0.00240373. Of 83 positive Dice values, 76 stay unchanged,
three improve and four worsen. Seven final images have Dice <0.5, three have
zero Dice and one has an empty prediction. 82 masks are nonempty, but only 80
overlap the reference: candidate coverage is not correct localisation or lesion
sensitivity.

The positive-only R50 has the highest Dice. Final E13 has 0.02782 lower Dice and
68.75 percentage points lower negative-frame FPR. The team selected this
operating point for fewer false alarms, not as the highest-overlap model.
The comparators' own best filtered operating points were not evaluated.

## Timing boundary

RTX 5060 Laptop GPU, Intel Core Ultra 9 275HX, PyTorch 2.9.1+cu128, CUDA 12.8,
batch 1, FP32, AMP/TF32 off, deterministic cuDNN, four CPU threads. Each scope
uses 20 warm-ups and three repeats of 100 timed calls on the same ten validation
images (five positive, five negative), each native 1920 x 1080. CUDA synchronises
before and after each call. These are warm serial measurements, not cold starts.

| E13 scope | Mean ms | Median ms | p95 ms |
|---|---:|---:|---:|
| One network forward | 18.41 | 18.26 | 21.01 |
| Four-view network | 67.25 | 66.66 | 71.46 |
| Complete frame SDK | 114.75 | 110.76 | 136.76 |

The SDK includes preprocessing, host/device transfer, four network calls,
native restoration, CPU return and connected-region filtering. It excludes video
decoding, HTTP, queues, persistence and browser display. Separate loop differences
are not per-stage profiling. No application FPS or clinical real-time claim follows.

## Evidence and remaining limits

[Compact metrics and timing](../../../Miscellaneous/model_results/2026-10-31/README.md)
come from the completed chapter-10 test bundle, rather than a new evaluation in
this publication. SDK source bytes match the verified v2 handover. Packaging parity
previously checked all 1,843 cached native probability maps, six exported-model
forward cases and 22 SDK/API tests. This repository update additionally checks the
modular backend's filtered-output contract and real CPU fixture.

C6 was reused and selected the postprocessing threshold. FAR and the historical
798-image benchmark were also used adaptively. E13 has one training seed, lacks
an identical fresh-start positive-only control, and did not satisfy all original
positive-protection gates. A new external patient/video-grouped positive/negative
evaluation with the model and rule locked beforehand remains necessary.
