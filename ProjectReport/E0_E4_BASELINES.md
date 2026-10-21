# E0 to E4 baseline experiments

Updated 22 October 2026. This English research summary supersedes the five earlier Chinese Word progress reports in the current repository tree. The original reports remain recoverable from commit `b0858f3`; no local experimental artifacts were deleted. Numerical evidence remains in [model_results](../Miscellaneous/model_results/).

## Question and common measurement protocol

We first asked whether a self-trained U-Net could approach a fixed Polyp-PVT reference, whether ImageNet initialization helped, and whether stronger augmentation produced useful complementary errors. We did not train or fine-tune E0.

The original benchmark contains 1,450 training entries and 798 test entries: Kvasir 100, CVC-ClinicDB 62, CVC-ColonDB 380, ETIS 196 and CVC-300 60. Test entries represent 701 distinct decoded RGB images, not 798 independent patients. We retained the original entry weighting for comparisons. Exact RGB overlap checks do not establish patient independence.

Images are RGB, bilinearly resized to 352 square, divided by 255 and ImageNet-normalized. Logits are bilinearly restored to native resolution before sigmoid, with `align_corners=False`; binary masks use `p >= 0.5`. No per-image min-max normalization is used for fusion. Dice and IoU are computed per image and averaged; MAE uses continuous probabilities. These metrics are not a direct reproduction of the paper's MATLAB evaluation.

## E0 fixed reference

The existing Polyp-PVT checkpoint was evaluated without training. Its two output logits are summed before native-resolution restoration. All 798 probability/mask pairs and metrics were replay-checked. E0 mean Dice was 0.836783, IoU 0.762996 and MAE 0.021855; 68 images had Dice below 0.5 and 14 had zero Dice.

The original Mac CPU forward-only average of about 197.78 ms excluded preprocessing, resizing, metrics and storage. It is not a video/Web FPS measurement. Checkpoint SHA256: `aef5ac34f7795b85af0ecbb730502db251d416a73a4c6b04a74e30aa45554048`.

## E1 random U-Net

A conventional padded U-Net, 31,031,745 parameters, was trained from random initialization on all 1,450 training entries for a fixed 100 epochs. It used paired horizontal/vertical flips, AdamW at 1e-4 with weight decay 1e-4, batch 2 with accumulation 4, and equal BCE plus per-image soft Dice loss, smooth 1. The fixed final epoch, not the best test epoch, was evaluated.

Mean test Dice/IoU were 0.5707/0.4876; MAE was about 0.04836, with 273 low-Dice and 158 zero-Dice entries. Falling training loss did not establish sufficient generalization. E1 provided a reproducible baseline, not an improvement over E0.

## E2 initialization controls

The fixed internal split was 1,160 development training / 290 validation entries. Three 100-epoch controls compared conventional U-Net, random ResNet34-U-Net, and ImageNet-pretrained ResNet34-U-Net. The latter two share architecture; their comparison isolates initialization more closely than the conventional-versus-ResNet comparison.

| Development arm | Best epoch | Best validation Dice |
|---|---:|---:|
| Conventional U-Net | 61 | 0.807327 |
| Random ResNet34 encoder | 66 | 0.872517 |
| ImageNet-pretrained ResNet34 encoder | 67 | 0.916181 |

The selected pretrained recipe was reinitialized and trained on all 1,450 entries for 67 epochs, retaining the original 100-epoch learning-rate schedule. It was not continued from development best. Test Dice/IoU were 0.754879/0.686153. Pretraining substantially improved this self-trained route, but E0 remained stronger on this benchmark.

## E3 fixed probability fusion

E3 averaged the frozen native-resolution E0 and E2 probability maps equally at threshold 0.5, without training or weight search. Mean Dice/IoU were 0.832898/0.760585, below E0. Low-Dice/zero-Dice counts were 72/17 versus E0's 68/14. Individual complementary cases did not establish aggregate or tail improvement.

## E4 stronger augmentation

Development changed only `flips` to `strong`: paired flips, paired rotations within 15 degrees, and image-only brightness/contrast factors 0.85 to 1.15. Geometry is shared with masks; masks use nearest-neighbor interpolation. Architecture, initialization source, loss, optimizer and development budget were held fixed.

Best validation Dice changed from 0.916181 to approximately 0.91777, selecting epoch 76 rather than 67. The final recipe was reinitialized and trained on all 1,450 entries for 76 epochs. Thus final E2-versus-E4 compares complete selected recipes, not augmentation alone.

| Subset | n | E0 Dice | E2 Dice | E4 Dice |
|---|---:|---:|---:|---:|
| All | 798 | 0.8368 | 0.7549 | 0.7609 |
| Kvasir | 100 | 0.9204 | 0.8885 | 0.9116 |
| CVC-ClinicDB | 62 | 0.9403 | 0.8851 | 0.8755 |
| CVC-ColonDB | 380 | 0.8114 | 0.7545 | 0.7295 |
| ETIS | 196 | 0.7901 | 0.6105 | 0.6828 |
| CVC-300 | 60 | 0.9039 | 0.8718 | 0.8456 |

E4 improved Kvasir/ETIS but worsened the other three subsets. MAE increased from 0.034564 to 0.045028. Zero Dice decreased from 51 to 19, while Dice below 0.5 increased from 149 to 151. Pooled pixel precision decreased from approximately 0.6813 to 0.6110 and recall increased from 0.7748 to 0.8313. These pooled pixel measures are not clinical lesion detection measures.

## Decision and limitations

We retained E2/E4 as distinct frozen candidates and proceeded to fixed E0+E4 fusion, rather than declaring stronger augmentation universally better. Original test data have been repeatedly inspected and are not fresh confirmation data. All images in these evaluations are polyp-positive; no negative-frame specificity, lesion counting, clinical efficacy or real-time video claim follows.

Historical CPU and GPU timings used different environments and scopes. We do not infer an end-to-end speed advantage from them. Current progress and later decisions are in the [report index](README.md).
