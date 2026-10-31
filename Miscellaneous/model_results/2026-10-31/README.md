# Final model evidence, 31 October 2026

These compact files reproduce the completed chapter-10 result bundle, without
dataset images, probability maps, private videos or training checkpoints.

- `model_comparison.csv`: matched positive segmentation and negative-frame errors
  at 320/FP32/rotate4/native resolution, 83 positive and 1,760 negative images.
- `negative_sequences.csv`: per-sequence false-positive results. Ten groups,
  1,755 negative sequence frames. Five negative singles do not enter the macro mean.
- `latency_summary.csv`: 300 timed calls per model/scope after 20 warm-ups,
  RTX 5060 Laptop GPU, fixed validation inputs. Network and full-SDK boundaries differ.

C6 was reused in development, including final output-rule selection. Values do
not establish clinical sensitivity, an independent final-test result or browser FPS.
The positive-only comparators were not tested at their best filtered operating points.

See the [full release/evaluation record](../../../SystemCode/model/docs/FINAL_MODEL_RELEASE.md)
and [current SDK](../../../SystemCode/model/endolens_e13/README.md).
