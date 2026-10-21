# Experiment reports

Updated22 October2026. The retained application model is E8 B seed42 round98. E11 is incomplete, and no E0/B fusion has passed all promotion gates.

| Report | Research question and decision |
|---|---|
| [E0-E4 baselines](E0_E4_BASELINES.md) | Fixed reference, random U-Net, initialization controls and augmentation trade-offs |
| [E5-E7 fusion](E5_E7_FUSION.md) | Probability averaging, calibration, grouped CV and frozen gates; no qualifying fusion |
| [E8 data expansion](E8_DATA_EXPANSION.md) | Matched A/B/C/D controls, three seeds and historical C6; retain B with repeatability qualification |
| [E9 source balance](E9_BALANCE.md) | Ratio-only70/30 revision; retain50/50 B |
| [E10 B/E0 fusion](E10_FUSION.md) | Complementarity, averaging, temperature and local rescue; retain B |
| [E11 background training](E11_BACKGROUND.md) | Bounded background BCE intervention; incomplete56/100 saved seed42 rounds |
| [Roadmap](ROADMAP.md) | Why the direction changed, remaining confirmation and Web work |

The English E0-E4 summary replaces the five earlier Chinese Word progress reports in this branch's current tree. The original binary reports remain in Git history at commit b0858f3. This is a structured English consolidation, not a claim of page-for-page translation or newly executed experiments.

Exact compact CSV/JSON tables are in [model_results](../Miscellaneous/model_results/); the current snapshot has [its own index](../Miscellaneous/model_results/2026-10-22/README.md). Shared summaries are not complete runnable/recoverable training directories. Original paths/timestamps in numerical metadata describe historical provenance, not new installation locations.

No patient images are included in these new reports. Original course-template PDFs remain separately identified in Miscellaneous/template_reference and are not EndoLens results.
