# Research branch guide

Updated22 October2026. This branch retains the executed E5-E11 implementation, full experimental reasoning in [ProjectReport](../../../ProjectReport/README.md), compact result evidence and separate teammate scratch/outline experiments. B remains the selected integration model; no fusion or incomplete E11 checkpoint is promoted.

## Read the evidence before rerunning

Start with [the roadmap](../../../ProjectReport/ROADMAP.md). E5-E7 studied E0/E4 fusion; E8 changed training-data/source exposure with A/B/C/D controls; E9 changed only source ratio; E10 studied retained B/E0 complementarity and constrained fusion; E11 changes only bounded background-pixel BCE weighting. Their datasets, validation rules and failure statuses are not interchangeable.

The current branch's README, stage guides, code comments, report/CLI text and report filenames are English. E0-E4 Python remains byte-identical. Ten later/teammate files have presentation-only text changes recorded in [the English migration registry](../../../Miscellaneous/model_artifacts/ENGLISH_SOURCE_MIGRATION.json). The verifier compares every AST field except originally non-English text constants; numeric constants, identifiers, control structure and all other strings must remain unchanged. It also verifies the original131 imported files against the archived commit.

## Original source and safe historical recovery

Executed E5-E11 source bytes are preserved at commit **c00b15fae599ae00d788393f366016e4e9c9516b** before presentation changes. Old plans/run metadata retain their original source checksums; they are not edited to match English files. Therefore a historical preflight may correctly reject the current English renderer's hash. Do not disable that guard or silently rewrite an old protocol.

For genuine historical recovery, create a separate checkout at the archive commit, for example from repository root:

~~~bash
git worktree add --detach ../EndoLensAI-executed-research c00b15fae599ae00d788393f366016e4e9c9516b
~~~

Then work in that checkout's SystemCode/model and restore the exact authorized original full runs/data/initializers. Never move a running local job into another checkout, replace its source midway or start a second process against its output. This publication did not modify the original local research project. Old non-English report strings remain historical source only, not current project-facing text.

The archive is code/provenance, not a complete recovery bundle. Shared E0 records may have been sanitized for publication; historical preflights can require the original complete private E0 directory. Complete optimizer/RNG/checkpoint/cache assets must come from their owner. CSV/JSON summaries do not replace them.

## Metadata and data roles

Miscellaneous/model_artifacts/research_snapshot contains the original FAR development manifest, audited E8 split files and frozen plans. Keep them read-only as evidence. Do not bulk-copy plans over existing runs or regenerate a more favorable split. Images, dataset archives, real masks, E0 weights/backbone, ResNet initializer and full probability caches are intentionally not included. The included B binary is a portable inference export, not a resumable optimizer checkpoint.

The root bootstrap restores only historical E2/E0 lightweight metadata. It does not restore E8/E10 complete dependencies. A new from-scratch English experiment needs audited data, a separately frozen current-source protocol and fresh outputs, not patched historical run hashes. Follow the original stage-specific checks and stop at decision points.

## Environment and safe regression checks

The research environment was Windows Python3.11, torch2.9.1+cu128, torchvision0.24.1+cu128, NumPy1.26.4 and Pillow10.2.0. Formal training requires real CUDA proof/smoke; the following are isolated CPU synthetic regression suites, including tiny synthetic update/recovery checks, not full experiments:

~~~bash
python scripts/verify_imported_code.py
python scripts/verify_research_publication.py
python scripts/check_repository.py
python scripts/run_checks.py
python scripts/run_research_checks.py --group all
~~~

Run from SystemCode/model except check_repository/verification scripts, which also resolve the root automatically. Logs are ignored under runs/publication_checks. Individual stage test scripts remain available. On restricted Windows hosts, allow the process a writable temporary directory; do not change frozen code to work around host permissions. Grouped execution can be split into early/training/fusion/teammate for practical checking.

No test reruns are a license to continue formal selection after NO_PASS. E10-B/C/D must remain blocked at their recorded stop points; E11 is not complete. Recover its actual process/run first, finish all declared seeds and apply unchanged gates before considering a new model version.

## Key implementation entry points

| Question | Code |
|---|---|
| Fixed E0/E4 fusion and source audit | e5_windows/run_e5.py |
| Development resource audit and calibrated selection | e6_windows/run_e6.py |
| Constrained affine revisions | e7a_windows/, e7a2_windows/ |
| Patient OOF diagnosis and prediction-only gates | e7b0_windows/, e7b1_windows/, e7b2_windows/ |
| Data audit/source probabilities/hard sampling | e8_windows/prepare_data.py, train_e8.py |
| Matched-seed summaries and ratio-only revision | e8_repeat_windows/, e9_balance_windows/ |
| B/E0 comparison, fixed fusion and temperature CV | e10a_windows/, e10b_windows/, e10c_windows/ |
| Prediction-only TTA component rescue | e10d_windows/run_e10d.py |
| Bounded background loss and matched checks | e11_background_windows/background_loss.py, run_e11.py |

Teammate unet_scratch*/unetpp*/unet_evaluation/unet_edges/polyp_outline work is preserved separately. Its initialization, input resolution and validation metrics differ from E8 B; do not compare unlike metric protocols or replace the B SDK with its weight format. No missing teammate weights or unprovided outcomes are invented.

## Publication boundaries

No raw patient imagery or private uploads were added. Dataset/weight rights still apply, including the unresolved PolypGen license-label discrepancy. New confirmation needs genuinely new audited data and declared positive/negative/video roles; FAR/C5/C6 and the original798 are already viewed. Keep original AI outputs distinct from human revisions and software smoke distinct from medical evidence.
