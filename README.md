# EndoLens AI

Polyp segmentation research and interactive recorded-video analysis.

We study whether a pretrained U-Net can generalize better through additional training data and controlled error-focused training, while keeping an existing Polyp-PVT model frozen as a reference. Fusion is an evaluated research question, not an assumed product improvement.

**Status on 22 October 2026:** E0-E10 experiments are documented; original E8 B is the retained integration baseline. E9 and E10 fusion revisions did not meet their full declared gates. E11 is incomplete: the latest saved seed42 record is56/100 rounds, not a completed result. A recorded-video model/API starter is tested; the existing React application still needs integration. This project is a research/teaching prototype, not clinical software.

## Start with the current model

The integration version is **unet-b-expanded-seed42-r98-v1**: ImageNet-pretrained ResNet34 encoder, five-stage U-Net decoder, seed42, validation-selected round98. The complete97,893,041-byte weight is included as a regular binary Git file; training datasets are not included. This public fork rejected new Git LFS objects, so no LFS download is required.

~~~bash
git clone --branch master https://github.com/TianQinNus/EndoLensAI.git
cd EndoLensAI
cd SystemCode/model/endolens_b
~~~

Follow the [model installation and startup guide](SystemCode/model/endolens_b/README.md). CPU inference is supported. Create a separate Web environment rather than changing an active training environment. After starting the backend, run in a second terminal:

~~~bash
python examples/api_smoke.py
~~~

The [Web integration guide](SystemCode/model/endolens_b/docs/WEB_INTEGRATION_GUIDE.md) maps upload, progress, native-resolution masks, candidate frames, keyframes, reviews and reports to the existing UI. It includes React request/overlay examples and a frozen OpenAPI snapshot. This is not a complete replacement frontend. The UI source currently lives on the [frontend branch](https://github.com/TianQinNus/EndoLensAI/tree/frontend); this update does not merge or overwrite it.

## Current results and limits

These are separate datasets/roles, not directly comparable rows in one benchmark.

| Evaluation | Model | Images | Mean Dice | MAE | Dice<0.5 | Zero Dice |
|---|---|---:|---:|---:|---:|---:|
| Historical PolypGen C6 positive frames | B | 83 | 0.847092 | 0.015525 | 7 | 2 |
| Same C6 evaluation | E0 | 83 | 0.830458 | 0.019135 | 10 | 4 |
| Reused FAR development, all images | B | 432 | 0.774268 | 0.037177 | 62 | 4 |
| Same FAR evaluation | E0 | 432 | 0.765853 | 0.040069 | 68 | 10 |

B's three-seed validation macro Dice is0.805842 +/-0.001631 sample SD versus matched original-data A0.750980 +/-0.005561. The full original source-wise gates did not repeat in every seed. FAR calibration Dice favors E0, and the descriptive patient-group interval for all-FAR B-E0 Dice includes zero. We do not claim universal or statistically established superiority over Polyp-PVT.

Equal B/E0 fusion reached calibration Dice0.779279 versus B0.753331 but increased zero-Dice count from2 to3. Temperature fusion reached OOF Dice0.787784 but MAE0.056249 and still failed protection gates. No fusion was promoted. See [complete experimental reports](ProjectReport/README.md) and [machine-readable evidence](Miscellaneous/model_results/2026-10-22/README.md).

## Research history and branch responsibilities

| Branch | Purpose |
|---|---|
| master | Current retained model, runnable integration starter, English project/progress documentation and compact evidence |
| model_exploration | E5-E11 critical experiment implementations, controlled hypotheses, protocols, reproduction notes and retained teammate experiments |
| frontend | Existing React/Vite work, maintained separately by the Web team |

Read the [research decision roadmap](ProjectReport/ROADMAP.md), [current experiment status](SystemCode/model/docs/EXPERIMENT_STATUS.md), [model protocol](SystemCode/model/docs/MODEL_PROTOCOL.md) and [team guide](SystemCode/model/docs/TEAM_GUIDE.md). E0-E4 sources remain frozen. On model_exploration, presentation-only English changes to later scripts are documented separately from the original executed source and hashes; never bypass a historical preflight to resume a changed experiment.

## Repository layout

~~~text
SystemCode/model/
  endolens_b/          Retained B weights, SDK, reference API, tests and Web examples
  e0_mac/              Historical fixed Polyp-PVT evaluation
  e1_mac/, e1_windows/ Historical random U-Net
  e2_windows/          Initialization controls
  e4_windows/          Strong augmentation control and frozen training core
  v3_windows/          Historical E0+E2 equal fusion
  docs/, scripts/      Protocols, team guidance and integrity checks
ProjectReport/         English E0-E11 research reports and roadmap
Miscellaneous/
  model_results/       Compact historical and current evidence, not resumable runs
  model_artifacts/     Split/protocol/source registries, no dataset images
  template_reference/  Unrelated original course template, not EndoLens work
Video/                 Demonstration guidance
~~~

The research branch additionally contains e5_windows through e11_background_windows. All experiment commands run from SystemCode/model; the standalone Web package commands run from SystemCode/model/endolens_b.

## Responsible interpretation and data rights

Candidate counts are **frames, not unique polyps**. Foreground scores are not calibrated diagnosis probabilities. Unprocessed/failed frames are unknown, not confirmed normal. No negative-frame specificity, lesion tracking, real-time endoscopy, clinical efficacy or deployment authorization has been established.

Historical798 entries, FAR, C5 and C6 have been viewed; none can be renamed a fresh blind confirmation set. Data expansion used1,151 original +1,118 PolypGen C1-C4 training images, center-balanced sampling, with original290+C5 204 validation. See [E8](ProjectReport/E8_DATA_EXPANSION.md).

Do not upload private clinical footage, dataset images, credentials or complete training caches. The local API has no production authentication or user isolation and must not be exposed publicly. See [third-party notices](THIRD_PARTY_NOTICES.md), including the unresolved PolypGen license-label discrepancy. No blanket commercial or open-source license is granted to upstream assets.

## Contributing

Project-authored documentation, source comments, CLI/API text and current report filenames use English. Keep numerical records and provenance intact; do not translate identifiers into a different experimental meaning. Original non-English reports remain in Git history. Follow [CONTRIBUTING.md](CONTRIBUTING.md), preserve teammate work, and do not force-push shared branches.
