# Model assets and provenance

This directory contains fixed split/reference metadata and import registries, not patient images or complete training runs. Historical E2 split files preserve original bytes. Lightweight E0 and shared numerical records are not recovery checkpoints. Machine paths in old metadata describe provenance, not installation locations.

The current exception is the selected **portable B inference weight**, included as a regular binary at SystemCode/model/endolens_b/weights/unet_b_expanded_seed42.pt. Run the package verifier after cloning. The97,893,041-byte file fits the regular GitHub file limit; this public fork rejected new LFS objects, so no LFS retrieval is required. Do not add other ignored weights or datasets.

| Task | Additional assets required |
|---|---|
| Read reports and metrics | None |
| Synthetic historical functional tests | Declared Python dependencies |
| Current B inference/Web smoke | Selected included binary weight and package dependencies; synthetic video is included |
| E2 training | Original dataset and verified ResNet34 initializer |
| E4 development | E2 inputs plus complete E2-C reference run/best checkpoint |
| E0 inference | Original test data, PolypPVT.pth, backbone initializer and compatible environment |
| E3/E5/later cache replay | Complete matching source probability maps/masks/manifests and protocol-specific checkpoints |
| Interrupted training recovery | Complete original run, optimizer/RNG, unchanged source/data/config and matching environment |

WEIGHT_REGISTRY.json describes historical paths/hashes; it is not a fabricated download service. CODE_IMPORT_MANIFEST.json protects frozen E0-E4 Python bytes. B_RELEASE_IMPORT.json records the original tested local integration export; its documentation was subsequently migrated to English, while weights/inference remain unchanged.

On model_exploration, RESEARCH_CODE_IMPORT.json and research_snapshot contain original E5-E11 source/split/protocol provenance. Read the branch reproduction guide before using historical source hashes. Documentation or report translations must not silently rewrite metric values or historical protocol signatures.

Full original runs remain separately maintained by the experiment owner. Do not copy Miscellaneous/model_results into runs and assume resume/preflight/fusion will work. Dataset/weight rights remain subject to the [third-party notices](../../THIRD_PARTY_NOTICES.md).
