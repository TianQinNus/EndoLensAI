# Third-party attribution and usage terms

## Polyp-PVT reference model

`SystemCode/model/lib/pvt.py` and `lib/pvtv2.py` were copied from the local checkout of [DengPingFan/Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT). The original implementation is by Bo Dong, Wenhai Wang, Jinpeng Li and Deng-Ping Fan; the paper includes additional coauthors. These are reference-model sources, not original EndoLens architectures. Local checkout revision: `cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4`; imported `lib` Python files have no tracked local modifications.

Paper: [Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers](https://arxiv.org/abs/2108.06932), CAAI Artificial Intelligence Research, 2023.

The original README states: “The source code is free for research and education use only. Any comercial use should get formal permission first.” The full original README is preserved in [UPSTREAM_POLYP_PVT.md](SystemCode/model/docs/UPSTREAM_POLYP_PVT.md); images and original training helpers are not bundled. Check upstream terms before redistribution or commercial use. No umbrella permissive license is applied here.

## U-Net, ResNet and library dependencies

The EndoLens U-Net and ResNet34-U-Net experiments implement established ideas; ResNet34 uses torchvision and official ImageNet initialization. Do not present them as newly invented architectures.

- [U-Net paper](https://arxiv.org/abs/1505.04597)
- [ResNet paper](https://arxiv.org/abs/1512.03385)
- [torchvision ResNet34](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html)

PyTorch, torchvision, timm, NumPy and Pillow remain subject to their respective licenses. Dependencies are installed, not vendored.

## Datasets and weights

Dataset images, segmentation labels, original training checkpoints and real-image probability caches are not newly committed. The explicit exception is the retained portable B inference weight, distributed as a reviewed regular Git binary, and clearly synthetic demonstration outputs. Obtain other assets through authorized sources or the group's transfer. Benchmark filenames/group identifiers and hashes are metadata, not a claim of restored patient identity. Availability does not imply unlimited reuse rights.

Expanded B used PolypGen C1-C4 for training, C5 validation and C6 historical evaluation; FAR-POLYP-SEG was used for development, not B training. PolypGen sources: [author repository](https://github.com/DebeshJha/PolypGen), [paper](https://doi.org/10.1038/s41597-023-01981-y), [data identifier](https://www.synapse.org/Synapse:syn45200214). The downloaded archive license.txt labels CC BY-NC-SA4.0 while README/paper label CC BY4.0. This unresolved discrepancy is retained; noncommercial research intent is not a legal conclusion about commercial use or downstream weight rights. Clarify terms with rights holders before broader distribution.

The reference Web stack adds OpenCV, PyAV/FFmpeg, FastAPI, Starlette, Uvicorn, python-multipart, HTTPX and transitives, each governed by its installed license/notice. See [the integration notices](SystemCode/model/endolens_b/THIRD_PARTY_NOTICES.md). No clinical/regulatory approval or blanket commercial authorization is granted.

## Course template

The repository originally contained a NUS-ISS HDB-BTO submission example. Its original README and example files are retained under `Miscellaneous/template_reference/` for reference; they are unrelated to the EndoLens implementation or results. Attribution inside those files is preserved.
