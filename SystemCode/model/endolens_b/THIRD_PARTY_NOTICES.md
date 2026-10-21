# Sources and use boundaries

This is a research/teaching team integration release, not a blanket license for upstream code, datasets, weights or commercial use. Verify applicable original terms before public redistribution/commercial deployment.

## Training data and initialization

Original project development data derive from [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) and [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/). Additional positive frames use [PolypGen](https://github.com/DebeshJha/PolypGen), [paper](https://doi.org/10.1038/s41597-023-01981-y), [data identifier](https://www.synapse.org/Synapse:syn45200214), C1-C4 training/C5 validation/C6 historical evaluation.

The downloaded PolypGen archive license.txt says CC BY-NC-SA4.0 while README/paper label CC BY4.0. This unresolved discrepancy is retained; the effect on weight redistribution/commercial use requires clarification from rights holders, not an automatic legal conclusion from this export. Current intent is noncommercial research.

The encoder used torchvision's ImageNet-pretrained ResNet34 initialization before segmentation training. The complete trained parameters are included, but do not eliminate source/rights obligations.

No patient images, GT masks or dataset archives are newly distributed in this package. Examples are constructed synthetic graphics; NPZ/PNG outputs are actual model predictions on those graphics, not clinical labels. Reference parity records contain checks, not the real patient images used for replay.

## Dependencies

Dependencies are installed, not bundled as wheels/source. Their installed distribution LICENSE/NOTICE files govern use: PyTorch, torchvision, NumPy, Pillow, OpenCV, PyAV, FastAPI, Starlette, Uvicorn, python-multipart, HTTPX and transitives. requirements.txt and verification/dependency_versions.json record relevant versions.

PyAV-distributed FFmpeg components/codecs and platform distribution may have additional licensing/patent considerations; this prototype has not completed a commercial distribution review. React/Vite dependencies remain managed by the existing frontend project.

## Demonstration

Label the page/reports Research Demo / Not for Clinical Use. Do not assert medical certification, clinical accuracy, validated negative exclusion or unique-lesion counting. Do not upload identifiable clinical recordings. The local server has no production authentication and must not be exposed publicly.
