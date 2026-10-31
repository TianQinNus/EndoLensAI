# Sources and third-party notices

This is a project-team research/integration handover, not a new grant of rights
to upstream data, weights or dependencies. Retain the applicable upstream notices.

- Original positive development data: Kvasir-SEG and CVC-ClinicDB project splits.
  Sources: https://datasets.simula.no/kvasir-seg/ and
  https://polyp.grand-challenge.org/CVCClinicDB/ .
- Expanded positive/negative data: PolypGen. Author repository:
  https://github.com/DebeshJha/PolypGen ; data descriptor:
  https://doi.org/10.1038/s41597-023-01981-y . The downloaded archive has a known
  notice discrepancy: license.txt states CC BY-NC-SA4.0 while README/paper indicate
  CC BY4.0. The project records that discrepancy and uses the data for noncommercial
  research; this document makes no legal conclusion about downstream weight rights.
- Encoder initialization: official torchvision ImageNet1K V2 ResNet50. Full
  trained segmentation parameters are included; inference downloads nothing.
- Architecture: teammate-provided ResNet50/GN U-Net adapted for a portable SDK.
  No original backbone or U-Net architecture claim is made.

No raw patient frame, mask or dataset archive is redistributed. Example images
and videos are explicitly synthetic. Reference probabilities/masks are actual
outputs on these synthetic graphics, not clinical ground truth. Verification
records contain aggregate results and source sample IDs, without patient imagery.

Dependencies are installed separately, not bundled as wheels or copied source:
PyTorch/torchvision, NumPy, Pillow, OpenCV, PyAV, FastAPI, Starlette, Uvicorn,
python-multipart and HTTPX. Their licenses and notices remain those distributed
with the installed packages. PyAV/FFmpeg codec terms depend on the chosen binary
distribution. Python package versions are in requirements and verification records.

The optional local reference server has no authentication or user isolation.
Public hosting controls are part of the receiving web project's deployment work.
