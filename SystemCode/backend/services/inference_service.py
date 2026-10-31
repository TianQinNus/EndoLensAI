import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..endolens_model import ModelAdapter
from ..endolens_model.postprocess import postprocess


def image_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


@dataclass
class InferenceService:
    adapter: ModelAdapter
    area_gate: float

    def predict(self, frame_rgb: np.ndarray, model_version: str):
        result = self.adapter.predict(frame_rgb, model_version)
        if self.area_gate != 0.0:
            raise ValueError("The final release has no area gate")
        # Production adapters already attach the canonical native filtered result.
        # The fallback supports injected raw-only test adapters with the same rule.
        filtered = result.postprocess_result
        if filtered is None:
            filtered = postprocess(result.probability_map)
        return result, filtered

    @staticmethod
    def source_image(frame_rgb: np.ndarray) -> Image.Image:
        return Image.fromarray(frame_rgb)

    @staticmethod
    def thumbnail(source: Image.Image) -> Image.Image:
        thumbnail = source.copy()
        thumbnail.thumbnail((320, 180), Image.Resampling.BILINEAR)
        return thumbnail

    @staticmethod
    def mask_image(result) -> Image.Image:
        return Image.fromarray(result.binary_mask * 255, mode="L")
