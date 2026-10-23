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
        return result, postprocess(
            result.probability_map,
            segmentation_threshold=0.5,
            min_region_area_ratio=self.area_gate,
        )

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
