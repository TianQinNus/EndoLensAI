from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..endolens_model import MODEL_VERSION


class AnalysisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    sampling_mode: Literal["interval", "every_frame"] = "interval"
    interval_seconds: float | None = Field(
        default=1.0, ge=0.1, le=900, allow_inf_nan=False
    )
    model_version: Literal[MODEL_VERSION] = MODEL_VERSION

    @model_validator(mode="after")
    def consistent(self):
        if self.sampling_mode == "interval" and self.interval_seconds is None:
            raise ValueError("interval_seconds required for interval mode")
        if self.sampling_mode == "every_frame":
            self.interval_seconds = None
        return self


class ExportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["csv", "json"] = "csv"
    include_reviewed_masks: bool = True
