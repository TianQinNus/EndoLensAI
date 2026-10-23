import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD = 100 * 1024**2
WARNINGS = [
    "RESEARCH_DEMO_NOT_CLINICAL",
    "POSITIVE_ONLY_MODEL:NO_NEGATIVE_FRAME_SPECIFICITY_CLAIM",
    "AREA_GATE_UNVALIDATED_ENGINEERING_RULE",
    "CANDIDATE_COUNT_IS_FRAMES_NOT_DISTINCT_POLYPS",
    "FOREGROUND_SCORE_NOT_CALIBRATED_CLINICAL_PROBABILITY",
]


def area_gate() -> float:
    value = float(os.getenv("ENDO_MIN_REGION_AREA_RATIO", "0"))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid ENDO_MIN_REGION_AREA_RATIO")
    return value


def storage_dir() -> str:
    return os.getenv("ENDO_STORAGE_DIR", str(ROOT / "runtime_data"))


def device(requested: str | None = None) -> str:
    return requested or os.getenv("ENDO_DEVICE", "auto")


def cors_origins() -> list[str]:
    origins = [
        origin.strip()
        for origin in os.getenv(
            "ENDO_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
        ).split(",")
        if origin.strip()
    ]
    if "*" in origins:
        raise ValueError(
            "Wildcard CORS is disabled; set explicit frontend origins in "
            "ENDO_CORS_ORIGINS"
        )
    return origins


def cors_methods() -> list[str]:
    return ["GET", "POST", "PUT", "OPTIONS"]


def cors_headers() -> list[str]:
    return ["Accept", "Content-Type", "Range"]
