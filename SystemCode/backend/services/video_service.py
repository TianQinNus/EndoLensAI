from pathlib import Path

from ..video import decoded, probe, schedule


class VideoService:
    """Owns video probing, PTS decoding, and sampling decisions."""

    probe = staticmethod(probe)
    decoded = staticmethod(decoded)
    schedule = staticmethod(schedule)

    @staticmethod
    def filename(value: str | None) -> str:
        return Path((value or "upload.mp4").replace("\\", "/")).name[:120]
