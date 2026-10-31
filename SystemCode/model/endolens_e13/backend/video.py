"""Sequential PTS-based MP4/H264 decoding, never frame_index/fps timestamps."""
import math
from pathlib import Path
import av

MAX_PIXELS = 1920*1080
MAX_DURATION = 900.
MAX_DECODED_FRAMES = 60_000
MAX_SAMPLED_FRAMES = 600


def validate_frame(frame):
    if frame.width*frame.height > MAX_PIXELS or min(frame.width, frame.height) < 1:
        raise ValueError("Only frames up to1080p supported in this starter")
    if frame.is_corrupt:
        raise ValueError("Corrupt decoded frame")
    if any("DISPLAYMATRIX" in str(side.type) for side in frame.side_data):
        raise ValueError("Rotation/display matrix unsupported; transcode to upright square-pixel H264 MP4")


def open_video(path):
    container = av.open(str(Path(path)), options={"protocol_whitelist":"file","enable_drefs":"0"})
    if "mov" not in container.format.name:
        container.close();raise ValueError("Require an actual MP4/MOV-family container, not a renamed file")
    if not container.streams.video:
        container.close(); raise ValueError("No video stream")
    stream = container.streams.video[0]
    if stream.codec_context.name != "h264" or str(stream.metadata.get("rotate", "0")) not in ("0", "0.0"):
        container.close(); raise ValueError("Starter accepts upright H264 MP4 only")
    if stream.sample_aspect_ratio is not None and float(stream.sample_aspect_ratio) != 1.:
        container.close(); raise ValueError("Non-square pixels unsupported")
    return container, stream


def probe(path):
    container, stream = open_video(path)
    try:
        duration = container.duration/av.time_base if container.duration else (float(stream.duration*stream.time_base) if stream.duration is not None else None)
        if duration is None or not math.isfinite(duration) or not 0 < duration <= MAX_DURATION:
            raise ValueError("Missing duration or duration exceeds15minutes")
        frame = next(container.decode(stream), None)
        if frame is None:
            raise ValueError("No decodable video frames")
        validate_frame(frame)
        if frame.pts is None:
            raise ValueError("Presentation timestamps required")
        return {"width":frame.width,"height":frame.height,"duration_seconds":duration,
                "fps":float(stream.average_rate) if stream.average_rate else None,
                "total_frames":stream.frames or None, "codec":"h264",
                "timestamp_origin_seconds":container.start_time/av.time_base if container.start_time is not None else float(frame.pts*frame.time_base)}
    finally:
        container.close()


def decoded(path, origin):
    container, stream = open_video(path)
    previous = None
    try:
        for index, frame in enumerate(container.decode(stream)):
            validate_frame(frame)
            if index >= MAX_DECODED_FRAMES or frame.pts is None:
                raise ValueError("Decode limit exceeded or missing PTS")
            seconds = float(frame.pts*frame.time_base)-origin
            if not math.isfinite(seconds) or seconds < -1e-6 or seconds > MAX_DURATION or previous is not None and seconds <= previous:
                raise ValueError("Invalid/non-increasing video PTS")
            previous = seconds
            yield index, max(0., seconds), frame
    finally:
        container.close()


def schedule(path, origin, mode, interval, cancelled=lambda:False):
    output, next_time = [], 0.
    for index, seconds, _ in decoded(path, origin):
        if cancelled():
            return None
        if mode == "every_frame" or seconds+1e-9 >= next_time:
            output.append({"frame_index":index,"timestamp_seconds":seconds})
            if mode == "interval":
                next_time = (math.floor((seconds+1e-9)/interval)+1)*interval
            if len(output) > MAX_SAMPLED_FRAMES:
                raise ValueError("More than600 sampled frames; use a longer sampling interval or shorter video")
    return output
