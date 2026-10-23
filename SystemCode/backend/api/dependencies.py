import io
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, Request
from PIL import Image

from .runtime import RuntimeProtocol


def get_runtime(request: Request) -> RuntimeProtocol:
    return request.app.state.runtime


def get_record(runtime: RuntimeProtocol, kind: str, key: str) -> dict[str, Any]:
    try:
        return runtime.store.get(kind, key)
    except KeyError as error:
        raise HTTPException(404, "Record not found") from error


def get_frame_rows(runtime: RuntimeProtocol, task_id: str) -> list[dict[str, Any]]:
    get_record(runtime, "analysis", task_id)
    return runtime.store.list("frame", task_id)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identity() -> str:
    return uuid.uuid4().hex


def image_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()
