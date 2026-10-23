from typing import Any, Protocol


class StoreProtocol(Protocol):
    lock: Any

    def get(self, kind: str, identity: str) -> dict[str, Any]: ...
    def list(self, kind: str, task: str | None = None) -> list[dict[str, Any]]: ...
    def put(self, kind: str, identity: str, data: dict[str, Any], task: str | None = None, idx: int | None = None) -> None: ...
    def update(self, identity: str, **updates: Any) -> dict[str, Any]: ...
    def path(self, relative: str): ...
    def write(self, relative: str, data: bytes): ...


class RuntimeProtocol(Protocol):
    store: StoreProtocol
    adapter: Any
    area_gate: float
    inference: Any
    jobs: Any
