from __future__ import annotations


class PoseService:
    def __init__(self) -> None:
        self._status = "idle"
        self._backend = "benchmark-style modular scaffold"

    def snapshot(self) -> dict[str, str]:
        return {
            "status": self._status,
            "backend": self._backend,
        }
