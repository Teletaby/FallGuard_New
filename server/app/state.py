from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class AppState:
    uploads: dict[str, dict[str, Any]] = field(default_factory=dict)
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)
    frame_sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    incidents: dict[str, dict[str, Any]] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)


state = AppState()