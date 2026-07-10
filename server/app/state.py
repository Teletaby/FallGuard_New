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
    detection_settings: dict[str, Any] = field(default_factory=lambda: {
        'fall_threshold': 0.8,
        'fall_delay_seconds': 2,
        'privacy_mode': 'full_video',
        'pre_fall_buffer_seconds': 5,
        'hide_overlays': True,
    })
    lock: Lock = field(default_factory=Lock)


state = AppState()