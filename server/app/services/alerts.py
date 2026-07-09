from __future__ import annotations

import time
from typing import Any
import uuid

from app.state import state


class AlertService:
    def list_active_alerts(self) -> list[dict[str, Any]]:
        cutoff = time.time() - 35

        with state.lock:
            state.alerts = [alert for alert in state.alerts if alert["timestamp"] >= cutoff]
            return list(state.alerts)

    def record_alert(self, camera_id: str, camera_name: str, confidence: float, person_id: int | None = None) -> dict[str, Any]:
        alert = {
            "alert_id": uuid.uuid4().hex,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "confidence": float(confidence),
            "person_id": person_id,
            "timestamp": time.time(),
        }

        with state.lock:
            state.alerts.append(alert)

        return alert