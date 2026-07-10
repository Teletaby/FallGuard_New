from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any


from app.state import state


class AlertService:
    def __init__(self) -> None:
        self._subscriber_lock = asyncio.Lock()
        self._subscribers: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]] = {}

    def list_active_alerts(self) -> list[dict[str, Any]]:
        cutoff = time.time() - 35

        with state.lock:
            state.alerts = [alert for alert in state.alerts if alert["timestamp"] >= cutoff]
            return list(state.alerts)

    async def subscribe(self) -> tuple[str, asyncio.Queue[dict[str, Any]]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        subscriber_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()

        async with self._subscriber_lock:
            self._subscribers[subscriber_id] = (loop, queue)

        return subscriber_id, queue

    async def unsubscribe(self, subscriber_id: str) -> None:
        async with self._subscriber_lock:
            self._subscribers.pop(subscriber_id, None)

    def _broadcast(self, alert: dict[str, Any]) -> None:
        subscribers = list(self._subscribers.items())

        for subscriber_id, (loop, queue) in subscribers:
            if loop.is_closed():
                continue

            try:
                loop.call_soon_threadsafe(queue.put_nowait, alert)
            except RuntimeError:
                loop.call_soon_threadsafe(self._subscribers.pop, subscriber_id, None)

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

        self._broadcast(alert)
        return alert

    @staticmethod
    def format_sse_event(event_type: str, payload: dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"