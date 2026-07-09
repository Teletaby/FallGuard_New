from __future__ import annotations

from typing import Any

from app.state import state


class CameraService:
    def list_cameras(self) -> list[dict[str, Any]]:
        with state.lock:
            return list(state.cameras.values())

    def list_camera_definitions(self) -> list[dict[str, Any]]:
        with state.lock:
            return list(state.cameras.values())

    def register_uploaded_video(self, upload_record: dict[str, Any]) -> dict[str, Any]:
        camera_id = upload_record["upload_id"]
        camera_record = {
            "id": camera_id,
            "name": upload_record["filename"],
            "source": upload_record["path"],
            "isLive": False,
            "status": "ready",
        }

        with state.lock:
            state.cameras[camera_id] = camera_record

        return camera_record
