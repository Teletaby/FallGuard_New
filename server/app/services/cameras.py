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

    def register_uploaded_video(self, upload_record: dict[str, Any], display_name: str | None = None) -> dict[str, Any]:
        camera_id = upload_record["upload_id"]
        with state.lock:
            video_camera_count = sum(1 for camera in state.cameras.values() if camera.get("source_kind") == "video")

        camera_record = {
            "id": camera_id,
            "name": f"Camera {video_camera_count + 1}",
            "stream_name": display_name or upload_record["filename"],
            "source": upload_record["path"],
            "source_kind": "video",
            "isLive": True,
            "status": "Looping",
            "playback_mode": "loop",
            "snapshot_url": f"/api/cameras/{camera_id}/snapshot",
            "stream_url": f"/api/cameras/{camera_id}/stream.mjpeg?fps=24",
        }

        with state.lock:
            state.cameras[camera_id] = camera_record

        from app.services import frame_loop_service

        frame_loop_service.start(camera_record)

        return camera_record
