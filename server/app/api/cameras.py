from __future__ import annotations

import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.services import camera_service, frame_loop_service, storage_service
from app.state import state


router = APIRouter(prefix="/api/cameras")


@router.get("")
def get_cameras() -> dict[str, object]:
    return {"cameras": camera_service.list_cameras()}


@router.get("/all_definitions")
def get_all_definitions() -> dict[str, object]:
    return {"cameras": camera_service.list_camera_definitions()}


@router.post("/upload")
async def upload_video(
    file: UploadFile | None = File(default=None),
    video_file: UploadFile | None = File(default=None),
    name: str = Form(default=""),
) -> dict[str, object]:
    upload_file = video_file or file
    if upload_file is None:
        raise HTTPException(status_code=400, detail="video file is required")

    upload_record = await storage_service.save_upload(upload_file)
    camera_record = camera_service.register_uploaded_video(upload_record, display_name=name.strip() or None)
    return {"upload": upload_record, "camera": camera_record, "camera_id": camera_record["id"]}


@router.get("/{camera_id}/snapshot")
def get_camera_snapshot(camera_id: str) -> Response:
    latest_frame = frame_loop_service.stream_frame(camera_id)

    if latest_frame:
        return Response(content=latest_frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return Response(content=_placeholder_snapshot(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/{camera_id}/stream.mjpeg")
def get_camera_mjpeg_stream(camera_id: str, fps: float = 24.0) -> StreamingResponse:
    preview_fps = max(1.0, min(fps, 30.0))

    def generate():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"

        while frame_loop_service.is_active(camera_id):
            frame = frame_loop_service.stream_frame(camera_id)
            if frame:
                yield boundary + frame + b"\r\n"
            time.sleep(1.0 / preview_fps)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/add")
def add_camera() -> dict[str, str]:
    raise HTTPException(status_code=501, detail="camera creation is not wired yet")


@router.post("/stop/{camera_id}")
def stop_camera(camera_id: str) -> dict[str, str]:
    with state.lock:
        camera = state.cameras.get(camera_id)
        if camera is None:
            raise HTTPException(status_code=404, detail=f"camera {camera_id} not found")

        camera["isLive"] = False
        camera["status"] = "Offline"
        camera["color"] = "gray"
        camera["fps"] = 0

    frame_loop_service.stop(camera_id)

    with state.lock:
        frame_source = state.frame_sources.get(camera_id)
        if frame_source is not None:
            frame_source["status"] = "stopped"

    return {"message": f"camera {camera_id} stopped"}


@router.delete("/remove/{camera_id}")
def remove_camera(camera_id: str) -> dict[str, str]:
    frame_loop_service.stop(camera_id)

    deleted_upload = storage_service.delete_upload(camera_id)

    with state.lock:
        state.cameras.pop(camera_id, None)
        state.frame_sources.pop(camera_id, None)

    if deleted_upload:
        return {"message": f"camera {camera_id} removed and upload deleted"}

    return {"message": f"camera {camera_id} removed"}


@router.post("/add_existing")
def add_existing_camera() -> dict[str, str]:
    with state.lock:
        camera_ids = list(state.cameras.keys())

    if not camera_ids:
        raise HTTPException(status_code=404, detail="no cameras available to restart")

    camera_id = camera_ids[0]
    with state.lock:
        camera = state.cameras.get(camera_id)
        if camera is None:
            raise HTTPException(status_code=404, detail=f"camera {camera_id} not found")

        camera["isLive"] = True
        camera["status"] = "Looping" if camera.get("source_kind") == "video" else "Monitoring"
        camera["color"] = "green"
        camera["fps"] = camera.get("fps") or 24

    if camera.get("source_kind") == "video":
        frame_loop_service.start(camera)

    return {"message": f"camera {camera_id} restarted"}


def _placeholder_snapshot() -> bytes:
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    canvas[:] = (17, 24, 39)
    cv2.rectangle(canvas, (24, 24), (616, 336), (51, 65, 85), 3)
    cv2.putText(canvas, "Waiting for frame", (170, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (248, 250, 252), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Backend snapshot stream", (150, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (148, 163, 184), 2, cv2.LINE_AA)
    success, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return encoded.tobytes() if success else b""