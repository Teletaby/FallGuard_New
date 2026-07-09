from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services import camera_service, storage_service


router = APIRouter(prefix="/api/cameras")


@router.get("")
def get_cameras() -> dict[str, object]:
    return {"cameras": camera_service.list_cameras()}


@router.get("/all_definitions")
def get_all_definitions() -> dict[str, object]:
    return {"cameras": camera_service.list_camera_definitions()}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> dict[str, object]:
    upload_record = await storage_service.save_upload(file)
    camera_record = camera_service.register_uploaded_video(upload_record)
    return {"upload": upload_record, "camera": camera_record}


@router.post("/add")
def add_camera() -> dict[str, str]:
    raise HTTPException(status_code=501, detail="camera creation is not wired yet")


@router.post("/stop/{camera_id}")
def stop_camera(camera_id: str) -> dict[str, str]:
    raise HTTPException(status_code=501, detail=f"stop is not wired for camera {camera_id}")


@router.delete("/remove/{camera_id}")
def remove_camera(camera_id: str) -> dict[str, str]:
    raise HTTPException(status_code=501, detail=f"remove is not wired for camera {camera_id}")


@router.post("/add_existing")
def add_existing_camera() -> dict[str, str]:
    raise HTTPException(status_code=501, detail="existing camera registration is not wired yet")