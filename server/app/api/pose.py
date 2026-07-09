from fastapi import APIRouter

from app.services import pose_service


router = APIRouter(prefix="/api/pose")


@router.get("/status")
def pose_status() -> dict[str, str]:
    return pose_service.snapshot()