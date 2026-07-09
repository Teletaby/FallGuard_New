from fastapi import APIRouter

from app.services import alert_service


router = APIRouter(prefix="/api/alerts")


@router.get("/active")
def get_active_alerts() -> dict[str, object]:
    return {"success": True, "alerts": alert_service.list_active_alerts()}