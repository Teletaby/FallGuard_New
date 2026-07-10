import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.services import alert_service


router = APIRouter(prefix="/api/alerts")


@router.get("/active")
def get_active_alerts() -> dict[str, object]:
    return {"success": True, "alerts": alert_service.list_active_alerts()}


@router.get("/stream")
async def stream_alerts(request: Request) -> StreamingResponse:
    subscriber_id, queue = await alert_service.subscribe()

    async def event_stream():
        try:
            active_alerts = alert_service.list_active_alerts()
            yield alert_service.format_sse_event("snapshot", {"alerts": active_alerts})

            while True:
                if await request.is_disconnected():
                    break

                try:
                    alert = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                yield alert_service.format_sse_event("alert", alert)
        finally:
            await alert_service.unsubscribe(subscriber_id)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})