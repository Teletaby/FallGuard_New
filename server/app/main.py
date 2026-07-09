from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.alerts import router as alerts_router
from app.api.auth import router as auth_router
from app.api.cameras import router as cameras_router
from app.api.health import router as health_router
from app.api.incidents import router as incidents_router
from app.api.pose import router as pose_router
from app.services import pose_service
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    pose_service.prime()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(alerts_router)
app.include_router(cameras_router)
app.include_router(incidents_router)
app.include_router(pose_router)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": settings.app_name,
        "mode": "modular",
        "routes": [
            "/health",
            "/api/admin/check",
            "/api/alerts/active",
            "/api/cameras",
            "/api/incidents",
            "/api/pose/status",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)