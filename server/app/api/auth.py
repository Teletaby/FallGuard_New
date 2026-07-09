from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import auth_service


router = APIRouter(prefix="/api/admin")


class LoginRequest(BaseModel):
    password: str


@router.get("/check")
def admin_check() -> dict[str, bool]:
    return {"authenticated": auth_service.is_authenticated()}


@router.post("/login")
def admin_login(payload: LoginRequest) -> dict[str, bool]:
    if not auth_service.login(payload.password):
        raise HTTPException(status_code=401, detail="invalid password")
    return {"authenticated": True}


@router.post("/logout")
def admin_logout() -> dict[str, bool]:
    auth_service.logout()
    return {"authenticated": False}