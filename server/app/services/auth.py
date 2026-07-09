from __future__ import annotations

from app.core.config import settings


class AuthService:
    def __init__(self) -> None:
        self._authenticated = False

    def login(self, password: str) -> bool:
        self._authenticated = password == settings.admin_password
        return self._authenticated

    def logout(self) -> None:
        self._authenticated = False

    def is_authenticated(self) -> bool:
        return self._authenticated