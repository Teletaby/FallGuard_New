from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "FallGuard API"
    api_prefix: str = "/api"
    base_dir: Path = Path(__file__).resolve().parents[2]
    admin_password: str = "admin"
    cors_origins: tuple[str, ...] = ("*",)

    @property
    def uploads_dir(self) -> Path:
        return self.base_dir / "uploads"

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    def ensure_directories(self) -> None:
        for path in (self.uploads_dir, self.data_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = AppConfig()