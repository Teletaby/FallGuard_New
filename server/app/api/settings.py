from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.state import state


router = APIRouter(prefix='/api')


class SettingsUpdateRequest(BaseModel):
    fall_threshold: float | None = Field(default=None)
    fall_delay_seconds: int | None = Field(default=None)
    privacy_mode: str | None = Field(default=None)
    pre_fall_buffer_seconds: int | None = Field(default=None)
    hide_overlays: bool | None = Field(default=None)


def _snapshot_settings() -> dict[str, object]:
    with state.lock:
        return dict(state.detection_settings)


@router.get('/settings')
def get_settings() -> dict[str, object]:
    return {
        'success': True,
        'settings': _snapshot_settings(),
        'telegram_token': False,
        'telegram_bot_name': '',
    }


@router.post('/settings')
def update_settings(payload: SettingsUpdateRequest) -> dict[str, object]:
    updates: dict[str, object] = {}

    if payload.fall_threshold is not None:
        updates['fall_threshold'] = max(0.0, min(1.0, float(payload.fall_threshold)))

    if payload.fall_delay_seconds is not None:
        updates['fall_delay_seconds'] = max(1, int(payload.fall_delay_seconds))

    if payload.privacy_mode is not None:
        updates['privacy_mode'] = str(payload.privacy_mode)

    if payload.pre_fall_buffer_seconds is not None:
        updates['pre_fall_buffer_seconds'] = max(1, int(payload.pre_fall_buffer_seconds))

    if payload.hide_overlays is not None:
        updates['hide_overlays'] = bool(payload.hide_overlays)

    with state.lock:
        state.detection_settings.update(updates)
        current = dict(state.detection_settings)

    return {
        'success': True,
        'settings': current,
    }