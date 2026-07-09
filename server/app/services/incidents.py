from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

from app.core.config import settings
from app.state import state


class IncidentService:
    def list_incidents(self) -> list[dict[str, Any]]:
        with state.lock:
            return list(state.incidents.values())

    def create_incident(self, payload: dict[str, Any]) -> dict[str, Any]:
        incident_id = uuid.uuid4().hex
        incident = {"incident_id": incident_id, **payload}

        with state.lock:
            state.incidents[incident_id] = incident

        return incident

    def pdf_path_for(self, incident_id: str) -> Path:
        return settings.data_dir / f"{incident_id}.pdf"

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        with state.lock:
            return state.incidents.get(incident_id)

    def update_notes(self, incident_id: str, notes: str) -> dict[str, Any] | None:
        with state.lock:
            incident = state.incidents.get(incident_id)
            if incident is None:
                return None
            incident["notes"] = notes
            return incident

    def delete_incident(self, incident_id: str) -> bool:
        with state.lock:
            return state.incidents.pop(incident_id, None) is not None