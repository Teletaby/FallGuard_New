from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services import incident_service


router = APIRouter(prefix="/api/incidents")


class NotesRequest(BaseModel):
    notes: str


@router.get("")
def list_incidents() -> dict[str, object]:
    return {"incidents": incident_service.list_incidents()}


@router.get("/{incident_id}/pdf")
def generate_incident_pdf(incident_id: str):
    pdf_path = incident_service.ensure_pdf(incident_id)
    if pdf_path is None or not pdf_path.exists():
        raise HTTPException(status_code=404, detail="incident pdf not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)


@router.post("/{incident_id}/notes")
def update_incident_notes(incident_id: str, payload: NotesRequest) -> dict[str, object]:
    incident = incident_service.update_notes(incident_id, payload.notes)
    if incident is None:
        raise HTTPException(status_code=404, detail="incident not found")
    return {"incident": incident}


@router.delete("/{incident_id}")
def delete_incident(incident_id: str) -> dict[str, bool]:
    if not incident_service.delete_incident(incident_id):
        raise HTTPException(status_code=404, detail="incident not found")
    return {"deleted": True}