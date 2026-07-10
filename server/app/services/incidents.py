from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from textwrap import wrap
from typing import Any
import json
import uuid

import cv2
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.core.config import settings
from app.state import state


class IncidentService:
    def __init__(self) -> None:
        self._incident_store_path = settings.data_dir / 'incident_reports.json'
        self._snapshots_dir = settings.data_dir / 'snapshots'
        self._incident_store_path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.load_incidents()

    def list_incidents(self) -> list[dict[str, Any]]:
        with state.lock:
            return list(state.incidents.values())

    def create_incident(self, payload: dict[str, Any], frame: Any | None = None) -> dict[str, Any]:
        incident_id = str(payload.get('id') or uuid.uuid4().hex[:8])
        timestamp = str(payload.get('timestamp') or datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        camera_id = str(payload.get('camera_id') or '')
        person_id = payload.get('person_id')
        confidence = float(payload.get('confidence') or 0.0)
        camera_name, stream_name = self._resolve_camera_metadata(camera_id, payload)

        snapshot_file = payload.get('snapshot_file')
        if snapshot_file is None and frame is not None:
            snapshot_file = self._save_snapshot(incident_id, timestamp, frame)

        incident = {
            'id': incident_id,
            'fall_id': str(
                payload.get('fall_id')
                or (
                    f'{camera_id}_person{person_id}_{int(datetime.now().timestamp())}'
                    if person_id is not None
                    else f'{camera_id}_{int(datetime.now().timestamp())}'
                )
            ),
            'timestamp': timestamp,
            'camera_name': camera_name,
            'camera_id': camera_id,
            'stream_name': stream_name,
            'person_id': person_id,
            'person_label': str(payload.get('person_label') or (f'Person #{person_id}' if person_id is not None else 'Detected Person')),
            'confidence': confidence,
            'severity': str(payload.get('severity') or self._severity_for(confidence)),
            'snapshot_file': snapshot_file,
            'notes': str(payload.get('notes') or ''),
            'location': str(stream_name or camera_name),
        }

        with state.lock:
            state.incidents[incident_id] = incident

        self.save_incidents()
        self.generate_pdf(incident)
        return incident

    def pdf_path_for(self, incident_id: str) -> Path:
        return settings.data_dir / f"{incident_id}.pdf"

    def ensure_pdf(self, incident_id: str) -> Path | None:
        incident = self.get_incident(incident_id)
        if incident is None:
            return None

        pdf_path = self.pdf_path_for(incident_id)
        if not pdf_path.exists():
            return self.generate_pdf(incident)

        return pdf_path

    def generate_pdf(self, incident: dict[str, Any]) -> Path:
        pdf_path = self.pdf_path_for(str(incident['id']))
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        pdf.setFont('Helvetica-Bold', 24)
        pdf.drawString(50, height - 50, 'FALL DETECTION INCIDENT REPORT')

        pdf.setStrokeColor(colors.HexColor('#1f2937'))
        pdf.setLineWidth(2)
        pdf.line(50, height - 60, width - 50, height - 60)

        pdf.setFont('Helvetica', 10)
        pdf.setFillColor(colors.HexColor('#6b7280'))
        pdf.drawString(50, height - 75, f"FallGuard AI Fall Detection System - Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        y_position = height - 110

        pdf.setFillColor(colors.HexColor('#f3f4f6'))
        pdf.rect(50, y_position - 60, width - 100, 50, fill=1, stroke=1)

        pdf.setFont('Helvetica-Bold', 12)
        pdf.setFillColor(colors.HexColor('#1f2937'))
        pdf.drawString(60, y_position - 20, f"Incident ID: {incident['id']}")
        pdf.drawString(60, y_position - 35, f"Severity Level: {incident.get('severity', 'LOW')}")
        pdf.drawString(350, y_position - 20, f"Timestamp: {incident.get('timestamp', '')}")
        pdf.drawString(350, y_position - 35, f"Camera: {incident.get('camera_name', '')}")

        y_position -= 80

        pdf.setFont('Helvetica-Bold', 14)
        pdf.setFillColor(colors.HexColor('#1f2937'))
        pdf.drawString(50, y_position, 'INCIDENT DETAILS')

        pdf.setLineWidth(1)
        pdf.setStrokeColor(colors.HexColor('#d1d5db'))
        pdf.line(50, y_position - 5, width - 50, y_position - 5)

        y_position -= 25

        details = [
            ('Location:', self._clean_location(incident)),
            ('Confidence Score:', f"{float(incident.get('confidence', 0.0)) * 100:.1f}%"),
            ('Severity:', incident.get('severity', 'LOW')),
            ('Detection Status:', 'Fall Detected - High Priority'),
        ]

        for label, value in details:
            pdf.setFont('Helvetica-Bold', 11)
            pdf.setFillColor(colors.HexColor('#1f2937'))
            pdf.drawString(60, y_position, label)

            pdf.setFont('Helvetica', 11)
            pdf.setFillColor(colors.HexColor('#374151'))
            pdf.drawString(200, y_position, str(value))
            y_position -= 20

        y_position -= 10

        snapshot_file = incident.get('snapshot_file')
        if snapshot_file:
            snapshot_path = self._snapshots_dir / str(snapshot_file)
            if snapshot_path.exists():
                try:
                    pdf.setFont('Helvetica-Bold', 14)
                    pdf.setFillColor(colors.HexColor('#1f2937'))
                    pdf.drawString(50, y_position, 'INCIDENT SNAPSHOT')

                    pdf.setLineWidth(1)
                    pdf.setStrokeColor(colors.HexColor('#d1d5db'))
                    pdf.line(50, y_position - 5, width - 50, y_position - 5)

                    y_position -= 20
                    img = ImageReader(str(snapshot_path))
                    img_width = 300
                    img_height = 225
                    pdf.setStrokeColor(colors.HexColor('#d1d5db'))
                    pdf.setLineWidth(1)
                    pdf.rect(60, y_position - img_height - 10, img_width, img_height, stroke=1)
                    pdf.drawImage(img, 62, y_position - img_height - 8, width=img_width - 4, height=img_height - 4)
                    y_position -= img_height - 10
                except Exception as exc:
                    pdf.setFont('Helvetica', 10)
                    pdf.setFillColor(colors.HexColor('#dc2626'))
                    pdf.drawString(60, y_position, f'Error loading snapshot: {exc}')

        y_position -= 30

        notes = str(incident.get('notes') or '').strip()
        if notes:
            pdf.setFont('Helvetica-Bold', 14)
            pdf.setFillColor(colors.HexColor('#1f2937'))
            pdf.drawString(50, y_position, 'ADDITIONAL NOTES')

            pdf.setLineWidth(1)
            pdf.setStrokeColor(colors.HexColor('#d1d5db'))
            pdf.line(50, y_position - 5, width - 50, y_position - 5)

            y_position -= 20
            pdf.setFont('Helvetica', 10)
            pdf.setFillColor(colors.HexColor('#374151'))
            for line in wrap(notes, width=100):
                pdf.drawString(60, y_position, line)
                y_position -= 15

        pdf.setFont('Helvetica', 9)
        pdf.setFillColor(colors.HexColor('#9ca3af'))
        pdf.drawString(50, 30, f"This is an official FallGuard incident report. Report ID: {incident['id']}")
        pdf.drawString(50, 15, 'For more information, visit the FallGuard Admin Panel')

        pdf.save()
        buffer.seek(0)
        pdf_path.write_bytes(buffer.getvalue())
        return pdf_path

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        with state.lock:
            return state.incidents.get(incident_id)

    def update_notes(self, incident_id: str, notes: str) -> dict[str, Any] | None:
        with state.lock:
            incident = state.incidents.get(incident_id)
            if incident is None:
                return None
            incident['notes'] = notes

        self.save_incidents()
        self.generate_pdf(incident)
        return incident

    def delete_incident(self, incident_id: str) -> bool:
        with state.lock:
            deleted = state.incidents.pop(incident_id, None) is not None

        if deleted:
            self.save_incidents()
            self.pdf_path_for(incident_id).unlink(missing_ok=True)

        return deleted

    def load_incidents(self) -> None:
        if not self._incident_store_path.exists():
            return

        try:
            payload = json.loads(self._incident_store_path.read_text(encoding='utf-8'))
        except Exception:
            return

        incidents: dict[str, dict[str, Any]] = {}
        records = payload.values() if isinstance(payload, dict) else payload
        for record in records:
            if not isinstance(record, dict):
                continue

            incident_id = str(record.get('id') or record.get('incident_id') or uuid.uuid4().hex[:8])
            normalized = dict(record)
            normalized['id'] = incident_id
            incidents[incident_id] = normalized

        with state.lock:
            state.incidents = incidents

    def save_incidents(self) -> None:
        with state.lock:
            records = list(state.incidents.values())

        self._incident_store_path.write_text(json.dumps(records, indent=2), encoding='utf-8')

    @staticmethod
    def _severity_for(confidence: float) -> str:
        if confidence > 0.8:
            return 'HIGH'
        if confidence > 0.6:
            return 'MEDIUM'
        return 'LOW'

    def _resolve_camera_metadata(self, camera_id: str, payload: dict[str, Any]) -> tuple[str, str]:
        camera_name = str(payload.get('camera_name') or camera_id or 'Unknown Camera')
        stream_name = str(payload.get('stream_name') or payload.get('location') or '')

        with state.lock:
            camera_record = state.cameras.get(camera_id)

        if isinstance(camera_record, dict):
            resolved_name = camera_record.get('name')
            resolved_stream_name = camera_record.get('stream_name')
            resolved_source = camera_record.get('source')

            if resolved_name:
                camera_name = str(resolved_name)

            if resolved_stream_name:
                stream_name = str(resolved_stream_name)
            elif resolved_source:
                stream_name = str(resolved_source)

        if not stream_name:
            stream_name = camera_name

        return camera_name, stream_name

    def _clean_location(self, incident: dict[str, Any]) -> str:
        location = str(incident.get('location') or incident.get('stream_name') or '').strip()
        if not location:
            return str(incident.get('camera_name') or 'Unknown Camera')

        for suffix in (' - Person #1', ' - Person #2', ' - Person #3', ' - Person #4', ' - Person #5'):
            if location.endswith(suffix):
                return location[: -len(suffix)].strip()

        if 'Person #' in location:
            return location.split('Person #', 1)[0].rstrip(' -').strip()

        return location

    def _save_snapshot(self, incident_id: str, timestamp: str, frame: Any) -> str | None:
        try:
            snapshot_name = f"{incident_id}_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
            snapshot_path = self._snapshots_dir / snapshot_name
            success = cv2.imwrite(str(snapshot_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            return snapshot_name if success else None
        except Exception:
            return None