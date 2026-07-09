from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

from fastapi import UploadFile

from app.core.config import settings
from app.state import state


class StorageService:
    async def save_upload(self, upload_file: UploadFile) -> dict[str, Any]:
        upload_id = uuid.uuid4().hex
        suffix = Path(upload_file.filename or "video.mp4").suffix or ".mp4"
        saved_name = f"{upload_id}{suffix}"
        saved_path = settings.uploads_dir / saved_name

        contents = await upload_file.read()
        saved_path.write_bytes(contents)

        record = {
            "upload_id": upload_id,
            "filename": upload_file.filename or saved_name,
            "path": str(saved_path),
            "size": len(contents),
        }

        with state.lock:
            state.uploads[upload_id] = record

        return record

    def list_uploads(self) -> list[dict[str, Any]]:
        with state.lock:
            return list(state.uploads.values())

    def delete_upload(self, upload_id: str) -> bool:
        with state.lock:
            record = state.uploads.pop(upload_id, None)

        if record is None:
            return False

        Path(str(record.get("path", ""))).unlink(missing_ok=True)
        return True