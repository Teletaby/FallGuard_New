# Feature Inventory and Cleanup Notes

This note captures the current code and feature structure before cleanup. The frontend should remain as-is for now.

## 1. Current App Server

Primary backend entrypoint: [server/main.py](server/main.py)

Language and runtime:
- Python
- Flask web server
- PyTorch for LSTM inference
- OpenCV and MediaPipe for video and pose processing
- ReportLab for PDF generation
- Requests for Telegram and HTTP calls

Support files and modules:
- [server/app/video_utils.py](server/app/video_utils.py)
- [server/app/skeleton_lstm.py](server/app/skeleton_lstm.py)
- [server/app/fall_logic.py](server/app/fall_logic.py)
- [server/run_server.py](server/run_server.py)
- [server/utils/dataset_loader.py](server/utils/dataset_loader.py)

Main feature groups in the current app server:
- Admin login and session checks
- Global settings for detection and privacy
- Telegram token, subscriber, blocked list, and test alert management
- Camera lifecycle management for add, stop, remove, restart, and upload
- Live MJPEG video feeds and snapshots
- Incident report listing, PDF export, notes, and delete
- Website alerts polling and fall event tracking
- Model loading and frame processing for fall detection

Important runtime data and assets:
- [server/models/](server/models)
- [server/data/](server/data)
- [server/uploads/](server/uploads)
- [server/app/static/](server/app/static)
- [server/app/index.html](server/app/index.html)
- [server/app/login.html](server/app/login.html)
- [server/app/debug.html](server/app/debug.html)

## 2. Benchmark Server

Benchmark backend entrypoints:
- [benchmark/backend/app/main.py](benchmark/backend/app/main.py)
- [benchmark/backend_mediapipe/app/main.py](benchmark/backend_mediapipe/app/main.py)

Benchmark inference modules:
- [benchmark/backend/app/benchmark.py](benchmark/backend/app/benchmark.py)
- [benchmark/backend_mediapipe/app/benchmark.py](benchmark/backend_mediapipe/app/benchmark.py)

Benchmark structure:
- YOLO backend and MediaPipe backend are split into separate FastAPI apps
- Session handling is isolated in a BenchmarkSession object
- Uploads, session state, live MJPEG streaming, and websocket notifications are kept small and explicit
- The backend serves a simple host page and a separate read-only viewer client

Benchmark frontend:
- [benchmark/frontend/src/main.ts](benchmark/frontend/src/main.ts)
- [benchmark/frontend/package.json](benchmark/frontend/package.json)

Benchmark assets:
- [benchmark/weights/](benchmark/weights)
- [benchmark/uploads/](benchmark/uploads)

## 3. Frontend Contract To Preserve

The existing frontend should stay unchanged while we clean up the backend.

Client entrypoints:
- [client/src/App.tsx](client/src/App.tsx)
- [client/src/main.tsx](client/src/main.tsx)
- [client/vite.config.ts](client/vite.config.ts)

Client pages currently depending on the app server:
- [client/src/pages/LoginPage.tsx](client/src/pages/LoginPage.tsx)
- [client/src/pages/DashboardPage.tsx](client/src/pages/DashboardPage.tsx)
- [client/src/pages/DebugPage.tsx](client/src/pages/DebugPage.tsx)

Backend endpoints the client currently expects:
- /api/admin/check
- /api/admin/login
- /api/admin/logout
- /api/settings
- /api/alerts/active
- /api/telegram/set_token
- /api/telegram/subscribers
- /api/telegram/add_subscriber
- /api/telegram/remove_subscriber
- /api/telegram/test_alert
- /api/telegram/blocked
- /api/telegram/unblock
- /api/cameras
- /api/cameras/all_definitions
- /api/cameras/add
- /api/cameras/stop/<camera_id>
- /api/cameras/remove/<camera_id>
- /api/cameras/add_existing
- /api/cameras/upload
- /api/incidents
- /api/incidents/<incident_id>/pdf
- /api/incidents/<incident_id>/notes
- /api/incidents/<incident_id>
- /video_feed/<camera_id>
- /snapshot/<camera_id>

## 4. Cleanup Direction

Keep:
- The current frontend routes and UI flow
- The endpoint contract above
- The app server behavior that feeds the frontend
- The current incident PDF layout and formatting

Refactor later:
- Split server/main.py into smaller modules by feature area
- Separate inference, camera management, Telegram, incidents, and admin auth
- Remove or quarantine legacy HTML if the React client fully replaces it
- Remove unused scripts, tests, and duplicate benchmark-only assets only after confirming they are not needed

Prefer the benchmark structure as the cleanup model:
- FastAPI-style separation of host page, API, and inference session logic
- Smaller session object boundaries
- Clear distinction between backend runtime and frontend viewer
