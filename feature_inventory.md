# Feature Inventory and Cleanup Notes

This note captures the current code and feature structure before cleanup. The frontend should remain as-is for now.

## 1. Current App Server

Primary backend entrypoint: [server/app/main.py](server/app/main.py)

Language and runtime:
- Python
- FastAPI web server with Uvicorn
- PyTorch for fall-detection LSTM inference
- Ultralytics YOLO/OpenVINO for pose extraction
- OpenCV for video and image handling
- ReportLab for PDF generation

Support files and modules:
- [server/app/api/](server/app/api/)
- [server/app/services/](server/app/services/)
- [server/app/core/config.py](server/app/core/config.py)
- [server/app/state.py](server/app/state.py)
- [server/run_server.py](server/run_server.py)

Main feature groups in the current app server:
- Admin login and session checks
- Global settings for detection and privacy
- Camera lifecycle management for add, stop, remove, restart, and upload
- Live MJPEG video feeds and snapshots
- Incident report listing, PDF export, notes, and delete
- Website alerts polling, streaming, and fall event tracking
- Model loading and frame processing for fall detection

Important runtime data and assets:
- [server/models/](server/models)
- [server/data/](server/data)
- [server/uploads/](server/uploads)

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
- [client/src/pages/LandingPage.tsx](client/src/pages/LandingPage.tsx)
- [client/vite.config.ts](client/vite.config.ts)

Client pages currently depending on the app server:
- [client/src/pages/LoginPage.tsx](client/src/pages/LoginPage.tsx)
- [client/src/pages/LandingPage.tsx](client/src/pages/LandingPage.tsx)
- [client/src/pages/DashboardPage.tsx](client/src/pages/DashboardPage.tsx)
- [client/src/pages/DebugPage.tsx](client/src/pages/DebugPage.tsx)

Backend endpoints the client currently expects:
- /api/admin/check
- /api/admin/login
- /api/admin/logout
- /api/settings
- /api/alerts/active
- /api/alerts/stream
- /api/cameras
- /api/cameras/all_definitions
- /api/cameras/add
- /api/cameras/stop/<camera_id>
- /api/cameras/remove/<camera_id>
- /api/cameras/add_existing
- /api/cameras/upload
- /api/cameras/<camera_id>/snapshot
- /api/cameras/<camera_id>/stream.mjpeg
- /api/incidents
- /api/incidents/<incident_id>/pdf
- /api/incidents/<incident_id>/notes
- /api/incidents/<incident_id>
- /api/pose/status

Telegram-related controls still appear in the client UI, but they are currently backed by local client state rather than live server routes in this app server.

## 4. Cleanup Direction

Keep:
- The current frontend routes and UI flow
- The endpoint contract above
- The app server behavior that feeds the frontend
- The current incident PDF layout and formatting

Refactor later:
- Split server/app/main.py into smaller modules by feature area
- Separate inference, camera management, Telegram, incidents, and admin auth
- Remove or quarantine legacy HTML if the React client fully replaces it
- Remove unused scripts, tests, and duplicate benchmark-only assets only after confirming they are not needed

Prefer the benchmark structure as the cleanup model:
- FastAPI-style separation of host page, API, and inference session logic
- Smaller session object boundaries
- Clear distinction between backend runtime and frontend viewer

## 5. How To Start The Web App

The current web app is split into a backend API and the React frontend. Start them in two terminals:

1. Start the backend API from the `server/` folder:

```bash
cd server
python run_server.py
```

The backend listens on `http://localhost:8000`.

2. Start the frontend from the `client/` folder:

```bash
cd client
npm install
npm run dev
```

Open the Vite URL shown in the terminal, usually `http://localhost:5173`.

If you want to start the backend directly without the wrapper script, you can also run:

```bash
cd server
python -m app.main
```
