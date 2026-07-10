# FallGuard User Mobile Frontend

This is the regular user mobile-friendly frontend for FallGuard.

## What this app includes

- Home view with two main actions:
  - Camera Status
  - Incident Reports
- Camera list with:
  - stream name
  - camera name
  - live/offline/fall status
- Live fall alert banner (with dismiss button)
- Browser notification support for fall alerts (when permission is granted)
- Incident Reports popup page with per-incident PDF access
- Incident list sorted newest-first

## Tech stack

- React + TypeScript
- Vite

## Folder location

- `user-mobile/`

## Prerequisites

- Node.js 18+
- Backend running (default expected at `http://localhost:8000`)

## Install and run

From workspace root:

```bash
cd user-mobile
npm install
npm run dev
```

Default dev URL:

- `http://localhost:5174`

## Open from phone on same network

This app is configured to expose Vite dev server on LAN.

1. Start backend on LAN host:

```bash
# from server folder
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

2. Start frontend:

```bash
cd user-mobile
npm run dev
```

3. Open on phone:

- `http://<YOUR_PC_IP>:5174`

The frontend automatically uses `http://<YOUR_PC_IP>:8000` for API calls when opened via LAN IP.

## Environment variables (optional)

You can override backend URL:

- `VITE_SERVER_BASE_URL`
- `VITE_API_BASE_URL`

Example `.env`:

```env
VITE_SERVER_BASE_URL=http://192.168.1.10:8000
```

## Build

```bash
cd user-mobile
npm run build
```

## Notes

- If browser blocks popups, incident report window may open in current tab.
- If incident PDF endpoint returns 404, that incident PDF is missing server-side.
- Browser notifications require explicit user permission.
