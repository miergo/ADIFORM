# Pose Estimation — Agent Brief

**Local push-up workout game.** Webcam or uploaded video → YOLO pose → live MJPEG + HUD → scorecard. Not a generic pose demo.

## Keep it simple

This is a small, local game. Every change should make the system **simpler or smaller**, not bigger.

| Rule | Meaning |
|------|---------|
| **KISS** | Smallest change that works. Edit an existing function/file before adding a new one. No extra layers "for later." |
| **YAGNI** | No speculative features, files, endpoints, dependencies, or abstractions. If the task does not ask for it, do not add it. |
| **One path** | One session, one status shape, one stream, one way to start/stop. No parallel implementations, duplicate trackers, or revived archive patterns. |
| **Thin UI** | React is a game client of the closed API. View state lives in `App.jsx`. Poll `/api/status`; do not invent a second source of truth. |

**Workout coach comes later.** A small LLM that writes a report from the game session is planned, but it is **not in this codebase**. Do not add LLM calls, prompt files, report generators, chat, or extra "coach" copy. Do not stub or partially build it. Keep `POST /api/stop` as simple game results that a future report could read — do not add a second analytics/report pipeline for that future.

## Product

**Game loop:** webcam or uploaded video → YOLO pose → live MJPEG + `/api/status` → HUD while playing → scorecard from `POST /api/stop`.

**Phases:** `idle` → `calibrating` → `ready` → `active`. Webcam: calibrate, then `POST /api/go`. Uploaded video: auto-starts into `active`.

**UI flow** (view state in `App.jsx`, no router): landing → webcam rules → session → ended. Uploaded video skips rules and goes straight to session.

## Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI, Ultralytics YOLO, OpenCV, NumPy |
| Frontend | React 19 + Vite |
| Weights | `backend/model/yolo26n-pose.pt` |
| Recordings | `backend/uploads/` |

## Edit map

Prefer these files. Do not add new ones unless the task requires it.

| Change type | File |
|-------------|------|
| Analysis, reps, calibration, overlays, metrics | `backend/pose.py` |
| Camera/video I/O, YOLO load, session thread, status | `backend/session.py` |
| HTTP routes | `backend/server.py` |
| UI shell, pages, components, styles | `frontend/src/App.jsx`, `frontend/src/pages/`, `frontend/src/components/`, `frontend/src/index.css` |

## Closed API

Do not invent routes, auth, WebSockets, or a DB.

- `GET /api/health`, `/api/status`, `/api/stream`
- `POST /api/start/webcam`, `/api/start/video`, `/api/go`, `/api/stop`

One in-memory `WorkoutSession`. Webcam and video share the same phases and status shape.

`POST /api/stop` returns game results: `{reps, score, top_bar, low_bar}`.

## Not product code

Do not extend or copy patterns from these:

- `backend/model/squats.ipynb` — exploratory only
- `frontend/src/archive/` — old UI; references removed recording/replay APIs
- `backend/preview_pose.py` — offline overlay script

## Anti-hallucination

- Keep `Pushup.update()` keys aligned with `IDLE_STATUS` / `_status_from` in `backend/session.py` and whatever the UI already reads.
- Do not add pydantic models, auth, or a database unless explicitly asked.
- Do not add dependencies, files, or endpoints unless the task asks for them.

## Run / test

```text
# backend (from backend/)
uvicorn server:app --reload --port 8000

# frontend
cd frontend && npm install && npm run dev
# Vite proxies /api → http://127.0.0.1:8000

# tests
cd backend && pytest
```

## Mentor mode

For teaching-only sessions (no code), attach `.cursor/rules/mentor.mdc`.
