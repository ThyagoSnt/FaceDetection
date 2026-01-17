# Face Server (Local Camera Feed Analysis)

A lightweight local service for **face enrollment** and **face recognition** from images, plus a simple **RTSP camera scheduler** that can run continuous analysis and send notifications (e.g., to Telegram). A small **Streamlit UI** is included for interactive use.

## What this project provides

- **FastAPI HTTP API**
  - Register (“add”) one or more face images under a `person_id`
  - Recognize (“retrieve”) a person from an uploaded image
  - Remove a person from the store
  - List registered persons and how many embeddings each has
  - Schedule RTSP camera jobs for continuous analysis
- **Background CameraScheduler**
  - Runs on app startup and can manage multiple camera jobs
- **Authentication**
  - Simple **Bearer token** auth using `AUTH_TOKEN`
- **Streamlit UI**
  - Upload images, list persons, schedule cameras, and fetch Telegram chat IDs

---

## Project structure

```
.
├── backend
│   ├── api
│   │   ├── app.py               # FastAPI entrypoint: backend.api.app:app
│   │   ├── container.py         # Dependency wiring (singletons)
│   │   ├── deps.py              # Bearer auth (AUTH_TOKEN)
│   │   ├── routers/             # API routes
│   │   ├── schemas.py           # Pydantic schemas
│   │   └── utils/               # Small helpers (e.g., image decode)
│   ├── src                      # Core logic (embedder/store/scheduler) *(may be moved to core/)*
│   └── ui
│       └── web_ui.py            # Streamlit UI
├── data                          # Persistent volume (face store, etc.)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── Makefile
```

> Note: depending on your refactor stage, the “core logic” may live in `backend/src/` or in a top-level `core/` package. The API imports should match your current layout.

---

## Requirements

- Docker + Docker Compose (recommended)
- For GPU support: NVIDIA drivers + NVIDIA Container Toolkit

If you run without GPU, remove/disable `gpus: all` in `docker-compose.yml`.

---

## Environment variables

Create a `.env` file in the project root. Minimal example:

```env
# API auth (required)
AUTH_TOKEN="replace_with_a_strong_random_token"

# Streamlit UI (recommended)
# If your FastAPI is exposed on localhost:8000, then:
UTILITIES_URL="http://localhost:8000/person"
CAMERA_SCHEDULER_URL="http://localhost:8000/cameras/schedule"
```

### Generating a strong token

Linux/macOS:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Then set:

```env
AUTH_TOKEN="PASTE_THE_OUTPUT_HERE"
```

---

## Quickstart (Docker)

1) Build and start:

```bash
docker compose up -d --build
```

2) Open the API docs:

- Swagger UI: `http://localhost:8000/docs`

3) Call a protected endpoint with the token:

```bash
curl -H "Authorization: Bearer $AUTH_TOKEN" http://localhost:8000/persons
```

> If you are using a `.env` file, you can export it in your shell:
>
> ```bash
> export $(grep -v '^#' .env | xargs)
> ```

---

## Authentication

All API routes are protected with Bearer auth.

Send this header on each request:

```
Authorization: Bearer <AUTH_TOKEN>
```

If missing/invalid, the API returns `401/403`.

---

## API endpoints

### `GET /persons`

Lists registered persons and number of embeddings per person.

Example:

```bash
curl -H "Authorization: Bearer $AUTH_TOKEN" http://localhost:8000/persons
```

### `POST /person` (multipart form)

A unified endpoint with actions:

- `action=add` + `person_id` (optional) + one or more `image` files  
- `action=retrieve` + one `image` file  
- `action=remove` + `person_id`  
- `action=camera_status` + `camera_id`

Examples:

**Add faces**
```bash
curl -X POST "http://localhost:8000/person"   -H "Authorization: Bearer $AUTH_TOKEN"   -F "action=add"   -F "person_id=alice"   -F "image=@/path/to/img1.jpg"   -F "image=@/path/to/img2.jpg"
```

**Retrieve (recognize)**
```bash
curl -X POST "http://localhost:8000/person"   -H "Authorization: Bearer $AUTH_TOKEN"   -F "action=retrieve"   -F "image=@/path/to/query.jpg"
```

**Remove**
```bash
curl -X POST "http://localhost:8000/person"   -H "Authorization: Bearer $AUTH_TOKEN"   -F "action=remove"   -F "person_id=alice"
```

### `POST /cameras/schedule`

Schedules one or more camera jobs.

Payload fields (see `backend/api/schemas.py`):

- `rtsp_link` (string)
- `camera_nickname` (string)
- `telegram_api_token` (string)
- `telegram_chat_id` (string)
- `threshold` (float, default `0.5`)
- `cooldown_seconds` (int, default `3600`)

Single job:

```bash
curl -X POST "http://localhost:8000/cameras/schedule"   -H "Authorization: Bearer $AUTH_TOKEN"   -H "Content-Type: application/json"   -d '{
    "rtsp_link": "rtsp://user:pass@ip:port/stream",
    "camera_nickname": "FrontDoor",
    "telegram_api_token": "YOUR_TELEGRAM_BOT_TOKEN",
    "telegram_chat_id": "123456789",
    "threshold": 0.55,
    "cooldown_seconds": 60
  }'
```

Batch (array):

```bash
curl -X POST "http://localhost:8000/cameras/schedule"   -H "Authorization: Bearer $AUTH_TOKEN"   -H "Content-Type: application/json"   -d '[
    { "rtsp_link": "rtsp://...", "camera_nickname": "Cam01", "telegram_api_token": "...", "telegram_chat_id": "123", "threshold": 0.55, "cooldown_seconds": 60 },
    { "rtsp_link": "rtsp://...", "camera_nickname": "Cam02", "telegram_api_token": "...", "telegram_chat_id": "123", "threshold": 0.60, "cooldown_seconds": 120 }
  ]'
```

---

## Streamlit UI

The UI lives at:

- `backend/ui/web_ui.py`

It reads `.env` and lets the user **set/override `AUTH_TOKEN` in the sidebar**, applying it to every request.

Run locally (outside Docker):

```bash
streamlit run backend/ui/web_ui.py
```

---

## Data persistence

By default, embeddings/person data is persisted under the mounted `data/` directory (e.g., `/data/person_store.json` inside the container).

- Ensure `./data` is mounted as a volume in `docker-compose.yml` so that registrations survive container restarts.

> If you later migrate to a different persistence layer (SQLite, FAISS, etc.), keep the same `/data` volume so you can persist the new files as well.

---

## Development notes

### Avoid rebuilding the image when only code changes

For development, prefer mounting code as a volume (Docker Compose) and avoid `COPY . .` in the Dockerfile. This way the image only rebuilds when the **Dockerfile** or **requirements.txt** changes.

Example compose volumes (typical):

```yaml
volumes:
  - ./backend:/app/backend
  - ./data:/data
```

### Hot reload

For a smoother dev loop, run Uvicorn with `--reload` (dev only):

```bash
python -m uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Troubleshooting

- **401/403 errors:** ensure you are sending `Authorization: Bearer <AUTH_TOKEN>` on every request.
- **No Telegram chat IDs listed:** your bot may not have received recent messages. Send a message to the bot (or to the group) and try again. In groups, adjust BotFather privacy settings if needed.
- **GPU not visible:** verify NVIDIA Container Toolkit installation and that `gpus: all` is supported by your Docker setup.

---

## License

Add your license choice here (MIT/Apache-2.0/Proprietary/etc.).