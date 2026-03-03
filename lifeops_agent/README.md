# LifeOps-Agent (Web)

A minimal, runnable personal life assistant agent using Qwen (DashScope OpenAI-compatible mode), FastAPI, SQLite, and a simple Jinja2 UI.

## Requirements

- Windows
- Python 3.10+
- Tesseract OCR (optional, for PNG OCR)
  - Ensure `tesseract.exe` is on PATH or set `TESSERACT_CMD` in `.env`

## Quick start (Windows)

1) Create and activate venv

```
python -m venv .venv
.venv\Scripts\activate
```

2) Install dependencies

```
pip install -r requirements.txt
```

3) Create `.env` based on `.env.example` and set `QWEN_API_KEY`

4) Run the app

```
python -m uvicorn lifeops.main:app --reload
```

Open http://127.0.0.1:8000

## Features (MVP)

- Chat with Qwen (DashScope OpenAI-compatible mode)
- Human-in-the-loop plan proposal and confirm
- Auto memory extraction into candidates
- Todos for today/upcoming
- Local document indexing (PDF/TXT/PNG+OCR) and RAG with citations

## Notes

- The local docs directory is read at runtime and not bundled in the project.
- SQLite database is created at `./data/lifeops.db` by default.
- No API keys are stored in the repository.

