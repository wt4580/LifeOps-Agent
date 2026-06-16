$ErrorActionPreference = "Stop"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.src.main:app --reload

