$ErrorActionPreference = "Stop"
python -m venv agent_env
agent_env\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.src

