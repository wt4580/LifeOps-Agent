from __future__ import annotations

import os
import tempfile
import atexit
from pathlib import Path
from unittest.mock import patch

import pytest


os.environ["QWEN_API_KEY"] = "test-key"
os.environ["DEMO_MODE"] = "false"
os.environ["DEMO_SEED"] = "false"
os.environ["AMAP_WEATHER_KEY"] = "test-weather-key"
_TEST_DIRECTORY = tempfile.TemporaryDirectory(prefix="lifeops-tests-")
atexit.register(_TEST_DIRECTORY.cleanup)
_TEST_ROOT = Path(_TEST_DIRECTORY.name)
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'business.db').as_posix()}"
os.environ["KNOWLEDGE_DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'knowledge.db').as_posix()}"
os.environ["CHECKPOINTER_DB_PATH"] = str(_TEST_ROOT / "checkpointer.db")
os.environ["PERSONAL_MEMORY_DIR"] = str(_TEST_ROOT / "memory")
os.environ["HARNESS_TRACE_DIR"] = str(_TEST_ROOT / "harness")

from app.src.common.config.db_config import init_db, engine
from app.src.common.config.knowledge_db_config import init_knowledge_db, knowledge_engine

init_db()
init_knowledge_db()
atexit.register(engine.dispose)
atexit.register(knowledge_engine.dispose)


@pytest.fixture(autouse=True)
def mock_all_llm_calls():
    consumer_modules = [
        "app.src.agent.planner.planner",
        "app.src.agent.agent_router",
        "app.src.agent.time.time_parser",
        "app.src.agent.memory.memory_extractor",
        "app.src.agent.graph_chat",
        "app.src.service.topic_service",
    ]
    patchers = []
    for mod in consumer_modules:
        patcher = patch(f"{mod}.chat_completion")
        mock = patcher.start()
        mock.return_value = "{}"
        patchers.append(patcher)
    yield
    for p in patchers:
        p.stop()
