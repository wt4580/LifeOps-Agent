from __future__ import annotations

import os
from unittest.mock import patch

import pytest


os.environ["QWEN_API_KEY"] = "test-key"
os.environ["AMAP_WEATHER_KEY"] = "test-weather-key"


@pytest.fixture(autouse=True)
def mock_all_llm_calls():
    consumer_modules = [
        "app.src.agent.planner.planner",
        "app.src.agent.agent_router",
        "app.src.agent.time.time_parser",
        "app.src.agent.memory.memory_extractor",
        "app.src.agent.graph_chat",
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
