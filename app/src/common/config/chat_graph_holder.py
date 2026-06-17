"""Holds the compiled chat graph, set during app startup.

Breaking the circular import chain:
  app_config → controller → chat_controller → chat_service → app_config
by moving the shared graph reference to a leaf module.
"""

from typing import Any

_chat_graph: Any = None


def set_chat_graph(graph: Any) -> None:
    global _chat_graph
    _chat_graph = graph


def get_chat_graph() -> Any:
    return _chat_graph
