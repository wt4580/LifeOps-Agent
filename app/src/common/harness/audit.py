"""Local JSONL diagnostics containing metadata only, never prompts or tool arguments."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from ..config.base_config import settings

logger = logging.getLogger(__name__)
_WRITE_LOCK = threading.Lock()


def write_turn(session_id: str, summary: dict) -> None:
    now = datetime.now(timezone.utc)
    fields = {"call_id", "parent_call_id", "kind", "name", "elapsed_ms", "status",
              "error_type", "usage_source", "prompt_tokens", "completion_tokens", "charged_tokens", "started_offset_ms"}
    record = {"timestamp": now.isoformat(),
              "session_hash": hashlib.sha256(session_id.encode()).hexdigest(), **summary,
              "invocations": [{key: value for key, value in invocation.items() if key in fields}
                              for invocation in summary.get("invocations", [])]}
    try:
        directory = Path(settings.harness_trace_dir)
        with _WRITE_LOCK:
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"{now.date().isoformat()}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, TypeError, ValueError):
        logger.warning("Unable to persist harness metadata turn=%s", summary.get("turn_id"))
