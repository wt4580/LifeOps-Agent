from __future__ import annotations

from ...common.config.base_config import settings
from .limits import check_file_size, check_text_size


def extract_txt(path: str) -> str:
    """Read bounded UTF-8 text; reject oversized input without silent truncation."""
    check_file_size(path)
    with open(path, "rb") as handle:
        raw = handle.read(settings.ingest_max_file_bytes + 1)
    if len(raw) > settings.ingest_max_file_bytes:
        from .limits import IngestionLimitExceeded
        raise IngestionLimitExceeded("Document exceeds INGEST_MAX_FILE_BYTES")
    return check_text_size(raw.decode("utf-8", errors="ignore"))
