"""Operational ingestion limits; failures abort the enclosing indexing transaction."""
import os
from ...common.config.base_config import settings


class IngestionLimitExceeded(ValueError):
    pass


def check_file_size(path: str) -> None:
    if os.path.getsize(path) > settings.ingest_max_file_bytes:
        raise IngestionLimitExceeded("Document exceeds INGEST_MAX_FILE_BYTES")


def check_text_size(text: str) -> str:
    if len(text) > settings.ingest_max_text_chars:
        raise IngestionLimitExceeded("Document exceeds INGEST_MAX_TEXT_CHARS")
    return text
