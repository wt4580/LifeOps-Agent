"""Concurrent indexing must not receive a summary derived from stale content."""
from datetime import datetime
from unittest.mock import Mock
import pytest
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker
from app.src.common.config.knowledge_db_config import KnowledgeBase
from app.src.domain.entity.knowledge_entity import Document, DocumentChunk
from app.src.util.retrieval import retrieval as module


@pytest.mark.parametrize('change', ['none', 'content', 'revision', 'delete', 'summary'])
def test_summary_write_checks_source_after_model_call(tmp_path, monkeypatch, change):
    engine = create_engine(f"sqlite:///{tmp_path / 'knowledge.db'}", connect_args={'timeout': 1})
    KnowledgeBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(module, 'KnowledgeSessionLocal', factory)
    with factory() as db:
        db.add(Document(file_path='a.txt', chunk_count=1))
        db.add(DocumentChunk(file_path='a.txt', chunk_text='old content', chunk_hash='old'))
        db.commit()
    def model(*args, **kwargs):
        # This independent writer must not be blocked by a model-call transaction.
        with factory() as db:
            if change == 'content':
                db.execute(update(DocumentChunk).values(chunk_text='new content', chunk_hash='new'))
            elif change == 'revision':
                db.execute(update(Document).values(updated_at=datetime(2030, 1, 1)))
            elif change == 'delete':
                db.query(Document).delete()
            elif change == 'summary':
                db.execute(update(Document).values(summary='newer summary'))
            db.commit()
        return 'old summary'
    monkeypatch.setattr(module, 'chat_completion', model)
    try:
        module._generate_document_summaries_batch()
        with factory() as db:
            doc = db.scalar(select(Document))
            if change == 'delete':
                assert doc is None
            else:
                expected = 'old summary' if change == 'none' else 'newer summary' if change == 'summary' else None
                assert doc.summary == expected
    finally:
        engine.dispose()
