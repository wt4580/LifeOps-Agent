"""Optional real FAISS integration; uses deterministic embeddings without downloads."""
import pytest
pytest.importorskip('faiss')
pytest.importorskip('langchain_community')
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker
from app.src.common.config.knowledge_db_config import KnowledgeBase
from app.src.domain.entity.knowledge_entity import DocumentChunk
from app.src.util.retrieval import rag_langchain as vector, retrieval


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float('apple' in text), float('banana' in text), 0.1]


def test_real_faiss_generation_rebuild_and_stale_hit_rejection(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'knowledge.db'}")
    KnowledgeBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(vector, 'KnowledgeSessionLocal', factory)
    monkeypatch.setattr(retrieval, 'KnowledgeSessionLocal', factory)
    monkeypatch.setattr(vector.settings, 'rag_vector_dir', str(tmp_path / 'vectors'))
    monkeypatch.setattr(vector, '_embeddings', lambda: LocalEmbeddings())
    with factory() as db:
        row = DocumentChunk(file_path='fruit.txt', chunk_text='apple fruit', chunk_hash='apple')
        db.add(row)
        db.commit()
        ident = row.id
    vector._load_vector_store.cache_clear()
    try:
        vector.build_vector_index()
        first = vector._active_vector_directory(tmp_path / 'vectors')
        hits = retrieval._vector_recall('apple', 3)
        assert [hit.chunk_id for hit in hits] == [ident]
        with factory() as db:
            db.execute(update(DocumentChunk).values(chunk_text='banana fruit', chunk_hash='banana'))
            db.commit()
        # Cached old FAISS objects must not leak old evidence after database edits.
        assert retrieval._vector_recall('apple', 3) == []
        vector.build_vector_index(rebuild=True)
        second = vector._active_vector_directory(tmp_path / 'vectors')
        assert second != first and first.is_dir()
        assert [hit.snippet for hit in retrieval._vector_recall('banana', 3)] == ['banana fruit']
        with factory() as db:
            db.query(DocumentChunk).delete()
            db.commit()
        vector.build_vector_index(rebuild=True)
        assert vector.vector_recall('banana', candidate_k=3) == []
    finally:
        vector._load_vector_store.cache_clear()
        engine.dispose()
