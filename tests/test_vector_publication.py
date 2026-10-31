"""Two-file vector indexes are published through one atomic generation pointer."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from app.src.util.retrieval import rag_langchain as module


@pytest.mark.parametrize('failure', ['save', 'publish', 'missing', 'none'])
def test_vector_publication_preserves_previous_generation_on_failure(tmp_path, monkeypatch, failure):
    old = tmp_path / 'index.faiss'
    old.write_bytes(b'old-index')
    (tmp_path / 'index.pkl').write_bytes(b'old-metadata')
    monkeypatch.setattr(module.settings, 'rag_vector_dir', str(tmp_path))
    monkeypatch.setattr(module, '_load_documents_from_db', Mock(return_value=[object()]))
    monkeypatch.setattr(module, '_collect_old_doc_signatures', Mock(return_value=[]))
    monkeypatch.setattr(module, '_collect_new_doc_signatures', Mock(return_value=[]))
    monkeypatch.setattr(module, '_require_langchain', Mock())
    monkeypatch.setattr(module, '_embeddings', Mock())
    def save(directory):
        from pathlib import Path
        path = Path(directory)
        (path / 'index.faiss').write_bytes(b'new-index')
        if failure == 'save':
            raise OSError('interrupted')
        if failure != 'missing':
            (path / 'index.pkl').write_bytes(b'new-metadata')
    store = Mock()
    monkeypatch.setattr(module, "_save_vector_store", lambda store, directory: save(directory))
    monkeypatch.setitem(sys.modules, 'langchain_community.vectorstores', SimpleNamespace(FAISS=Mock(from_documents=Mock(return_value=store))))
    if failure == 'publish':
        monkeypatch.setattr(module.os, 'replace', Mock(side_effect=OSError('interrupted publication')))
    try:
        if failure == 'none':
            module.build_vector_index()
            active = module._active_vector_directory(tmp_path)
            assert (active / 'index.faiss').read_bytes() == b'new-index'
            assert (active / 'index.pkl').read_bytes() == b'new-metadata'
        else:
            with pytest.raises((OSError, ValueError)):
                module.build_vector_index()
            assert module._active_vector_directory(tmp_path) == tmp_path
        assert old.read_bytes() == b'old-index'
        assert (tmp_path / 'index.pkl').read_bytes() == b'old-metadata'
    finally:
        module._load_vector_store.cache_clear()


def test_empty_generation_replaces_pointer_without_deleting_old_files(tmp_path, monkeypatch):
    (tmp_path / 'index.faiss').write_bytes(b'old')
    monkeypatch.setattr(module.settings, 'rag_vector_dir', str(tmp_path))
    monkeypatch.setattr(module, '_load_documents_from_db', Mock(return_value=[]))
    module.build_vector_index()
    assert (module._active_vector_directory(tmp_path) / 'empty').is_file()
    assert (tmp_path / 'index.faiss').read_bytes() == b'old'


@pytest.mark.parametrize('value', ['../outside', 'bad', None])
def test_invalid_pointer_never_falls_back_to_old_files(tmp_path, value):
    (tmp_path / 'current.json').write_text(json.dumps({'generation': value}))
    with pytest.raises(ValueError):
        module._active_vector_directory(tmp_path)

def test_loader_uses_one_published_generation(tmp_path, monkeypatch):
    generation = 'a' * 32
    directory = tmp_path / 'generations' / generation
    directory.mkdir(parents=True)
    (directory / 'index.faiss').write_bytes(b'index')
    (directory / 'index.pkl').write_bytes(b'metadata')
    (tmp_path / 'current.json').write_text(json.dumps({'generation': generation}))
    loader = Mock(return_value='loaded-generation')
    monkeypatch.setattr(module, '_read_vector_store', loader)
    monkeypatch.setattr(module.settings, 'rag_vector_dir', str(tmp_path))
    monkeypatch.setattr(module, '_require_langchain', Mock())
    monkeypatch.setattr(module, '_embeddings', Mock())
    monkeypatch.setattr(module, '_load_documents_from_db', Mock(return_value=[]))
    monkeypatch.setattr(module, '_collect_old_doc_signatures', Mock(return_value=[]))
    monkeypatch.setattr(module, '_collect_new_doc_signatures', Mock(return_value=[]))
    monkeypatch.setitem(sys.modules, 'langchain_community.vectorstores', SimpleNamespace(FAISS=Mock(load_local=loader)))
    module._load_vector_store.cache_clear()
    try:
        assert module._load_vector_store() == 'loaded-generation'
        assert loader.call_args.args[0] == directory
    finally:
        module._load_vector_store.cache_clear()
