import asyncio
from unittest.mock import AsyncMock, Mock
import pytest


@pytest.mark.parametrize('cleanup_failure', [None, 'scheduler', 'chat', 'post', 'summary', 'resources'])
def test_startup_failure_unwinds_all_acquired_resources(monkeypatch, caplog, cleanup_failure):
    from app.src.common.config import app_config as module
    from app.src.common.config.base_config import settings
    from app.src.service import scheduler, session_queue, chat_service
    from app.src.util.retrieval import retrieval
    from app.src.demo import seed
    order = []
    def cleanup(name):
        order.append(name)
        if cleanup_failure == name:
            raise RuntimeError('private-cleanup-content')
    async def initialize():
        module.resource_stack = Mock(aclose=AsyncMock(side_effect=lambda: cleanup('resources')))
    monkeypatch.setattr(module, 'init_resources', initialize)
    monkeypatch.setattr(module, 'set_chat_graph', Mock())
    monkeypatch.setattr(module, 'resource_stack', None)
    monkeypatch.setattr(module, 'checkpointer', None)
    monkeypatch.setattr(settings, 'demo_mode', True)
    monkeypatch.setattr(settings, 'demo_seed', True)
    monkeypatch.setattr(seed, 'seed_demo', Mock(side_effect=RuntimeError('private-startup-content')))
    monkeypatch.setattr(chat_service, 'start_post_graph_jobs', Mock())
    monkeypatch.setattr(retrieval, 'start_document_summary_jobs', Mock())
    monkeypatch.setattr(scheduler, 'stop_scheduler', lambda: cleanup('scheduler'))
    monkeypatch.setattr(session_queue, 'shutdown_chat_jobs', AsyncMock(side_effect=lambda _: cleanup('chat')))
    monkeypatch.setattr(chat_service, 'shutdown_post_graph_jobs', lambda: cleanup('post'))
    monkeypatch.setattr(retrieval, 'shutdown_document_summary_jobs', lambda: cleanup('summary'))
    async def scenario():
        async with module.lifespan(None):
            pytest.fail('startup must fail before serving')
    with pytest.raises(RuntimeError):
        asyncio.run(scenario())
    assert order == ['scheduler', 'chat', 'post', 'summary', 'resources']
    assert module.resource_stack is None and module.checkpointer is None
    module.set_chat_graph.assert_called_once_with(None)
    assert 'private-startup-content' not in caplog.text
    assert 'private-cleanup-content' not in caplog.text

@pytest.mark.parametrize('close_failure', [False, True])
def test_graph_initialization_failure_closes_checkpoint_and_clears_references(monkeypatch, close_failure):
    from contextlib import asynccontextmanager
    from app.src.common.config import app_config as module
    closed = []
    checkpoint = object()
    @asynccontextmanager
    async def checkpoint_context():
        try:
            yield checkpoint
        finally:
            closed.append(True)
            if close_failure:
                raise RuntimeError('close failed')
    monkeypatch.setattr(module, 'resource_stack', None)
    monkeypatch.setattr(module, 'checkpointer', None)
    monkeypatch.setattr(module, 'init_db', Mock())
    monkeypatch.setattr(module, 'init_knowledge_db', Mock())
    monkeypatch.setattr(module, 'create_checkpointer', checkpoint_context)
    monkeypatch.setattr(module, 'build_chat_graph', Mock(side_effect=RuntimeError('compile failed')))
    setter = Mock()
    monkeypatch.setattr(module, 'set_chat_graph', setter)
    with pytest.raises(RuntimeError):
        asyncio.run(module.init_resources())
    assert closed == [True]
    assert module.checkpointer is None and module.resource_stack is None
    setter.assert_called_once_with(None)


def test_duplicate_initialization_rejected_before_touching_databases(monkeypatch):
    from app.src.common.config import app_config as module
    existing = object()
    monkeypatch.setattr(module, 'resource_stack', existing)
    initialize = Mock()
    monkeypatch.setattr(module, 'init_db', initialize)
    with pytest.raises(RuntimeError, match='already initialized'):
        asyncio.run(module.init_resources())
    initialize.assert_not_called()
    assert module.resource_stack is existing
