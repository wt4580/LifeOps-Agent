from concurrent.futures import ThreadPoolExecutor
import threading
import pytest


@pytest.mark.parametrize('kind', ['post', 'summary'])
def test_shutdown_cancels_pending_waits_running_and_allows_restart(monkeypatch, kind):
    from app.src.service import chat_service
    from app.src.util.retrieval import retrieval
    module = chat_service if kind == 'post' else retrieval
    executor_name = '_POST_GRAPH_EXECUTOR' if kind == 'post' else '_SUMMARY_EXECUTOR'
    accepting_name = '_POST_GRAPH_ACCEPTING' if kind == 'post' else '_SUMMARY_ACCEPTING'
    shutdown = module.shutdown_post_graph_jobs if kind == 'post' else module.shutdown_document_summary_jobs
    start = module.start_post_graph_jobs if kind == 'post' else module.start_document_summary_jobs
    pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(module, executor_name, pool)
    monkeypatch.setattr(module, accepting_name, True)
    if kind == 'summary':
        monkeypatch.setattr(module, '_SUMMARY_PENDING', False)
        monkeypatch.setattr(module, '_SUMMARY_FUTURE', None)
    else:
        monkeypatch.setattr(module, '_POST_GRAPH_SLOTS', threading.BoundedSemaphore(1))
    entered, release = threading.Event(), threading.Event()
    def running():
        entered.set()
        release.wait(5)
    active = pool.submit(running)
    assert entered.wait(2)
    pending = pool.submit(lambda: pytest.fail('pending work must be cancelled'))
    closer = threading.Thread(target=shutdown)
    closer.start()
    try:
        for _ in range(100):
            if pending.cancelled(): break
            threading.Event().wait(.01)
        assert pending.cancelled()
        assert closer.is_alive()
        assert not getattr(module, accepting_name)
    finally:
        release.set()
        closer.join(5)
    assert not closer.is_alive()
    active.result()
    start()
    replacement = getattr(module, executor_name)
    try:
        assert replacement.submit(lambda: 42).result(2) == 42
    finally:
        replacement.shutdown()
