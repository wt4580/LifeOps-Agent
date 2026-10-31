"""Optional learning must have bounded admission and never block the chat path."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock
from app.src.service import chat_service as module


def submit(service):
    service._spawn_post_graph('user', 'user', 'statement', {'recent_dialogue': [{}]}, 'answer', lambda _: None)


def test_optional_background_queue_rejects_and_recovers_after_worker_failure(monkeypatch, caplog):
    slots = threading.BoundedSemaphore(1)
    queued = []
    monkeypatch.setattr(module, '_POST_GRAPH_SLOTS', slots)
    monkeypatch.setattr(module, '_POST_GRAPH_EXECUTOR', SimpleNamespace(submit=queued.append))
    monkeypatch.setattr(module, 'background_context', Mock(side_effect=RuntimeError('worker failed')))
    increment = Mock()
    monkeypatch.setattr(module.metrics, 'increment', increment)
    service = module.ChatService()
    submit(service)
    submit(service)
    assert len(queued) == 1
    assert 'capacity reached' in caplog.text
    queued[0]()
    submit(service)
    assert len(queued) == 2
    queued[1]()
    assert slots.acquire(blocking=False)
    slots.release()
    statuses = [call.kwargs['status'] for call in increment.call_args_list if call.args[0] == 'lifeops_background_admission_total']
    assert statuses == ['accepted', 'rejected_capacity', 'accepted']


def test_executor_shutdown_rejection_releases_slot_without_failing_chat(monkeypatch):
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(module, '_POST_GRAPH_SLOTS', slots)
    monkeypatch.setattr(module, '_POST_GRAPH_EXECUTOR', Mock(submit=Mock(side_effect=RuntimeError('shutdown'))))
    submit(module.ChatService())
    assert slots.acquire(blocking=False)
    slots.release()
