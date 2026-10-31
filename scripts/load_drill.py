"""Isolated concurrency drill with real SSE parsing and SQLite write checks.

Optional loopback TCP/Uvicorn transport; no paid model calls.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import socket
import threading
import subprocess
import sys
import tempfile
import time


def percentile(values, quantile):
    return sorted(values)[max(0, math.ceil(len(values) * quantile) - 1)] if values else None


def final_response(response):
    response.raise_for_status()
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    return validate_events(events)


def validate_events(events):
    finals = [event['result'] for event in events if event.get('type') == 'final']
    if len(finals) != 1 or any(event.get('type') == 'error' for event in events):
        raise ValueError('SSE did not complete successfully')
    if finals[0].get('tool_error'):
        raise ValueError('Agent operation failed')
    return finals[0]


def streamed_response(client, body, headers):
    started = time.perf_counter()
    events = []
    first_event = first_delta = final_at = None
    with client.stream('POST', '/api/chat/stream', json=body, headers=headers) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith('data: '):
                continue
            elapsed = time.perf_counter() - started
            event = json.loads(line[6:])
            events.append(event)
            if first_event is None:
                first_event = elapsed
            if first_delta is None and event.get('type') == 'delta' and event.get('delta'):
                first_delta = elapsed
            if event.get('type') == 'final':
                final_at = elapsed
    return validate_events(events), {'first_event': first_event, 'first_delta': first_delta, 'final': final_at}


@contextmanager
def application_client(app, transport):
    if transport == 'asgi':
        from fastapi.testclient import TestClient
        with TestClient(app, base_url='http://localhost', client=('127.0.0.1', 50100)) as client:
            yield client
        return
    import httpx
    import uvicorn
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(128)
    address = f'http://127.0.0.1:{listener.getsockname()[1]}'
    server = uvicorn.Server(uvicorn.Config(app, log_level='critical', access_log=False,
        timeout_graceful_shutdown=10))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError('Temporary Uvicorn server failed to start')
            time.sleep(.05)
        with httpx.Client(base_url=address, timeout=30, trust_env=False) as client:
            client.get('/health/ready').raise_for_status()
            yield client
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        listener.close()
        if thread.is_alive():
            raise RuntimeError('Temporary Uvicorn server did not stop')


def overload_case(client, user, cookie, factory):
    import asyncio
    from sqlalchemy import func, select
    from app.src.common.config.base_config import settings
    from app.src.domain.entity.chat_entity import ChatMessage, ChatRequest, Transaction
    from app.src.service.chat_service import chat_service
    entered, release = threading.Event(), threading.Event()
    original_chat = chat_service.chat
    original_limit = settings.chat_max_inflight_jobs
    async def held_chat(*args, **kwargs):
        entered.set()
        if not await asyncio.to_thread(release.wait, 15):
            raise RuntimeError('Overload hold timed out')
        return await original_chat(*args, **kwargs)
    headers = {'Cookie': cookie}
    body = {'session_id': user, 'message': '我花了10元买午饭', 'request_id': 'overload-held'}
    settings.chat_max_inflight_jobs = 1
    chat_service.chat = held_chat
    checks = {}
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            active = executor.submit(lambda: final_response(client.post('/api/chat/stream', json=body, headers=headers)))
            try:
                if not entered.wait(10):
                    raise RuntimeError('Held request did not enter execution')
                rejected_body = {**body, 'request_id': 'overload-rejected'}
                response = client.post('/api/chat/stream', json=rejected_body, headers=headers)
                response.raise_for_status()
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
                checks['retryable_capacity_error_without_final'] = (
                    len(events) == 1 and events[0].get('code') == 'chat_capacity_exceeded'
                    and events[0].get('retryable') is True and events[0].get('type') == 'error')
                with factory() as db:
                    checks['rejected_request_never_persisted'] = all(
                        db.scalar(select(func.count()).select_from(entity)) == 0
                        for entity in (ChatMessage, ChatRequest, Transaction))
            finally:
                release.set()
            completed = active.result(timeout=30)
        chat_service.chat = original_chat
        replay = final_response(client.post('/api/chat/stream', json=body, headers=headers))
        checks['completed_request_replay_matches'] = completed == replay
        final_response(client.post('/api/chat/stream', json={**body, 'request_id': 'after-overload'}, headers=headers))
        with factory() as db:
            checks['capacity_recovers_and_unique_writes_only'] = db.scalar(select(func.count()).select_from(Transaction)) == 2
        return {'checks': checks, 'passed': all(checks.values())}
    finally:
        release.set()
        chat_service.chat = original_chat
        settings.chat_max_inflight_jobs = original_limit


def application_load(root, requests, concurrency, transport="asgi", overload=False):
    os.environ.update({
        'DEMO_MODE': 'true', 'DEMO_SEED': 'false', 'QWEN_API_KEY': 'unused-offline-load',
        'BACKGROUND_LEARNING_ENABLED': 'false', 'SCHEDULER_ENABLED': 'false', 'RAG_BACKEND': 'bm25',
        'DATABASE_URL': f"sqlite:///{(root / 'business.db').as_posix()}",
        'KNOWLEDGE_DATABASE_URL': f"sqlite:///{(root / 'knowledge.db').as_posix()}",
        'CHECKPOINTER_DB_PATH': str(root / 'checkpoint.db'), 'PERSONAL_MEMORY_DIR': str(root / 'memory'),
        'HARNESS_TRACE_DIR': str(root / 'harness'), 'DOCS_DIR': str(root / 'documents'),
    })
    logging.disable(logging.CRITICAL)
    from sqlalchemy import func, select
    from app.src.common.config.app_config import create_app
    from app.src.common.config.db_config import SessionLocal, engine
    from app.src.common.config.knowledge_db_config import knowledge_engine
    from app.src.common.config.chat_graph_holder import set_chat_graph
    from app.src.domain.entity.chat_entity import Transaction
    from app.src.service.auth_service import COOKIE_NAME
    from scripts.verify import source_fingerprint
    initial_source = source_fingerprint()
    samples, failures = [], []
    try:
        with application_client(create_app(), transport) as client:
            identities = []
            for i in range(min(concurrency, 8)):
                user = f'load-user-{i}'
                response = client.post('/api/auth/login', json={'username': user, 'password': 'offline-load-2026'})
                response.raise_for_status()
                identities.append((user, f'{COOKIE_NAME}={client.cookies.get(COOKIE_NAME)}'))
            if overload:
                report = overload_case(client, *identities[0], SessionLocal)
                report.update({'mode': 'offline-demo', 'scenario': 'capacity-overload',
                    'transport': 'loopback-TCP-uvicorn' if transport == 'tcp' else 'in-process-ASGI',
                    'source_sha256': initial_source, 'source_unchanged': source_fingerprint() == initial_source})
                report['passed'] = report['passed'] and report['source_unchanged']
                return report
            def operation(index):
                user, cookie = identities[index % len(identities)]
                body = {'session_id': user, 'message': '我花了10元买午饭', 'request_id': f'load-{index}'}
                headers = {'Cookie': cookie}
                started = time.perf_counter()
                original, stream_timing = streamed_response(client, body, headers)
                replay = final_response(client.post('/api/chat/stream', json=body, headers=headers))
                if original != replay:
                    raise ValueError('Replay differs from original')
                health_started = time.perf_counter()
                client.get('/health/ready').raise_for_status()
                return (time.perf_counter() - started, time.perf_counter() - health_started, stream_timing)
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(operation, i) for i in range(requests)]
                for future in futures:
                    try:
                        samples.append(future.result(timeout=120))
                    except Exception as exc:
                        failures.append(type(exc).__name__)
            duration = time.perf_counter() - started
            with SessionLocal() as db:
                counts = dict(db.execute(select(Transaction.session_id, func.count()).group_by(Transaction.session_id)).all())
            expected = {user: sum(i % len(identities) == j for i in range(requests)) for j, (user, _) in enumerate(identities)}
            latencies = [value[0] for value in samples]
            source_unchanged = source_fingerprint() == initial_source
            return {'mode': 'offline-demo', 'transport': 'loopback-TCP-uvicorn' if transport == 'tcp' else 'in-process-ASGI', 'created_at': datetime.now(timezone.utc).isoformat(),
                'source_sha256': initial_source, 'source_unchanged': source_unchanged, 'requests': requests, 'concurrency': concurrency,
                'successful_operations': len(samples), 'failed_operations': len(failures), 'failure_types': failures,
                'duration_seconds': duration, 'operations_per_second': len(samples) / duration,
                'operation_includes': ['SSE expense request', 'identical request replay', 'readiness probe'],
                'latency_seconds': {'p50': percentile(latencies, .5), 'p95': percentile(latencies, .95), 'max': max(latencies, default=None)},
                'readiness_max_seconds': max((value[1] for value in samples), default=None),
                'stream_timing_observation': 'network-arrival' if transport == 'tcp' else 'ASGI-testclient-buffered',
                'stream_latency_seconds': {name: {
                    'observed_count': len(values), 'p50': percentile(values, .5), 'p95': percentile(values, .95)
                } for name in ('first_event', 'first_delta', 'final')
                  for values in [[sample[2][name] for sample in samples if sample[2][name] is not None]]},
                'writes_match_unique_requests_per_identity': counts == expected,
                'passed': source_unchanged and not failures and len(samples) == requests and counts == expected}
    finally:
        set_chat_graph(None)
        engine.dispose()
        knowledge_engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=int, default=16)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overload-check', action='store_true')
    parser.add_argument('--transport', choices=['asgi', 'tcp'], default='asgi')
    parser.add_argument('--data-root', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.requests <= 200 or not 1 <= args.concurrency <= 16:
        parser.error('requests must be 1..200 and concurrency 1..16')
    if args.data_root:
        report = application_load(args.data_root, args.requests, args.concurrency, args.transport, args.overload_check)
    else:
        with tempfile.TemporaryDirectory(prefix='lifeops-load-') as temporary:
            result = subprocess.run([sys.executable, '-m', 'scripts.load_drill', '--data-root', temporary,
                '--requests', str(args.requests), '--concurrency', str(args.concurrency), '--transport', args.transport, *(['--overload-check'] if args.overload_check else [])],
                capture_output=True, text=True, encoding='utf-8', timeout=300,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            if result.returncode not in (0, 1) or not result.stdout.strip():
                raise RuntimeError('Load drill subprocess failed; no report available')
            report = json.loads(result.stdout.splitlines()[-1])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
