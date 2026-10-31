"""Isolated ASGI concurrency drill with real SSE parsing and SQLite write checks.

No network listener or paid model calls; this is not a production throughput test.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def percentile(values, quantile):
    return sorted(values)[max(0, math.ceil(len(values) * quantile) - 1)] if values else None


def final_response(response):
    response.raise_for_status()
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    finals = [event['result'] for event in events if event.get('type') == 'final']
    if len(finals) != 1 or any(event.get('type') == 'error' for event in events):
        raise ValueError('SSE did not complete successfully')
    if finals[0].get('tool_error'):
        raise ValueError('Agent operation failed')
    return finals[0]


def application_load(root, requests, concurrency):
    os.environ.update({
        'DEMO_MODE': 'true', 'DEMO_SEED': 'false', 'QWEN_API_KEY': 'unused-offline-load',
        'BACKGROUND_LEARNING_ENABLED': 'false', 'SCHEDULER_ENABLED': 'false', 'RAG_BACKEND': 'bm25',
        'DATABASE_URL': f"sqlite:///{(root / 'business.db').as_posix()}",
        'KNOWLEDGE_DATABASE_URL': f"sqlite:///{(root / 'knowledge.db').as_posix()}",
        'CHECKPOINTER_DB_PATH': str(root / 'checkpoint.db'), 'PERSONAL_MEMORY_DIR': str(root / 'memory'),
        'HARNESS_TRACE_DIR': str(root / 'harness'), 'DOCS_DIR': str(root / 'documents'),
    })
    logging.disable(logging.CRITICAL)
    from fastapi.testclient import TestClient
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
        with TestClient(create_app(), base_url='http://localhost', client=('127.0.0.1', 50100)) as client:
            identities = []
            for i in range(min(concurrency, 8)):
                user = f'load-user-{i}'
                response = client.post('/api/auth/login', json={'username': user, 'password': 'offline-load-2026'})
                response.raise_for_status()
                identities.append((user, f'{COOKIE_NAME}={client.cookies.get(COOKIE_NAME)}'))
            def operation(index):
                user, cookie = identities[index % len(identities)]
                body = {'session_id': user, 'message': '我花了10元买午饭', 'request_id': f'load-{index}'}
                headers = {'Cookie': cookie}
                started = time.perf_counter()
                original = final_response(client.post('/api/chat/stream', json=body, headers=headers))
                replay = final_response(client.post('/api/chat/stream', json=body, headers=headers))
                if original != replay:
                    raise ValueError('Replay differs from original')
                health_started = time.perf_counter()
                client.get('/health/ready').raise_for_status()
                return (time.perf_counter() - started, time.perf_counter() - health_started)
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
            return {'mode': 'offline-demo', 'transport': 'in-process-ASGI', 'created_at': datetime.now(timezone.utc).isoformat(),
                'source_sha256': initial_source, 'source_unchanged': source_unchanged, 'requests': requests, 'concurrency': concurrency,
                'successful_operations': len(samples), 'failed_operations': len(failures), 'failure_types': failures,
                'duration_seconds': duration, 'operations_per_second': len(samples) / duration,
                'operation_includes': ['SSE expense request', 'identical request replay', 'readiness probe'],
                'latency_seconds': {'p50': percentile(latencies, .5), 'p95': percentile(latencies, .95), 'max': max(latencies, default=None)},
                'readiness_max_seconds': max((value[1] for value in samples), default=None),
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
    parser.add_argument('--data-root', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.requests <= 200 or not 1 <= args.concurrency <= 16:
        parser.error('requests must be 1..200 and concurrency 1..16')
    if args.data_root:
        report = application_load(args.data_root, args.requests, args.concurrency)
    else:
        with tempfile.TemporaryDirectory(prefix='lifeops-load-') as temporary:
            result = subprocess.run([sys.executable, '-m', 'scripts.load_drill', '--data-root', temporary,
                '--requests', str(args.requests), '--concurrency', str(args.concurrency)],
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
