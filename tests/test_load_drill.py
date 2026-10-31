import json
from types import SimpleNamespace
import pytest
from scripts.load_drill import final_response, percentile


def test_latency_percentiles_use_nearest_rank():
    assert percentile([], .95) is None
    assert percentile([3, 1, 2], .5) == 2
    assert percentile([3, 1, 2], .95) == 3


@pytest.mark.parametrize('events', [[], [{'type': 'error'}],
    [{'type': 'final', 'result': {}}, {'type': 'error'}],
    [{'type': 'final', 'result': {'tool_error': 'failed'}}],
    [{'type': 'final', 'result': {}}, {'type': 'final', 'result': {}}]])
def test_load_drill_rejects_incomplete_or_failed_sse(events):
    response = SimpleNamespace(raise_for_status=lambda: None,
        text='\n'.join('data: ' + json.dumps(event) for event in events))
    with pytest.raises(ValueError):
        final_response(response)


def test_load_drill_accepts_exactly_one_successful_final():
    response = SimpleNamespace(raise_for_status=lambda: None,
        text='data: ' + json.dumps({'type': 'final', 'result': {'answer': 'ok'}}))
    assert final_response(response) == {'answer': 'ok'}

def test_tcp_transport_runs_and_closes_owned_server():
    import httpx
    from fastapi import FastAPI, Request
    from scripts.load_drill import application_client
    app = FastAPI()
    @app.get('/health/ready')
    def ready():
        return {'ready': True}
    @app.get('/peer')
    def peer(request: Request):
        return {'host': request.client.host}
    from fastapi.responses import StreamingResponse
    import asyncio
    @app.post('/api/chat/stream')
    def chat():
        async def events():
            yield 'data: {"type":"progress"}\n\n'
            await asyncio.sleep(.2)
            yield 'data: {"type":"final","result":{}}\n\n'
        return StreamingResponse(events(), media_type='text/event-stream')
    with application_client(app, 'tcp') as client:
        address = str(client.base_url)
        assert client.get('/peer').json() == {'host': '127.0.0.1'}
        from scripts.load_drill import streamed_response
        _, timing = streamed_response(client, {}, {})
        assert timing['final'] - timing['first_event'] >= .1
    with pytest.raises((httpx.ConnectError, httpx.ConnectTimeout)):
        httpx.get(address.rstrip('/') + '/health/ready', timeout=1, trust_env=False)

def test_stream_metrics_distinguish_progress_from_answer_delta(monkeypatch):
    from contextlib import contextmanager
    from scripts import load_drill
    ticks = iter([0, .1, .3, .5])
    monkeypatch.setattr(load_drill.time, 'perf_counter', lambda: next(ticks))
    response = SimpleNamespace(raise_for_status=lambda: None, iter_lines=lambda: iter([
        ': heartbeat', 'data: {"type":"progress"}',
        'data: {"type":"delta","delta":"answer"}',
        'data: {"type":"final","result":{"answer":"answer"}}']))
    @contextmanager
    def stream(*args, **kwargs):
        yield response
    result, timing = load_drill.streamed_response(SimpleNamespace(stream=stream), {}, {})
    assert result == {'answer': 'answer'}
    assert timing == {'first_event': .1, 'first_delta': .3, 'final': .5}


def test_final_only_response_does_not_invent_first_delta():
    from contextlib import contextmanager
    from scripts.load_drill import streamed_response
    response = SimpleNamespace(raise_for_status=lambda: None, iter_lines=lambda: iter([
        'data: {"type":"final","result":{}}']))
    @contextmanager
    def stream(*args, **kwargs):
        yield response
    _, timing = streamed_response(SimpleNamespace(stream=stream), {}, {})
    assert timing['first_delta'] is None
    assert timing['final'] >= timing['first_event']
