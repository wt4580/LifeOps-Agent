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
    with application_client(app, 'tcp') as client:
        address = str(client.base_url)
        assert client.get('/peer').json() == {'host': '127.0.0.1'}
    with pytest.raises((httpx.ConnectError, httpx.ConnectTimeout)):
        httpx.get(address.rstrip('/') + '/health/ready', timeout=1, trust_env=False)
