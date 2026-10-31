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
