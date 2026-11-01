import json
import sys
from unittest.mock import Mock
from app.eval import run_e2e


def test_real_evaluation_plan_never_executes_provider(monkeypatch, tmp_path):
    output = tmp_path / 'plan.json'
    execute = Mock(side_effect=AssertionError('plan must not execute evaluation'))
    monkeypatch.setattr(run_e2e, 'evaluate', execute)
    monkeypatch.setattr(sys, 'argv', ['run_e2e', '--mode', 'real', '--plan-only', '--limit', '8', '--output', str(output)])
    assert run_e2e.main() == 0
    plan = json.loads(output.read_text(encoding='utf-8'))
    assert plan['n_cases'] == 8
    assert len(plan['groups']) == 8
    assert plan['n_messages'] == sum(len(case['messages']) for case in plan['cases'])
    assert plan['cases_sha256'] == run_e2e.case_fingerprint(plan['cases'])
    assert plan['provider_calls_executed'] == 0
    assert plan['execution_requires_paid_opt_in'] is True
    assert plan['model_quality_measured'] is False
    execute.assert_not_called()
