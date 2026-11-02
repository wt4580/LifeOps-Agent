import json
import sys
from unittest.mock import Mock
from app.eval import run_e2e


def test_targeted_case_selection_rejects_unknown_or_simulator_only_cases():
    import pytest
    assert [case['id'] for case in run_e2e.select_cases('real', case_ids=['multi-0'])] == ['multi-0']
    with pytest.raises(ValueError):
        run_e2e.select_cases('real', case_ids=['typo'])
    simulated = next(case['id'] for case in run_e2e.cases() if case.get('contract_only'))
    with pytest.raises(ValueError):
        run_e2e.select_cases('real', case_ids=[simulated])


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

def test_plan_records_requested_execution_limits_without_applying_them(monkeypatch, tmp_path):
    output = tmp_path / 'bounded-plan.json'
    monkeypatch.delenv('CHAT_TURN_MAX_LLM_CALLS', raising=False)
    monkeypatch.delenv('CHAT_TURN_MAX_TOKENS', raising=False)
    monkeypatch.setattr(sys, 'argv', ['run_e2e', '--mode', 'real', '--plan-only', '--limit', '8',
        '--turn-max-llm-calls', '8', '--turn-max-tokens', '20000', '--output', str(output)])
    assert run_e2e.main() == 0
    plan = json.loads(output.read_text(encoding='utf-8'))
    assert plan['turn_limits']['max_llm_calls'] == 8
    assert plan['turn_limits']['max_tokens'] == 20000
    assert plan['turn_limits']['monetary_cap_enforced'] is False
    assert 'CHAT_TURN_MAX_LLM_CALLS' not in run_e2e.os.environ


def test_real_execution_still_requires_opt_in_with_explicit_limits(monkeypatch):
    import pytest
    monkeypatch.setattr(sys, 'argv', ['run_e2e', '--mode', 'real', '--turn-max-llm-calls', '8'])
    execute = Mock()
    monkeypatch.setattr(run_e2e, 'evaluate', execute)
    with pytest.raises(SystemExit):
        run_e2e.main()
    execute.assert_not_called()
