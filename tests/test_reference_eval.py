import json
import os
import subprocess
import sys
from pathlib import Path

from app.eval.run_reference import load_reference_cases


def test_reference_fixtures_have_manual_labels_and_cover_multiple_documents():
    root = Path(__file__).resolve().parents[1]
    cases = load_reference_cases(root)
    assert any(len(case["expected"]) > 1 for case in cases)
    assert any(not case["expected"] for case in cases)
    assert any(case["id"] == "personal-guideline" for case in cases)
    assert any(case["id"] == "unknown-diary-owner" for case in cases)


def test_benchmark_runs_in_its_own_process_without_real_model_or_personal_database(tmp_path):
    report_file = tmp_path / "reference.json"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "QWEN_API_KEY": "", "DATABASE_URL": "sqlite:////must-not-be-opened.db"}
    result = subprocess.run([sys.executable, "-m", "app.eval.run_reference", "--output", str(report_file)],
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=60)
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["meta"]["model_quality_measured"] is False
    assert report["meta"]["dataset_split"] == "development"
    assert report["meta"]["synthetic"] is True
    assert len(report["meta"]["corpus_sha256"]) == 64
    assert len(report["meta"]["source_sha256"]) == 64
    assert report["metrics"]["metric_version"] == 2
    assert report["metrics"]["recall@5"] >= 0.8
    assert report["negative_queries"] == {"n": 2, "empty_results": 2}
    assert report["gate"]["passed"] is True
