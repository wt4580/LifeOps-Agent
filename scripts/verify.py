"""Run independent offline verification stages and retain every stage's evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def verify(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stages = [
        ("lint", ["ruff", "check", "--select", "E9,F63,F7,F82", "app", "tests", "scripts"]),
        ("unit", ["pytest", "tests", "-q", f"--junitxml={output / 'unit.xml'}"]),
        ("e2e", ["app.eval.run_e2e", "--output", str(output / "e2e.json")]),
        ("recovery", ["scripts.recovery_drill", "--output", str(output / "recovery.json")]),
        ("reference", ["app.eval.run_reference", "--output", str(output / "reference.json")]),
    ]
    results = []
    for name, arguments in stages:
        started = time.perf_counter()
        print(f"Running {name}", flush=True)
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run([sys.executable, "-m", *arguments], stdout=log, stderr=subprocess.STDOUT,
                                        timeout=600, check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                log.write("\nVerification stage exceeded 600 seconds\n")
                code = 124
            except OSError as exc:
                log.write(f"\nUnable to start stage: {type(exc).__name__}\n")
                code = 125
        results.append({"stage": name, "exit_code": code, "passed": code == 0,
                        "duration_seconds": round(time.perf_counter() - started, 2)})
        # Persist incrementally, so earlier evidence survives a later interruption.
        (output / "verification.json").write_text(json.dumps({"offline": True, "stages": results,
            "complete": len(results) == len(stages), "passed": len(results) == len(stages) and all(r["passed"] for r in results)},
            indent=2), encoding="utf-8")
    return all(result["passed"] for result in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/verification"))
    args = parser.parse_args()
    return 0 if verify(args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
