"""Verify an existing test image and export reports before removing owned resources."""
import argparse
import json
from pathlib import Path
import subprocess
from uuid import uuid4


def verify_container(image, output, context="desktop-linux"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    name = "lifeops-verification-" + uuid4().hex
    docker = ["docker", "--context", context]
    created = False
    try:
        subprocess.run([*docker, "create", "--name", name, "--network", "none", "--read-only",
                        "--tmpfs", "/tmp:rw,size=256m", "--tmpfs", "/data:rw,size=128m,uid=10001,gid=10001",
                        "--mount", "type=volume,destination=/reports", image,
                        "python", "-m", "scripts.verify", "--output", "/reports/verification"], check=True)
        created = True
        with (output / "container.log").open("w", encoding="utf-8") as log:
            attached = subprocess.run([*docker, "start", "-a", name], stdout=log, stderr=subprocess.STDOUT)
        state = subprocess.run([*docker, "inspect", name, "--format", "{{json .State}}"],
                               check=True, capture_output=True, text=True)
        status = json.loads(state.stdout)
        # Anonymous Docker volume survives process exit, unlike tmpfs.
        exported = subprocess.run([*docker, "cp", f"{name}:/reports/.", str(output)])
        report = {"image": image, "container": name, "state": status,
                  "attach_exit_code": attached.returncode, "reports_exported": exported.returncode == 0}
        (output / "container-result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return status.get("Status") == "exited" and status.get("ExitCode") == 0 and exported.returncode == 0
    finally:
        if created:
            subprocess.run([*docker, "rm", "-f", "-v", name], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context", default="desktop-linux")
    args = parser.parse_args()
    return 0 if verify_container(args.image, args.output, args.context) else 1


if __name__ == "__main__":
    raise SystemExit(main())
