"""Disposable offline browser fixture; adds latency, never changes model results."""
import os
import time
import hashlib
from pathlib import Path

import uvicorn

from app.src.demo.provider import DemoClient
from app.src.__main__ import app
from app.src import __main__ as entrypoint
from app.src.common.config.base_config import settings

if not settings.demo_mode:
    raise RuntimeError('Browser fixture requires DEMO_MODE=true')

source = hashlib.sha256()
root = Path(entrypoint.__file__).resolve().parents[1]
for path in sorted(root.rglob('*')):
    if path.is_file() and path.suffix in {'.py', '.js', '.css', '.html'}:
        source.update(path.relative_to(root).as_posix().encode())
        source.update(path.read_bytes())


@app.get('/__acceptance/info')
def acceptance_info():
    return {'fixture': 'lifeops-offline-browser', 'demo_mode': settings.demo_mode,
            'source_sha256': source.hexdigest()}

original = DemoClient.create


def delayed(self, **kwargs):
    time.sleep(float(os.environ.get('BROWSER_DEMO_DELAY_SECONDS', '0.6')))
    return original(self, **kwargs)


DemoClient.create = delayed
uvicorn.run(app, host='0.0.0.0', port=5000, log_level='warning')
