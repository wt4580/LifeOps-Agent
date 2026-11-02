"""Disposable offline browser fixture; adds latency, never changes model results."""
import os
import time

import uvicorn

from app.src.demo.provider import DemoClient
from app.src.__main__ import app

original = DemoClient.create


def delayed(self, **kwargs):
    time.sleep(float(os.environ.get('BROWSER_DEMO_DELAY_SECONDS', '0.6')))
    return original(self, **kwargs)


DemoClient.create = delayed
uvicorn.run(app, host='0.0.0.0', port=5000, log_level='warning')
