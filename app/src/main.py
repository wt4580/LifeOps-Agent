"""LifeOps-Agent 应用入口

职责：
1. 创建FastAPI应用实例（通过app_config）
2. 挂载静态资源
3. 启动应用
"""

from __future__ import annotations

from pathlib import Path

from fastapi.staticfiles import StaticFiles

from .common.config.app_config import create_app

# --------------------------------------------------------------------------------------
# 创建应用
# --------------------------------------------------------------------------------------

app = create_app()

# --------------------------------------------------------------------------------------
# 挂载静态资源（统一管理）
# --------------------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")