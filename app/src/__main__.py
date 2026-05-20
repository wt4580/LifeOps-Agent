import uvicorn
from app.src.common.config.app_config import create_app
from app.src.common.config.log_config import logger
import uvicorn
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = create_app()
# --------------------------------------------------------------------------------------
# 挂载静态资源（统一管理）
# --------------------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == '__main__':
    # 开发环境启动方式
    logger.info("启动服务")

    uvicorn.run(app, host="0.0.0.0", port=5000)