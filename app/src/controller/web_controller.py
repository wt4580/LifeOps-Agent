"""Web页面控制器 - 处理Web页面相关的HTTP请求"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pathlib import Path
from fastapi.templating import Jinja2Templates

router = APIRouter()

# 模板目录配置（相对于controller目录的上级）
# 当前文件: app/src/controller/web_controller.py
# 目标目录: app/src/templates (需要向上1级)
TEMPLATES_DIR = Path(__file__).parent.parent / "common" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    """渲染单页 UI"""
    return templates.TemplateResponse("index.html", {"request": request})