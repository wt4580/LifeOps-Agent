"""应用配置与初始化

职责：
1. 创建FastAPI应用实例
2. 配置中间件（CORS、异常处理等）
3. 初始化资源（数据库、LangGraph等）
4. 注册所有Controller路由
5. 管理应用生命周期
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
import asyncio
import json
from fastapi.middleware.cors import CORSMiddleware
from .db_config import init_db
from .knowledge_db_config import init_knowledge_db
from .checkpointer_config import create_checkpointer
from .log_config import logger
from .chat_graph_holder import set_chat_graph
from ..domain.global_exception import register_global_exception_handlers
from ...agent.graph_chat import build_chat_graph
from ...controller import (
        auth_router,
        bookkeeping_router,
        chat_router,
        plan_router,
        todo_router,
        rag_router,
        health_router,
        web_router,
        speech_router,
    )

checkpointer = None
resource_stack: AsyncExitStack | None = None

# --------------------------------------------------------------------------------------
# 注册路由（在lifespan中调用）
# --------------------------------------------------------------------------------------

def register_routers(app: FastAPI):
    """注册所有Controller路由"""
    
    # Controller中已定义完整路径，直接注册即可
    app.include_router(chat_router)
    app.include_router(plan_router)
    app.include_router(todo_router)
    app.include_router(rag_router)
    app.include_router(health_router)
    app.include_router(web_router)
    app.include_router(auth_router)
    app.include_router(bookkeeping_router)
    app.include_router(speech_router)
    
    logger.info("All routers registered")


# --------------------------------------------------------------------------------------
# 资源初始化
# --------------------------------------------------------------------------------------

async def init_resources():
    """初始化应用所需资源"""
    global checkpointer, resource_stack

    init_db()
    init_knowledge_db()
    logger.info("Database initialized (business + knowledge)")

    stack = AsyncExitStack()
    try:
        checkpointer = await stack.enter_async_context(create_checkpointer())
        set_chat_graph(build_chat_graph(checkpointer=checkpointer))
    except Exception:
        await stack.aclose()
        raise

    resource_stack = stack
    
    logger.info("LangGraph initialized")


# --------------------------------------------------------------------------------------
# 应用生命周期管理
# --------------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    # 启动时执行
    logger.info("LifeOps-Agent starting...")
    try:
        await init_resources()
        from .base_config import settings
        if settings.demo_mode and settings.demo_seed:
            from ...demo.seed import seed_demo
            await asyncio.to_thread(seed_demo)

        # 启动后台定时任务（APScheduler）
        from ...service.scheduler import start_scheduler
        if settings.scheduler_enabled and not settings.demo_mode:
            start_scheduler()

        logger.info("LifeOps-Agent started successfully")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    try:
        yield
    finally:
        # 关闭时执行
        global resource_stack

        from ...service.scheduler import stop_scheduler
        stop_scheduler()

        if resource_stack is not None:
            await resource_stack.aclose()
            resource_stack = None
        logger.info("LifeOps-Agent shutting down...")


# --------------------------------------------------------------------------------------
# 创建应用
# --------------------------------------------------------------------------------------

def create_app() -> FastAPI:
    """创建并配置FastAPI应用"""
    
    app = FastAPI(
        title="LifeOps-Agent",
        description="个人生活助手智能体",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # 注册全局异常处理器
    register_global_exception_handlers(app)

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and path != "/api/auth/login":
            from ...service.auth_service import authenticate, COOKIE_NAME
            username = await asyncio.to_thread(authenticate, request.cookies.get(COOKIE_NAME))
            if username is None:
                return JSONResponse({"detail": "请先登录"}, status_code=401)
            supplied = request.query_params.get("session_id")
            if supplied is not None and supplied != username:
                return JSONResponse({"detail": "不能访问其他会话"}, status_code=403)
            scoped_paths = ("/api/chat", "/api/todos", "/api/accounts", "/api/transactions", "/api/plan")
            scoped = path.startswith(scoped_paths)
            body_session = None
            if request.headers.get("content-type", "").startswith("application/json") or (
                scoped and request.method in {"POST", "PUT", "PATCH"}
            ):
                try:
                    body = await request.json()
                    if isinstance(body, dict):
                        body_session = body.get("session_id")
                        if body_session is not None and body_session != username:
                            return JSONResponse({"detail": "不能访问其他会话"}, status_code=403)
                except (ValueError, json.JSONDecodeError):
                    pass
            if supplied is not None and supplied != username:
                return JSONResponse({"detail": "不能访问其他会话"}, status_code=403)
            uses_query_session = request.method in {"GET", "DELETE"} or path == "/api/plan/propose"
            effective = supplied if uses_query_session else body_session
            if scoped and effective != username:
                return JSONResponse({"detail": "必须提供当前会话 ID"}, status_code=422)
            request.state.username = username
        return await call_next(request)
    
    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    register_routers(app)  # 注册所有路由

    return app


