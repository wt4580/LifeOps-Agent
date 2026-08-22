"""Controller层 - HTTP路由处理

包含以下控制器：
- chat_controller: 聊天接口
- plan_controller: 待办计划接口
- todo_controller: 待办事项查询接口
- rag_controller: 知识库检索接口
- health_controller: 健康检查接口
- web_controller: Web页面接口
"""

from .auth_controller import router as auth_router
from .bookkeeping_controller import router as bookkeeping_router
from .chat_controller import router as chat_router
from .speech_controller import router as speech_router
from .plan_controller import router as plan_router
from .todo_controller import router as todo_router
from .rag_controller import router as rag_router
from .health_controller import router as health_router
from .web_controller import router as web_router

__all__ = [
    "auth_router",
    "bookkeeping_router",
    "chat_router",
    "speech_router",
    "plan_router",
    "todo_router",
    "rag_router",
    "health_router",
    "web_router",
]
