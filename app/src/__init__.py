"""
lifeops 包（项目核心业务层）

这个目录可以理解为“后端智能体的大脑 + 工具箱”。

阅读建议（零基础顺序）：
1) `settings.py`：先看配置从哪里来（.env）。
2) `db.py` + `models.py`：再看数据怎么落库。
3) `main.py`：看 FastAPI 路由如何把页面操作转成后端动作。
4) `graph_chat.py`：看 LangGraph 状态机如何编排“路由 -> 工具 -> 回复”。
5) `planner.py` / `time_parser.py`：看待办草案与时间解析逻辑。
6) `retrieval.py` / `rag_langchain.py`：看知识库索引与检索链路。
7) `agent_router.py`：看 LLM 如何选择工具。

这个 `__init__.py` 本身不放运行逻辑，主要作用是：
- 告诉 Python：`lifeops` 是一个可导入的包。
- 提供包级文档，帮助团队成员快速建立项目地图。
"""
