# LifeOps-Agent

一个面向个人效率与生活管理的智能体项目，基于 `FastAPI + LangGraph + SQLite + Qwen`，提供：

- 自然语言聊天与任务规划
- 人在回路（HITL）待办确认
- 本地文档知识库问答（RAG）
- 结构化个人记忆抽取与复用

适合用作：

- 个人智能助手原型
- 智能体工程实践样例（路由、工具调用、状态持久化）
- 本地知识增强问答系统（含证据引用）

---

## 核心能力

- **多轮对话编排**：LangGraph 管理对话流程，支持工具路由与回退。
- **待办草案确认**：先生成草案，再由用户确认入库，降低误操作风险。
- **待办状态管理**：支持今日/7日查询、完成状态勾选、分组展示。
- **知识库问答**：支持 `bm25 / langchain / hybrid` 检索策略，返回 citations。
- **记忆能力**：抽取 `life_events`、`user_profiles` 等结构化信息用于后续对话。
- **状态分层**：
  - Checkpointer：跨轮会话状态与中断恢复
  - 业务数据库：待办、画像、事件、摘要
  - 知识库数据库：文档分块与检索索引

---

## 技术架构

### 后端

- `FastAPI`：HTTP API 与页面入口
- `LangGraph`：聊天状态机、工具调用、HITL 逻辑
- `SQLAlchemy + SQLite`：业务数据存储
- `langgraph-checkpoint-sqlite`：会话状态持久化

### RAG

- 文档解析：`PDF / TXT / DOCX / PNG(OCR)`
- 检索后端：`bm25`、`langchain`、`hybrid`
- 可选重排：`bge-reranker-base`

### 前端

- 原生 `HTML + JS + CSS`
- 支持流式回复展示、思考摘要折叠、待办草案确认卡片

---

## 项目结构

```text
LifeOps-Agent/
  app/
    src/
      common/         # 配置、响应体、全局异常
      controller/     # API 路由层
      domain/         # dto/entity/vo
      service/        # 业务服务层
      util/           # 智能体编排、RAG、时间解析等
      static/         # 前端资源
      templates/      # 页面模板
      main.py         # FastAPI app 对外入口
      __main__.py     # python -m app.src 启动入口
    docs/             # 架构与开发文档
  requirements.txt
  README.md
```

---

## 快速开始（Windows PowerShell）

### 1) 创建并激活虚拟环境

```powershell
cd C:\develop\code\wt\LifeOps-Agent
python -m venv agent_env
.\agent_env\Scripts\Activate.ps1
```

### 2) 安装依赖

```powershell
pip install -r requirements.txt
```

### 3) 配置环境变量

当前配置读取路径是：`app/src/common/env/.env`（由 `base_config.py` 指定）。

创建目录和配置文件：

```powershell
New-Item -ItemType Directory -Force app\src\common\env
New-Item -ItemType File -Force app\src\common\env\.env
```

建议最小配置示例：

```dotenv
QWEN_API_KEY=your_api_key
QWEN_MODEL=qwen-turbo
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 存储
DATABASE_URL=sqlite:///data/lifeops.db
KNOWLEDGE_DATABASE_URL=sqlite:///data/knowledge_base.db
CHECKPOINTER_DB_PATH=data/checkpointer.db

# RAG
RAG_BACKEND=bm25
RAG_TOP_K=5
RAG_CANDIDATE_K=24
RAG_VECTOR_DIR=data/vector/faiss
RAG_EMBED_MODEL=BAAI/bge-small-zh-v1.5
RAG_RERANK_MODEL=BAAI/bge-reranker-base
RAG_USE_RERANK=1
```

> 提示：模型权重建议本地下载，不建议提交到 Git 仓库。

### 4) 启动服务

```powershell
python -m app.src
```

或：

```powershell
uvicorn app.src.main:app --host 0.0.0.0 --port 5000 --reload
```

访问：`http://127.0.0.1:5000`

---

## 使用流程（建议）

1. 打开首页聊天，输入你的计划或问题。
2. 若命中待办意图，系统会生成待办草案并等待你确认。
3. 点击确认后写入待办数据库，可在今日/7日视图查看。
4. 执行文档索引后，可在知识问答中获得带引用的回答。

---

## API 速览

- `GET /health`：健康检查
- `POST /api/chat`：标准聊天接口
- `POST /api/chat/stream`：SSE 流式聊天接口
- `POST /api/plan/propose`：生成待办草案
- `POST /api/plan/confirm`：确认草案并入库
- `GET /api/todos/today`：今日待办（含完成/未完成分组）
- `GET /api/todos/upcoming?days=7`：未来待办查询
- `POST /api/todos/{todo_id}/status`：更新待办完成状态
- `POST /api/index`：知识库索引
- `POST /api/ask`：知识库问答

---

## 开发与扩展建议

- **接入新模型**：优先在 `app/src/common/config/llm_config.py` 统一管理。
- **扩展工具路由**：修改 `app/src/util/agent_router.py` 与 `app/src/util/graph_chat.py`。
- **增加业务能力**：按 `controller -> service -> domain` 分层新增，便于维护。
- **前端交互优化**：集中在 `app/src/static/app.js` 和 `app/src/static/style.css`。

---

## 常见问题

- **`httpx` 兼容报错**：请按 `requirements.txt` 重新安装依赖。
- **OCR 不生效**：检查本机 Tesseract 安装与 `TESSERACT_CMD` 配置。
- **问答命中差**：先 `POST /api/index?rebuild=1` 重建索引，再调整 `RAG_TOP_K` 与阈值。

---

## 安全与提交规范

- 不提交 `.env`、数据库文件、向量索引、模型权重。
- 不在代码中硬编码 API Key。
- 生产环境建议收紧 CORS 与日志内容。

---

## 路线图（可选）

- 多用户鉴权与租户隔离
- 更细粒度的记忆检索策略
- 更完整的前端组件化与测试体系
