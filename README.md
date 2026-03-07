# LifeOps-Agent (Web)

一个可运行的个人生活助手智能体项目：`FastAPI + LangGraph + SQLite + Jinja2`，接入 `Qwen (DashScope OpenAI 兼容模式)`，
支持聊天编排、Human-in-the-loop 待办写入、结构化记忆抽取、本地知识库 RAG（含可追溯引用）。

---

## 当前能力概览

- 持续聊天：对话落库（`chat_messages`），前端保留 `session_id`
- ReAct 路由：`LLM Router -> Tool -> LLM/Result`，并返回可审计 `trace`
- Human-in-the-loop：待办只能“先草案，再用户确认入库”
- 记忆抽取：聊天后自动抽取 `memory_candidates`（候选态）
- 个人记忆：结构化存储 `life_events` + `user_profiles`，支持跨会话记忆范围配置
- 待办查询：今日 / 未来 N 天
- 本地知识库：递归索引 `PDF/TXT/PNG(OCR)`，`ask` 返回 `citations`
- 可切换检索后端：`bm25 | langchain | hybrid`
- 来源感知 RAG：`DocumentChunk` 记录 `source_type/doc_topic`，支持来源过滤减少误归因
- 证据门控：支持“证据可信度阈值 + 无证据拒答模板”

---

## 项目结构（以 `lifeops_agent/` 为根）

```text
lifeops_agent/
  README.md                    # 子目录文档入口（指向仓库根 README）
  requirements.txt             # Python 依赖清单
  run.ps1                      # Windows 启动脚本（可选）
  .gitignore                   # Git 忽略规则
  .env.example                 # 环境变量模板
  .env                         # 本机私有配置（不要提交）
  data/                        # 运行期数据目录
    lifeops.db                 # SQLite 数据库文件

  lifeops/                     # 后端核心包
    __init__.py                # 包说明与模块导入入口
    main.py                    # FastAPI 入口：路由 + 状态编排 + API 输出
    settings.py                # 配置中心：读取 .env
    db.py                      # 数据库引擎/会话工厂/建表初始化（含轻量列迁移）
    models.py                  # ORM 表模型定义

    llm_qwen.py                # Qwen 调用封装（OpenAI 兼容）
    agent_router.py            # LLM Router：决策调用哪个工具
    graph_chat.py              # LangGraph 状态机：聊天主流程编排
    planner.py                 # 待办草案生成与确认/拒绝识别
    time_parser.py             # 时间解析（规则兜底 + LLM）
    memory_extractor.py        # 对话结构化记忆抽取
    personal_memory.py         # 个人事件/画像抽取与主动建议决策

    retrieval.py               # 索引/检索/RAG 主入口（bm25/langchain/hybrid）
    rag_langchain.py           # 向量检索链路（FAISS + 可选重排）

    ingest/                    # 多格式文档解析层
      __init__.py              # ingest 包说明
      scanner.py               # 递归扫描并按类型分发解析
      pdf_text.py              # PDF 按页提取文本
      txt_text.py              # TXT 文件读取
      ocr_png.py               # PNG 预处理 + OCR 文本提取

    templates/
      index.html               # 单页前端模板（Jinja2）

    static/
      app.js                   # 前端交互逻辑（fetch + 渲染）
      style.css                # 前端样式（双栏与 trace 面板）

  docs/
    architecture.md            # 架构图与状态机说明
    operation-log.md           # 每日功能操作日志
    dev-retrospective.md       # 研发问题与决策复盘
```

### 补充：每个核心文件是做什么的（恢复详细说明）

#### 1) Web 层 / 应用入口

- `lifeops/main.py`
  - FastAPI 应用入口。
  - 负责把一次 `/api/chat` 请求编排成完整智能体流程：
    1. 写入用户消息
    2. 调用 `graph_chat` 执行状态机
    3. 得到回复/草案/引用/trace
    4. 写入 assistant 消息
    5. 触发摘要更新与记忆抽取（旁路增强）
- `lifeops/templates/index.html`
  - 单页 UI 模板：聊天、草案确认、待办查询、索引、知识库问答、trace。
- `lifeops/static/app.js`
  - 前端事件绑定与 API 调用。
  - 渲染聊天区、待办区、索引结果、引用、trace。
- `lifeops/static/style.css`
  - 页面布局与样式（左侧业务区 + 右侧 sticky trace）。

#### 2) 配置 / 数据库

- `lifeops/settings.py`
  - 从 `.env` 读取运行配置（Qwen、SQLite、OCR、RAG 后端参数）。
  - 包含 RAG 证据门控配置（`RAG_EVIDENCE_MIN_SCORE`、`RAG_NO_EVIDENCE_TEMPLATE`）。
- `lifeops/db.py`
  - 创建 SQLAlchemy `engine` 与 `SessionLocal`。
  - 启动时确保 SQLite 路径存在并建表。
  - 对旧库做 `document_chunks` 轻量列迁移（补 `source_type/doc_topic`）。
- `lifeops/models.py`
  - ORM 表定义：
    - `ChatMessage`
    - `TodoItem`
    - `MemoryCandidate`
    - `LifeEvent`
    - `UserProfile`
    - `DocumentChunk`（含 `source_type/doc_topic`）
    - `ConversationSummary`

#### 3) LLM 与智能体编排

- `lifeops/llm_qwen.py`
  - 封装 Qwen 调用（OpenAI SDK 兼容方式）与耗时日志。
- `lifeops/agent_router.py`
  - Router 决策模块：让 LLM 输出结构化 action/args/trace。
- `lifeops/graph_chat.py`
  - LangGraph 状态机主流程（load_context -> hitl_check -> decide -> run_tool -> finalize）。
  - 支持强制动作（`/api/ask` 走 `query_knowledge`）。
- `lifeops/planner.py`
  - 草案生成、确认/拒绝识别、日期兜底合并。
- `lifeops/time_parser.py`
  - 时间解析：相对日规则兜底 + LLM 复杂表达解析。
- `lifeops/memory_extractor.py`
  - 对话后抽取结构化记忆候选（schema 校验失败则丢弃）。
- `lifeops/personal_memory.py`
  - 个人事件与画像信息抽取，主动建议待办与日程决策。

#### 4) 检索与索引

- `lifeops/retrieval.py`
  - 索引构建入口（SQLite + 可选向量索引）
  - 检索入口（`bm25/langchain/hybrid`）
  - 证据融合（含 RRF）
  - 来源过滤（按 `doc_topic` 场景过滤）
  - RAG 回答拼接与引用标签生成
  - 证据阈值门控与无证据拒答
- `lifeops/rag_langchain.py`
  - LangChain 向量链路实现：
    - 文档加载（PDF/TXT/MD/DOCX/PNG OCR）
    - 分块（Recursive splitter）
    - FAISS 持久化
    - 可选 cross-encoder 重排

#### 5) 文档解析 ingest

- `lifeops/ingest/scanner.py`：递归扫描并分发 PDF/TXT/PNG
- `lifeops/ingest/pdf_text.py`：PDF 按页提取文本
- `lifeops/ingest/txt_text.py`：TXT 读取
- `lifeops/ingest/ocr_png.py`：图片预处理 + OCR

---

### 补充：请求的数据流（恢复详细解释）

以 `/api/chat` 为例：

1. 前端发送 `{message, session_id}`
2. 后端先把 user 消息写入 `chat_messages`
3. 执行 `run_chat_graph`：
   - 读最近对话与摘要
   - HITL 检查（是否确认/拒绝草案）
   - Router 决策 action
   - 执行工具（`query_todos` / `query_knowledge` / `plan_gate` / `normal_chat`）
4. 得到 `answer/proposal/citations/trace`
5. 把 assistant 消息写入 `chat_messages`
6. 旁路更新摘要与记忆候选
7. 返回 JSON 给前端并渲染

---

## 环境要求

- Windows（已按该环境开发）
- Python 3.10+
- Tesseract OCR（若要启用 PNG OCR）
- Qwen API Key（仅放 `.env`）

---

## 快速启动（Windows PowerShell）

```powershell
cd E:\pycharm\PythonProject\lifeops\lifeops_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env` 至少填入：

```dotenv
QWEN_API_KEY=你的key
QWEN_MODEL=qwen-turbo
```

启动服务：

```powershell
uvicorn lifeops.main:app --reload
```

打开：`http://127.0.0.1:8000`

---

## 检索后端配置（关键）

`.env` 示例：

```dotenv
# bm25 | langchain | hybrid
RAG_BACKEND=hybrid
RAG_TOP_K=5
RAG_CANDIDATE_K=24
RAG_VECTOR_DIR=data/vector/faiss
RAG_EMBED_MODEL=BAAI/bge-small-zh-v1.5
RAG_RERANK_MODEL=BAAI/bge-reranker-base
RAG_USE_RERANK=1
RAG_CHUNK_SIZE=600
RAG_CHUNK_OVERLAP=120
RAG_EVIDENCE_MIN_SCORE=0.02
RAG_NO_EVIDENCE_TEMPLATE=当前证据不足以支持确定结论。我可以继续按关键词重检，或请你提供更具体的文件名/术语。

# 个人记忆范围
PERSONAL_MEMORY_SCOPE=global
PERSONAL_MEMORY_GLOBAL_ID=default_user
```

切换后建议重建索引：

```powershell
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/api/index?rebuild=1"
```

回退到稳定 BM25：

```dotenv
RAG_BACKEND=bm25
```

---

## API 速览

### `GET /health`

```json
{"status":"ok"}
```

### `POST /api/chat`

请求：

```json
{"message":"我后天要做什么？","session_id":"可选uuid"}
```

响应（字段会按本轮动作变化）：

```json
{
  "answer": "...",
  "session_id": "...",
  "proposal_id": null,
  "proposal": null,
  "used_tool": "query_todos",
  "citations": [
    {
      "path": "...",
      "page": 1,
      "snippet": "...",
      "score": 0.12,
      "reason": "rewrite-bm25",
      "source_type": "pdf",
      "doc_topic": "guideline"
    }
  ],
  "trace": {"meta": {}, "steps": []}
}
```

说明：
- 若命中待办意图且未确认：通常返回 `used_tool=plan_gate`
- 若用户确认后生成草案：会返回 `proposal_id + proposal`
- 若命中知识库工具：会返回 `citations`（含 `source_type/doc_topic`）

### `POST /api/plan/confirm`

```json
{"proposal_id":"..."}
```

说明：
- 这是唯一“把草案真正写入 todo_items”的接口。
- 由于服务重启导致内存草案丢失，会返回 404。

### `GET /api/todos/today`
### `GET /api/todos/upcoming?days=7`

说明：
- 两个接口都按时间排序返回。
- `today` 接口可能附带 LLM 一句话 `summary`。

### `POST /api/index`
- 支持 `?rebuild=1`

说明：
- `rebuild=1` 会重建索引（用于 OCR 参数或分块策略变更后的刷新）。
- 在 `langchain/hybrid` 模式会额外构建向量索引。

### `POST /api/ask`

```json
{"question":"work.png里面有什么"}
```

返回：`answer + citations + trace`

说明：
- `/api/ask` 复用 LangGraph，通过 `forced_action=query_knowledge` 走统一工具链。

---

## 2 分钟验收流程

1. 打开页面，聊天输入：`周三和李总开会，下午三点`
2. 模型提示是否建草案，回复：`需要`
3. 点击 `Confirm Add To Todos`
4. 点击 `Upcoming 7 days`，看到新增待办
5. 点击 `Run Index` 建索引
6. 在 `Ask Knowledge Base` 提问，确认出现 `citations`
7. 右侧 `Trace` 可看到 `input -> router -> tool -> llm/finalize`

---

## 常见问题

- `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`
  - 常见于 `httpx` 版本不匹配，按 `requirements.txt` 重装依赖。

- `TesseractNotFoundError`
  - 配置 `.env` 中 `TESSERACT_CMD` 为 `tesseract.exe` 绝对路径，或确认 PATH 生效后重开终端。

- `No relevant context found.` / 回答与证据不一致
  - 先执行 `/api/index?rebuild=1`
  - 检查 `DOCS_DIR` 是否正确
  - 调低/调高 `RAG_EVIDENCE_MIN_SCORE` 观察拒答与命中平衡
  - 若是个人近况问题，优先检查是否命中了个人记忆而非外部指南

---

## 安全与提交规范

- 不提交 `.env`、`data/`、`.venv/`
- 不在代码中硬编码任何 API Key
- 生产环境请收紧 CORS 与日志脱敏策略
