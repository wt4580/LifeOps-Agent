# LifeOps-Agent (Web)

一个可运行的个人生活助手智能体 Demo：FastAPI + SQLite + Jinja2 单页 Web UI + Qwen（DashScope OpenAI 兼容模式）。

本项目重点展示：
- 带“可审计 trace”的 ReAct 路由（LLM 决策 -> 工具调用 -> LLM 总结）
- Human-in-the-loop 待办写入（先确认，再入库）
- 本地多格式知识库索引与可追溯引用（PDF/TXT/PNG OCR）

---

## 项目结构（目录树）

> 以 `lifeops_agent/` 为项目根目录。

```
lifeops_agent/
  README.md
  requirements.txt
  run.ps1
  .gitignore
  .env.example
  .env                     # 本机私有配置（不要提交）
  data/
    lifeops.db             # SQLite 数据库（运行时自动创建，不要提交）

  lifeops/
    __init__.py
    main.py                # FastAPI 入口：路由 + ReAct 编排 + trace
    settings.py            # 环境变量配置（.env）
    db.py                  # SQLAlchemy engine/session + init_db
    models.py              # ORM 表结构：chat/todo/memory/doc_chunks/summary

    llm_qwen.py            # OpenAI SDK 调 Qwen（计时日志）
    agent_router.py        # ReAct Router：让 LLM 决定调用哪些工具
    planner.py             # 待办草案生成（LLM JSON）+ Human-in-the-loop
    time_parser.py         # 时间解析：相对日兜底 + LLM 解析复杂表达
    memory_extractor.py    # 对话结构化记忆抽取（候选）

    retrieval.py           # 索引 / 检索 / RAG 生成（带 citations）

    ingest/
      __init__.py
      scanner.py           # 递归扫描 docs_dir，分发给不同解析器
      pdf_text.py          # PDF 提取文本（保留页码）
      txt_text.py          # TXT 直接读取
      ocr_png.py           # PNG OCR（pytesseract）

    templates/
      index.html            # 单页 UI（左侧功能区 + 右侧 trace 面板）

    static/
      app.js                # 极简前端逻辑：fetch API + 渲染
      style.css             # 样式（含右侧 trace sticky 布局）
```

---

## 每个文件是做什么的？（按模块解释）

### 1) Web 层 / 应用入口

- `lifeops/main.py`
  - FastAPI 应用入口。
  - **核心职责**：把一次 `/api/chat` 请求编排成“智能体流水线”。
    - 把用户消息写入 `chat_messages` 表
    - 调用 `agent_router.route_decision()` 让 LLM 决策下一步动作（ReAct）
    - 若需要工具：执行数据库查询（如 `query_todos`）或进入 Human-in-the-loop 待办流程
    - 最后把 assistant 回复写入 DB
    - 返回 `trace`，方便你在网页右侧审计这次请求经历了什么
  - 还包含：
    - `/api/plan/confirm`：把草案写入 `todo_items`
    - `/api/todos/today` & `/api/todos/upcoming`：待办查询
    - `/api/index` & `/api/ask`：索引与 RAG 问答

- `lifeops/templates/index.html`
  - 单页 UI：聊天、待办确认、索引、问答。
  - 右侧的 Trace 面板用于展示 /api/chat 返回的可审计轨迹。

- `lifeops/static/app.js`
  - 前端 JS：用 `fetch()` 调用后端 API。
  - 负责渲染：聊天记录、proposal 草案、todos、index 结果、ask citations、trace。

- `lifeops/static/style.css`
  - 样式文件。
  - 重点：页面左右布局，右侧 trace 面板使用 `position: sticky` 跟随滚动。

### 2) 配置 / 数据库

- `lifeops/settings.py`
  - 用 `pydantic-settings` 读取 `.env`。
  - 配置项包括：
    - `QWEN_API_KEY`、`QWEN_MODEL`、`BASE_URL`
    - `DATABASE_URL`（建议用固定路径，避免相对路径串库）
    - `DOCS_DIR`、`TESSERACT_CMD` 等

- `lifeops/db.py`
  - SQLAlchemy engine/session 工厂。
  - `init_db()` 会在启动时创建表。

- `lifeops/models.py`
  - ORM 表结构：
    - `ChatMessage`：持久化对话
    - `TodoItem`：确认写入的待办
    - `MemoryCandidate`：从对话中抽取的结构化记忆候选
    - `DocumentChunk`：知识库分块
    - `ConversationSummary`：对话摘要（跨重启中期记忆）

### 3) LLM 能力层（Qwen + ReAct）

- `lifeops/llm_qwen.py`
  - 通过官方 `openai` SDK，以 DashScope OpenAI 兼容模式调用 Qwen。
  - 记录 LLM 调用耗时日志（便于性能排查）。

- `lifeops/agent_router.py`
  - ReAct Router：
    - 输入：用户消息 + signals + summary + 最近对话
    - 输出：**严格 JSON**（Pydantic 校验）
      - `action`：normal_chat/query_todos/ask_user_confirm_proposal/propose_todo
      - `args`：工具参数
      - `trace`：可审计的决策理由（不是 CoT）
  - 目的：让“模型真正参与决定要不要调用工具”。

- `lifeops/planner.py`
  - `propose_plan(text)`：让 LLM 把一句话变成待办 JSON（items[{title,due_at}])
  - Human-in-the-loop：
    - 生成草案 ≠ 写入 todo
    - 只有用户点确认才写入 `todo_items`

- `lifeops/time_parser.py`
  - 时间解析模块。
  - 对“今天/明天/后天/大后天”使用确定性规则兜底（避免错一天）。
  - 对更复杂表达交给 LLM 输出 `YYYY-MM-DD`。

- `lifeops/memory_extractor.py`
  - 从最近对话抽取 “会议/DDL/生日/待办等” 结构化候选。
  - 输出必须符合 JSON schema（Pydantic 校验），失败即丢弃，不影响主流程。

### 4) 知识库索引 / RAG

- `lifeops/retrieval.py`
  - `/api/index`：扫描本地目录 -> 文本抽取 -> chunk -> 存 `document_chunks`
  - `/api/ask`：关键词检索 topK -> 把 chunks 作为 context -> 调用 LLM 回答
  - 返回 citations：file_path/page/snippet/score（用于可追溯）

- `lifeops/ingest/scanner.py`
  - 递归扫描 `DOCS_DIR`，识别文件类型并分发给对应解析器。

- `lifeops/ingest/pdf_text.py`
  - 用 `pypdf` 提取每页文字并保留页码。

- `lifeops/ingest/txt_text.py`
  - 按文本读取 `.txt`。

- `lifeops/ingest/ocr_png.py`
  - 用 Pillow + pytesseract 做 OCR（支持 `TESSERACT_CMD` 指定路径）。

---

## 数据流（你可以怎么理解这个智能体）

以 `/api/chat` 为例（ReAct）：

1) 用户发消息 -> 写入 `chat_messages`
2) Router（LLM）决策：这句话是“查询安排”还是“新增待办”还是“普通聊天”
3) 若需要工具：
   - 查询就查 DB 返回事实
   - 新增待办就进入 Human-in-the-loop（先问要不要生成草案）
4) 把工具结果交回给 LLM 生成最终自然语言回复
5) assistant 回复写回 `chat_messages`
6) 返回 `trace`，前端右侧展示“这一轮做了什么”

---

## Quick start（Windows）

1) 创建并激活 venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) 安装依赖

```powershell
pip install -r requirements.txt
```

3) 配置 `.env`

复制 `.env.example` 为 `.env`，填入：
- `QWEN_API_KEY=...`

4) 启动

```powershell
uvicorn lifeops.main:app --reload
```

打开： http://127.0.0.1:8000

---

## 安全说明

- `.env`、`data/`、`.venv/` 都应在 `.gitignore` 中忽略。
- 不要把真实 `QWEN_API_KEY` 提交到仓库。

---

## API 文档（MVP）

> Base URL：默认 `http://127.0.0.1:8000`
>
> 说明：所有 API 都是 JSON（除了 `/` 返回 HTML）。

### 0) GET /health

- 用途：健康检查
- 响应：

```json
{ "status": "ok" }
```

---

### 1) GET /

- 用途：渲染单页 Web UI（Jinja2 模板）
- 响应：HTML

---

### 2) POST /api/chat

- 用途：对话入口（ReAct 编排的核心接口）
  - 先落库用户消息
  - 调用 Router（LLM）决定下一步动作（查询待办 / 进入待办确认 / 普通聊天）
  - 必要时调用工具（DB 查询 / 生成草案）
  - 生成 assistant 回复并落库
  - 返回可审计 trace

#### Request Body

```json
{
  "message": "我明天要去理发",
  "session_id": "可选：uuid"
}
```

- `session_id`：前端通常保存在 localStorage。首次可不传，后端会生成并返回。

#### Response Body

```json
{
  "answer": "我可以为你设置提醒。需要我为你生成待办草案吗？回复“需要/好/可以”即可。",
  "session_id": "9b7b2b2c-...",
  "proposal_id": null,
  "proposal": null,
  "used_tool": "plan_gate",
  "trace": {
    "steps": [
      {"type": "input", "user_message": "我明天要去理发", "session_id": "..."},
      {"type": "llm", "name": "router", "input": {"signals": ["明天"], "recent_dialogue_len": 0}, "output": {"action": "ask_user_confirm_proposal", "args": {"text": "我明天要去理发"}}},
      {"type": "tool", "name": "ask_user_confirm_proposal", "input": {"text": "我明天要去理发"}}
    ]
  }
}
```

#### 重要行为说明（Human-in-the-loop）

- 当用户输入像是“新增待办”的内容时：
  - 本接口不会直接写入 todo
  - 而是先返回 `answer` 询问用户是否生成草案

- 当用户回复“需要/好/可以”等肯定词时：
  - 本接口会返回 `proposal_id` + `proposal`（草案 JSON）
  - 前端需要让用户点击“确认加入待办”，再调用 `/api/plan/confirm` 才会入库

#### 常见错误

- 500：通常是 `.env` 缺失 `QWEN_API_KEY` 或网络/模型调用失败

---

### 3) POST /api/plan/propose（保留接口，可选）

- 用途：显式把一句话直接变成待办草案（不入库）
- 注意：当前 UI 主要通过 `/api/chat` 的意图识别触发，不必须使用这个接口。

#### Request

```json
{ "text": "下周三下午三点开会" }
```

#### Response

```json
{
  "proposal_id": "uuid",
  "proposal": {
    "items": [
      {"title": "开会", "due_at": "2026-03-11T15:00:00"}
    ]
  }
}
```

---

### 4) POST /api/plan/confirm

- 用途：把草案写入 `todo_items`（真正入库动作）
- Human-in-the-loop：只有用户确认才调用它

#### Request

```json
{ "proposal_id": "uuid" }
```

#### Response

```json
{ "inserted": 1 }
```

#### 常见错误

- 404：`proposal_id not found (maybe server restarted)`
  - 因为 `proposal_cache` 是内存缓存，服务重启会丢失草案。

---

### 5) GET /api/todos/today

- 用途：查询“今天”的待办（仅 `due_at` 非空）

#### Response

```json
{
  "items": [
    {"id": 1, "title": "理发", "due_at": "2026-03-05T09:00:00"}
  ],
  "summary": "今天你需要完成 1 项待办：理发。"
}
```

> `summary` 为可选：当前实现会调用 LLM 对清单做一句话总结。

---

### 6) GET /api/todos/upcoming?days=7

- 用途：查询未来 N 天待办（默认 7 天，仅 `due_at` 非空）

#### Query

- `days`：int，可选，默认 7

#### Response

```json
{
  "items": [
    {"id": 1, "title": "和王总开会", "due_at": "2026-03-11T15:00:00"}
  ]
}
```

---

### 7) POST /api/index

- 用途：扫描本地文档目录 `DOCS_DIR`（递归）并写入 `document_chunks`
- 支持文件：PDF/TXT/PNG(OCR)
- 去重规则：同 path + page + chunk_hash 则跳过

#### Request

无 body。

#### Response

```json
{ "inserted": 18, "skipped": 0 }
```

字段含义：
- `inserted`：新增写入的 chunk 数量
- `skipped`：本次扫描中，因去重而跳过的 chunk 数量

---

### 8) POST /api/ask

- 用途：知识库问答（RAG）
  - 先从 `document_chunks` 检索 topK
  - 再把 chunks 作为 context 调用 LLM 生成回答
  - 返回 citations 供前端展示（可追溯）

#### Request

```json
{ "question": "work.png 里讲了什么？" }
```

#### Response

```json
{
  "answer": "...（会带 [1][2] 这样的引用标记）",
  "citations": [
    {
      "path": "E:/work/agent/docu/work.png",
      "page": null,
      "snippet": "...（截断片段）",
      "score": 3.0
    }
  ]
}
```

#### 常见错误

- `No relevant context found.`：检索没有命中（可能需要先运行 `/api/index`）

---

## API 快速验证（PowerShell 示例）

> 下面示例假设服务运行在 8000 端口。

### 1) 聊天 + 触发待办草案

```powershell
$sid = [guid]::NewGuid().ToString()
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/chat -ContentType 'application/json' -Body (@{message='大后天去理发'; session_id=$sid} | ConvertTo-Json)
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/chat -ContentType 'application/json' -Body (@{message='需要'; session_id=$sid} | ConvertTo-Json)
```

### 2) 确认写入 todo

把上一步返回的 `proposal_id` 复制进来：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/plan/confirm -ContentType 'application/json' -Body (@{proposal_id='YOUR_PROPOSAL_ID'} | ConvertTo-Json)
```

### 3) 查询未来待办

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/todos/upcoming?days=7
```

### 4) 索引与问答

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/index
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/ask -ContentType 'application/json' -Body (@{question='stage2 是什么？'} | ConvertTo-Json)
```
