# LifeOps-Agent

基于 `FastAPI + LangGraph + SQLite + Qwen` 的个人生活管理智能体。通过自然语言对话管理待办、日程、知识库，具备多步任务分解、主动建议和长期记忆能力。

## 功能

- **多轮对话** — LangGraph 状态机编排，支持工具路由、计划分解、HITL 确认
- **多步任务分解** — 复杂请求自动拆分子步骤（如"查天气+查日程+综合回答"），按序执行后汇总；简单请求走快速通道跳过 LLM
- **人在回路（HITL）** — 待办草案先经用户确认再入库，支持`pending_add_confirmation` 双阶段确认（生成草案→确认加入）
- **待办管理** — 创建、查询、完成状态切换、截止日期跟踪；用户说"要"/"需要"即自动确认草案入库
- **日历集成** — 查看日程事件，识别中国传统节日（端午、中秋、春节等），支持多日区间查询
- **天气查询** — 高德天气 API，实况 + 未来预报
- **知识库 RAG** — PDF/TXT/PNG 文档索引，BM25 / 向量 / 混合检索，可选重排序；语义切块 + 章节标题提取 + 块摘要生成
- **个人记忆** — 自动抽取饮食、运动、消费等生活事件；推断隐性偏好与模式（深度反思每 3 轮触发）
- **主动建议** — 基于日历、待办、记忆，在对话中主动提示时间冲突/健康风险等
- **用户画像** — 逐步积累身高体重、饮食运动偏好、健康状况等信息；隐式偏好推断带确信度标记
- **模糊输入识别** — 打错字（如"上一台哦"）通过上下文推断路由到正确的工具

## 架构

```text
用户输入 → ChatService → LangGraph 状态机 → 工具调用 → 应答返回
                              │
                    ┌─────────┼──────────┐
                    │         │          │
                路由决策   多步分解  HITL/确认
                    │         │          │
              query_todos  query_calendar  query_weather
              query_knowledge  normal_chat  propose_todo
              confirm_proposal              ask_user_confirm_proposal
                              │
                         主动建议决策
```

### 数据流

1. `ChatService.chat()` 接收消息 → 保存到 `chat_messages`
2. 抽取画像事实与个人事件 → 更新 `user_profiles` / `life_events`
3. 生成前置建议 → 进入 LangGraph 状态机
4. 图内流程：`load_context → decompose → decide → run_tool → advance_plan → (loop | finalize) → proactive_decision`
5. 保存助手消息 → 更新摘要 → 提取记忆候选 → 可选深度反思
6. 如果 `proposal_confirmed` 标记为真，自动调 `confirm_proposal()` 写入 DB

### 提案确认流程

```
用户: "导员没有回我关于报销的事"
  → Router: ask_user_confirm_proposal
  → _node_run_tool: 调用 propose_plan → 缓存草案 → 设 pending_add_confirmation=True
  → 回复: "已生成待办草案，请确认是否加入待办。"
用户: "要"
  → _node_decide: 检测 pending_add_confirmation + detect_affirmation → action=confirm_proposal
  → _node_run_tool: 设 proposal_confirmed=True
  → chat_service: 自动调 confirm_proposal() 写入 SQLite
  → 回复: "好的，已添加到待办。"
```

## 项目结构

```text
app/src/
├── agent/                    # 智能体层（核心大脑）
│   ├── agent_router.py       # LLM 路由决策（含知识主题感知 + 文档大纲注入）
│   ├── graph_chat.py         # LangGraph 状态机构建与执行
│   ├── memory/
│   │   ├── memory_extractor.py   # 对话记忆提取（confidence/insight_type）
│   │   └── personal_memory.py    # 生活事件/画像抽取、主动建议、深度反思
│   ├── planner/planner.py    # 待办草案规划 + 确认/拒绝检测
│   └── time/time_parser.py   # 自然语言时间解析
├── common/
│   ├── config/               # 配置文件、数据库、LLM、日志
│   │   ├── base_config.py    # pydantic-settings 配置模型
│   │   ├── llm_config.py     # LLM 调用封装（DashScope 兼容）
│   │   ├── db_config.py      # 业务数据库（lifeops.db）含列迁移
│   │   ├── knowledge_db_config.py  # 知识库数据库含列迁移
│   │   ├── chat_graph_holder.py    # 全局图实例持有器
│   │   └── ...
│   ├── env/.env.example      # 环境变量模板（含 HF 镜像说明）
│   ├── static/               # 前端资源
│   └── templates/            # HTML 模板
├── controller/               # API 路由层
├── domain/
│   ├── dto/                  # 数据传输对象
│   └── entity/               # SQLAlchemy ORM 模型
│       ├── chatstate_entity.py     # ChatState TypedDict（所有图字段定义）
│       ├── knowledge_entity.py     # DocumentChunk（含 section 元数据）
│       └── chat_entity.py          # 聊天/待办/记忆/摘要模型
├── service/                  # 业务服务层
│   ├── chat_service.py       # 聊天主服务（含自动确认提案）
│   ├── plan_service.py       # 待办规划服务（proposal 缓存 + 确认入库）
│   ├── calendar_service.py   # 日历服务
│   ├── weather_service.py    # 天气服务
│   └── chinese_holidays.py   # 中国传统节日动态计算
└── util/
    ├── ingest/               # 文档加载与 OCR
    │   ├── scanner.py        # 语义切块 + 章节标题提取 + 表格检测
    │   ├── pdf_text.py       # PDF 文本提取
    │   ├── txt_text.py       # TXT 文本提取
    │   └── ocr_png.py        # PNG OCR
    └── retrieval/            # RAG 检索（BM25 + 向量 + 混合）
        ├── retrieval.py      # 主入口：索引/检索/摘要/大纲/RAG 回答
        └── rag_langchain.py  # LangChain 向量后端
```

## 快速开始

### 环境要求

- Python 3.11+
- 可选：Tesseract OCR（PNG 文字识别）

### 安装

```powershell
git clone <repo-url>
cd LifeOps-Agent
python -m venv agent_env
.\agent_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 配置

复制环境变量模板并编辑：

```powershell
cp app\src\common\env\.env.example app\src\common\env\.env
```

至少需要填写 `QWEN_API_KEY`，其余项有默认值。详细配置说明见 `.env.example` 中的注释。

关键配置项：

| 变量 | 必填 | 说明 |
|------|------|------|
| `QWEN_API_KEY` | 是 | 阿里云 DashScope API Key |
| `QWEN_MODEL` | 否 | 模型名（默认 `qwen-turbo`） |
| `BASE_URL` | 否 | API 地址 |
| `AMAP_WEATHER_KEY` | 否 | 高德天气 Key |
| `DOCS_DIR` | 否 | 知识库文档目录 |
| `HF_ENDPOINT` | 否 | HuggingFace 镜像（国内用 `https://hf-mirror.com`） |
| `RAG_EMBED_MODEL` | 否 | 向量模型（默认 `BAAI/bge-small-zh-v1.5`） |

### 启动

```powershell
python -m app.src
```

访问 `http://localhost:5000`

### 首次使用

1. 打开首页聊天框，输入你的计划或问题
2. 如需知识库问答，在 `.env` 设置 `DOCS_DIR` → 调用 `/api/index` 触发索引（索引后自动生成块摘要）
3. 待办草案生成后说"要"/"好的"/"确认"即自动入库
4. 持续使用后画像和记忆会自动积累，每 3 轮触发深度反思

## API 速览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web 聊天页面 |
| GET | `/health` | 健康检查 |
| POST | `/api/chat` | 标准聊天 |
| POST | `/api/chat/stream` | SSE 流式聊天（实时进度显示） |
| POST | `/api/plan/propose` | 生成待办草案 |
| POST | `/api/plan/confirm` | 确认草案入库 |
| GET | `/api/todos/today` | 今日待办 |
| GET | `/api/todos/upcoming?days=7` | 未来待办 |
| POST | `/api/todos/{id}/status` | 更新完成状态 |
| POST | `/api/index` | 知识库索引（含块摘要生成） |
| POST | `/api/ask` | 知识库问答 |

## 测试

```powershell
python -m pytest tests/ -v
```

测试覆盖 planner、agent_router、graph_chat、memory、time_parser 的核心逻辑及所有 DTO 校验，共 120+ 项测试。LLM 调用在 conftest 中自动 mock，运行测试无需 API Key 或网络。

- `tests/conftest.py` — 全局 mock 所有 `chat_completion` 调用，隔离外部依赖
- `tests/test_planner.py` — 意图检测、确认/拒绝识别、日期解析、标题归一化、草案生成
- `tests/test_graph_chat.py` — 简单查询判断、纯确认文本、相对日期、主动建议决策
- `tests/test_agent_router.py` — RouteContext 构建、路由决策正常/回退路径
- `tests/test_memory.py` — 事件/画像抽取、主动建议决策、对话记忆提取
- `tests/test_time_parser.py` — 日期上下文构建、自然语言日期解析
- `tests/test_chinese_holidays.py` — 中国传统节日离线计算
- `tests/test_dtos.py` — 所有 DTO 的字段校验、默认值、边界条件

## 开发说明

- **新增模型**：在 `common/config/base_config.py` 加配置项，`llm_config.py` 统一调用
- **新增工具**：`agent_router.py` 加路由规则 + `prompt` 描述，`graph_chat.py` 加 `_node_run_tool` 分支
- **新增状态字段**：`domain/entity/chatstate_entity.py` 加 TypedDict 字段，`graph_chat.py` 的 `_build_initial_state` / `_build_checkpoint_context` 同步更新
- **数据库迁移**：新列通过 `ALTER TABLE ADD COLUMN` 在 `init_db()` 中自动补齐（无需 Alembic）
- **RAG 索引流程**：`scanner.py` 负责切块 + 提取元数据 → `retrieval.py` 入库 → `_generate_chunk_summaries()` 批量生成摘要 → `get_available_knowledge_topics()` / `get_document_outline()` 供路由层参考
- **调试日志**：设置 `LOG_LEVEL=DEBUG` 查看 LLM 调用、路由细节和 RAG 检索过程

## .gitignore 覆盖内容

- `.env` 及所有敏感配置
- `data/`（SQLite 数据库、向量索引）
- 模型权重（`bge-*/`、`*.safetensors`）
- `__pycache__`、`.idea`、`.vscode`
- 虚拟环境 `agent_env/`
