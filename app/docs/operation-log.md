# LifeOps-Agent 操作日志

> 目标：按日期记录项目实际完成内容，并在每个“成功功能”完成后立即追加，保证过程可追溯。

## 记录规则

- 每天至少 1 条记录（建议收工前补齐）。
- 每成功完成一个功能，立即新增一条“功能更新”记录。
- 每条记录必须包含：时间、目标、改动文件、验收结果。
- 记录事实，不写空话；路径用相对路径，便于快速定位。

## 记录模板（复制使用）

```markdown
## YYYY-MM-DD

### [HH:MM] 功能更新：<功能名称>
- 目标：
- 改动：
- 涉及文件：
  - `path/to/file`
- 验收：
- 结果：成功/失败
- 备注：
```

---

## 2026-03-06

### [当前] 功能更新：建立项目操作日志机制
- 目标：创建统一文档，记录每日工作与功能完成情况。
- 改动：新增日志文档并定义记录规范与模板。
- 涉及文件：
  - `docs/operation-log.md`
- 验收：文档已创建，可直接按模板追加当天和后续记录。
- 结果：成功
- 备注：从本条开始执行“每完成一个功能就更新日志”。

### [当前] 功能更新：修复 hybrid 向量索引的 DOCX 依赖问题
- 目标：解决 `/api/index` 在 `RAG_BACKEND=hybrid` 下出现 `No module named 'docx2txt'`。
- 改动：补齐 `docx2txt` 依赖；DOCX loader 改为按需导入，缺依赖时仅跳过 `.docx` 并记录 warning，不阻塞其他文件建索引。
- 涉及文件：
  - `requirements.txt`
  - `lifeops/rag_langchain.py`
  - `README.md`
- 验收：代码层已消除“全量导入导致整体失败”的风险；重新安装依赖后可重跑 `/api/index` 验证 `vector_error` 消失。
- 结果：成功
- 备注：这次修改是“补依赖 + 运行时降级兜底”双保险。

### [当前] 功能更新：graph_chat 状态机注释增强
- 目标：让执行顺序与 LLM 调用点一眼可见，降低新人阅读门槛。
- 改动：补充模块总览时序、节点分流说明、LLM 触发条件与工具分支行为注释。
- 涉及文件：
  - `lifeops/graph_chat.py`
- 验收：静态检查无报错，注释已覆盖“先走谁后走谁 + 何时调 LLM”的关键路径。
- 结果：成功
- 备注：后续可按相同风格继续补 `agent_router.py` 与 `retrieval.py` 的调用链注释。

### [当前] 功能更新：记忆力模块增强（个人事件 + 主动建议 + 聊天压缩）
- 目标：让系统记住用户饮食/活动/研究等事实并在聊天中给出主动建议，同时控制聊天记录体量。
- 改动：新增 `life_events` 表与个人事件抽取/建议模块；在 `/api/chat` 接入事件落库与建议拼接；消息超过阈值时执行摘要并删除最旧聊天记录。
- 涉及文件：
  - `lifeops/models.py`
  - `lifeops/personal_memory.py`
  - `lifeops/main.py`
  - `lifeops/settings.py`
  - `.env.example`
- 验收：静态检查通过；主流程不受抽取失败影响；支持配置 `CHAT_RETENTION_LIMIT/CHAT_PRUNE_TARGET`。
- 结果：成功
- 备注：后续可补充 `life_events` 查询 API 与前端画像面板。

### [当前] 功能更新：主动建议 trace 可观测性增强
- 目标：让用户能看清“建议是否生成、依据来自哪里、具体输出了什么”。
- 改动：在 `/api/chat` 增加 `proactive_advice_prepare/proactive_advice/proactive_advice_append` 三段 trace，并记录 recent_events/todos 的计数与示例标题。
- 涉及文件：
  - `lifeops/main.py`
- 验收：静态检查通过；trace 在“有建议/无建议”两种场景均可显示明确结果。
- 结果：成功
- 备注：后续可在前端把 postprocess 节点渲染成可折叠卡片。

### [当前] 功能更新：向量模型本地优先加载 + 远程回退
- 目标：避免每次强依赖外网访问 HuggingFace，提升本地可用性。
- 改动：新增 `RAG_EMBED_MODEL_LOCAL/RAG_RERANK_MODEL_LOCAL` 配置；模型加载流程改为“本地路径优先，失败回退远程模型ID”；日志明确打印当前来源（local/remote）。
- 涉及文件：
  - `lifeops/settings.py`
  - `lifeops/rag_langchain.py`
  - `.env.example`
- 验收：静态检查通过；可在日志中看到 `Model source [embedding|reranker]=local/remote`。
- 结果：成功
- 备注：后续可增加“离线模式开关”避免任何远程探测。

### [当前] 功能更新：/api/index 增加向量索引差分统计
- 目标：让索引结果可判断“是否真的改动”，不再只看总数。
- 改动：在向量构建阶段比较旧索引与新索引，新增返回字段：`vector_prev_chunks`、`vector_added_chunks`、`vector_rewritten_chunks`、`vector_deleted_chunks`、`vector_unchanged_chunks`、`vector_changed`。
- 涉及文件：
  - `lifeops/rag_langchain.py`
- 验收：语法检查通过；`/api/index` 将透传这些字段到响应体。
- 结果：成功
- 备注：改写统计以 `source+page+chunk_pos` 为位置键进行比较。

### [当前] 功能更新：身份画像前置到 chat 流程（先查画像再进 graph）
- 目标：让回答先参考用户身高/体重/喜好/疾病，再进行路由与知识库问答。
- 改动：新增 `user_profiles` 表；新增画像抽取与前置建议逻辑；`/api/chat` 改为“先更新/读取画像 -> 生成 pre_advice -> 注入 graph”；RAG 回答支持使用画像上下文。
- 涉及文件：
  - `lifeops/models.py`
  - `lifeops/personal_memory.py`
  - `lifeops/main.py`
  - `lifeops/graph_chat.py`
  - `lifeops/retrieval.py`
- 验收：编译检查通过；trace 可看到 `preprocess:profile_context`，并能审计画像与前置建议是否参与本轮处理。
- 结果：成功
- 备注：后续可补 `GET/POST /api/profile` 做人工维护和可视化。

### [当前] 功能更新：chat 流程重排（画像/事件前置，B3 事后建议下线）
- 目标：先完成个人信息与个人事件分析，再把前置知识送入状态机执行，避免“先答后补”。
- 改动：`/api/chat` 改为前置执行画像更新、事件抽取落库、前置知识提示生成；通过 `pre_steps + pre_advice` 注入 `run_chat_graph`；移除事后 B3 建议拼接路径。
- 涉及文件：
  - `lifeops/main.py`
  - `lifeops/graph_chat.py`
  - `lifeops/personal_memory.py`
- 验收：编译检查通过；trace 开头可见 `preprocess:profile_context/personal_events/pre_graph_hint`，且主回答不再追加事后“补充建议”段落。
- 结果：成功
- 备注：前置建议由“画像 + recent life_events”生成，作为 graph 的先验知识参与路由与 RAG。

## 2026-03-07

### [当前] 功能更新：事实问句 RAG 优先 + 无命中回退普通聊天
- 目标：让“事实类问题”先检索知识库，避免直接回答“不知道”。
- 改动：
  - 在 graph 路由中新增事实问句识别，命中后优先走 `query_knowledge`。
  - `query_knowledge` 无命中时回退 `normal_chat`，不再硬返回 `No relevant context found.`。
  - Router 提示词同步调整为“事实/文档问题优先检索”。
- 涉及文件：
  - `lifeops/graph_chat.py`
  - `lifeops/agent_router.py`
- 验收：静态检查通过；`chat` 下事实问句可见 `query_knowledge -> fallback/answer` 的 trace 路径。
- 结果：成功
- 备注：保留了可审计 trace，便于后续对比命中与回退效果。

### [当前] 功能更新：个人近况问题路由修正（避免误触发 RAG）
- 目标：解决“我最近饮食怎么样”误查外部指南导致归因错误的问题。
- 改动：
  - 新增个人近况识别守卫（`personal_state_question`）。
  - 对个人近况类提问不强制 RAG，优先走 `normal_chat + 个人记忆上下文`。
  - Router 规则细化：事实/文档问题与个人近况问题分流。
- 涉及文件：
  - `lifeops/graph_chat.py`
  - `lifeops/agent_router.py`
- 验收：静态检查通过；相关问题不再默认进入 `query_knowledge`。
- 结果：成功
- 备注：这是对“RAG-first 过强策略”的定向修复。

### [当前] 功能更新：DocumentChunk 增加来源元数据（source_type/doc_topic）
- 目标：为检索与回答增加“来源感知”，支撑后续来源过滤与可解释性。
- 改动：
  - `document_chunks` 模型新增 `source_type`、`doc_topic` 字段。
  - 索引阶段自动从路径推断来源与主题并写入。
  - 启动时增加 SQLite 轻量迁移：老库自动补列（不依赖 Alembic）。
- 涉及文件：
  - `lifeops/models.py`
  - `lifeops/retrieval.py`
  - `lifeops/db.py`
- 验收：编译检查通过；本地运行时可插入带元数据的 `DocumentChunk` 记录。
- 结果：成功
- 备注：后续可基于 `doc_topic` 做白名单/黑名单检索策略。

### [当前] 功能更新：RAG 证据可信度阈值 + 无证据拒答模板
- 目标：降低“证据弱仍强答”的误归因风险。
- 改动：
  - 新增 `RAG_EVIDENCE_MIN_SCORE` 与 `RAG_NO_EVIDENCE_TEMPLATE` 配置。
  - `rag_answer` 增加门控逻辑：当证据强度不足时直接返回拒答模板。
  - 回答上下文标签增加 `source_type/doc_topic`，增强可解释性与约束能力。
- 涉及文件：
  - `lifeops/settings.py`
  - `.env.example`
  - `lifeops/retrieval.py`
  - `lifeops/graph_chat.py`
- 验收：静态检查通过；证据不足场景会返回拒答模板而不是强行生成结论。
- 结果：成功
- 备注：默认阈值按当前 hybrid 分数分布调为 `0.02`，后续可按真实问答日志微调。

### [当前] 文档更新：研发复盘文档补全（含逐阶段加改删）
- 目标：沉淀从 0 到 1 的完整研发脉络，便于复盘与面试讲述。
- 改动：
  - 新增复盘文档并补充“问题清单 + 决策思路 + 产品研发进程（加/改/删）”。
- 涉及文件：
  - `docs/dev-retrospective.md`
- 验收：文档已完成并可直接用于回顾与对外讲述。
- 结果：成功
- 备注：后续可再产出 1 页面试版精简稿。
