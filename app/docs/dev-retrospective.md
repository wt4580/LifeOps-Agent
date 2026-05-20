# LifeOps-Agent 研发过程复盘（问题与思路回顾）

> 目的：帮助你快速回忆从 0 到 1 研发这个项目时的关键决策、踩坑、排障路径和思维演进。

## 1. 项目目标与架构演进

### 初始目标
- 在 Windows 本地搭建一个可运行、可演示、可写简历的个人生活助手：
  - Web（FastAPI + Jinja2）
  - Qwen（DashScope OpenAI 兼容）
  - 持续对话、待办管理、HITL 计划确认
  - 本地知识库检索（PDF/TXT/PNG OCR）
  - 可追溯引用（citations）

### 架构演进路线
1. 先做可运行 MVP（health + 页面 + chat）
2. 再补 DB 与 todo 闭环
3. 再补 index/ask（RAG）
4. 再做意图识别与 HITL 串联
5. 再迁移到 Graph 状态机（可审计 trace）
6. 再做个人事件/画像/主动建议
7. 再做混合检索（bm25 + vector + rerank）

---

## 2. 你在研发中遇到的核心问题清单（按主题）

## A. 环境与依赖问题

### A1. `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`
- 现象：chat / plan proposal 接口 500。
- 根因：`openai` 与 `httpx` 版本组合不兼容（OpenAI SDK 初始化 http 客户端时参数不匹配）。
- 处理思路：
  - 明确当前环境里的实际版本
  - 通过 requirements 固化兼容版本组合
  - 重装依赖并重启服务验证
- 经验：AI 项目早期最常见风险不是业务逻辑，而是“SDK 版本漂移”。

### A2. 终端/进程状态混乱（端口占用、旧服务残留）
- 现象：改完代码不生效、端口被占用、不同 shell 输出不一致。
- 根因：多终端多环境并行，旧 uvicorn 进程未完全退出。
- 处理思路：先确认 8000 对应进程，再重启单一服务实例。
- 经验：排障第一步永远是“确认你正在打到哪个进程”。

### A3. Git 推送问题（HTTPS 重置 / SSH 公钥）
- 现象：`Recv failure: Connection was reset`、`Permission denied (publickey)`。
- 根因：网络波动 + SSH key/known_hosts 路径异常。
- 处理思路：
  - 分离“网络层问题”和“认证层问题”
  - 单独修复 SSH 配置后再推送
- 经验：不要把“分支问题”和“网络认证问题”混在一起排查。

---

## B. OCR 与索引问题

### B1. `TesseractNotFoundError`（已安装却仍报找不到）
- 现象：命令行 `tesseract --version` 正常，但应用侧 OCR 失败。
- 根因：运行应用的环境变量与当前终端不一致，或 Python 进程未读取新 PATH。
- 处理思路：
  - 重启终端/IDE
  - 使用 `TESSERACT_CMD` 显式指定路径作为兜底
- 经验：Windows 下 PATH 生效边界经常在“进程级别”，不是“机器级别”。

### B2. OCR 对中英文混合/截图文本识别弱
- 现象：`work.png` 内容识别不完整，检索不到。
- 根因：图像质量 + OCR 参数 + 中英混合文本噪声。
- 处理思路：
  - 调整 OCR 语言为 `chi_sim+eng`
  - 改进预处理和参数
  - rebuild index 后再测
- 经验：RAG 问题常常是“输入质量问题”而不是“模型问题”。

### B3. `index` 看起来没有更新
- 现象：新增文档后 `inserted=0`，但总量变化看不清。
- 根因：去重策略（path+page+hash）导致很多 chunk 被判定为重复。
- 处理思路：在返回值里补充向量相关统计（总量/路径），并解释 inserted/skipped 语义。
- 经验：索引接口必须“可解释”，否则用户无法判断是否生效。

---

## C. 时间解析与待办闭环问题

### C1. `due_at` 为空或日期偏差（大后天算错）
- 现象：用户说“明天/大后天”后，待办时间空或错 1 天。
- 根因：纯 LLM 时间解析不稳定，受当前日期上下文影响。
- 处理思路：
  - 引入确定性规则兜底（今天/明天/后天/大后天）
  - 复杂表达再交给 LLM
  - 给 trace 打印解析结果
- 经验：时间语义必须有 deterministic guardrail。

### C2. plan confirm 成功但 `/todos/today/upcoming` 为空
- 现象：`inserted:2`，查询却空。
- 根因：时间过滤窗口、写入字段、会话/全局范围不一致。
- 处理思路：统一写入与查询口径，补齐排序与范围逻辑。
- 经验：闭环要看“写入后可读回”，不是只看写入成功。

### C3. HITL 流程被打断仍可继续生成多个草案
- 现象：草案未确认时又触发新草案，状态看起来“不断点”。
- 根因：当前 `pending_intents` 只管理“口头确认阶段”，不锁“草案确认阶段”。
- 处理思路：澄清状态语义；若要严格串行，需要新增 proposal-level gate。
- 经验：HITL 要分两层状态：intent-confirm 与 proposal-confirm。

---

## D. 路由与 Graph 状态机问题

### D1. “看起来像规则，不像 Agent”
- 现象：很多触发依赖关键词，模型参与感弱。
- 根因：单轮路由 + 单次工具执行，本质是可控工作流。
- 处理思路：
  - 迁移到 Graph，增强可审计 trace
  - 逐步引入模型路由与工具选择
- 经验：先稳定再智能；可控性与智能性要平衡迭代。

### D2. trace 不够清晰
- 现象：只能看到部分步骤，看不懂“为什么这么走”。
- 处理思路：补充节点输入输出、router why、postprocess 标记。
- 经验：trace 本质是“调试产品”，不是装饰。

### D3. 事实类问题不走 RAG
- 现象：问“我的实验室叫什么名字”被判 `normal_chat` 直接说不知道。
- 根因：早期 guard 只覆盖文件/文档句式。
- 处理思路：把事实问句纳入 RAG-first，并保留 miss -> normal fallback。
- 经验：路由规则要区分“事实查询”与“闲聊”。

### D4. 个人近况被误走 RAG
- 现象：问“我最近饮食怎么样”触发知识库检索，污染答案。
- 根因：RAG-first 过强，未区分“个人状态问题”。
- 处理思路：新增 personal-state guard：此类问题优先 normal_chat + 个人记忆。
- 经验：路由是高杠杆点，过强会引入系统性偏差。

---

## E. RAG 质量与语义归因问题

### E1. 检索命中不稳定（`work.png` vs `work`）
- 现象：带扩展名问法命不中，简化词命中。
- 根因：分词与 query rewrite 对文件名处理不稳。
- 处理思路：
  - 增加 path recall
  - query rewrite 保留文件 stem
  - 三路召回融合展示
- 经验：文件类查询必须有 path 通道，不要只靠语义检索。

### E2. “指南菜谱”被当成“你吃过的东西”
- 现象：RAG 把文档建议误归因为用户个人事实。
- 根因：缺少 source-aware 约束与回答提示边界。
- 处理思路：
  - 路由上减少不必要 RAG
  - `rag_answer` 提示词明确“外部知识 != 用户已发生行为”
- 经验：RAG 最大风险是“归因错位”，不是“检索不到”。

### E3. 你提出的更深层问题（切片语义缺失）
- 本质：分段是局部片段，不知道“本文档在干什么”。
- 改进方向：
  1) chunk 附加文档级元信息（文档类型/主题/来源）
  2) 检索后做 rerank + source filtering
  3) 回答时增加“证据来源类型约束”
- 经验：只优化 topK 不够，要优化“证据解释层”。

---

## F. 记忆系统问题

### F1. “短上下文”与“长期记忆”混淆
- 现象：模型像忘了前面说过的话。
- 根因：短期对话窗口有限，且跨会话记忆策略不一致。
- 处理思路：
  - 聊天记录入库 + 摘要压缩
  - 画像/事件做结构化存储
  - 增加 global memory scope
- 经验：记忆要分层：短期窗口、中期摘要、长期结构化事件。

### F2. 全量聊天保留成本过高
- 现象：担心聊天无限增长。
- 处理思路：超过阈值后做“摘要 + 删旧”，保留关键长期信息。
- 经验：记忆不是“全存”，而是“可检索的压缩存”。

---

## 3. 关键技术决策（你当时的研发思路）

1. 先跑通再增强：先功能闭环，再做智能增强。
2. Human-in-the-loop 优先：所有写入类动作必须可控。
3. 时间语义必须兜底：LLM 可解析，规则负责稳定。
4. 可审计优先：trace 先做出来，后续优化有依据。
5. RAG 分层优化：召回、重排、回答约束分开治理。
6. 个人记忆与知识库分治：避免“外部知识污染个人事实”。

---

## 4. 你项目当前的“高价值亮点”

- Graph 编排 + trace 可审计
- HITL 待办写入闭环
- 时间感知个人事件抽取与建议
- 本地多模态文档索引（PDF/TXT/PNG OCR）
- 混合检索与可追溯引用
- 跨会话个人记忆（可配置 scope）

---

## 5. 下一步建议（按收益排序）

1. 给 `DocumentChunk` 增加 source_type / doc_topic 元数据，做来源过滤。
2. rag加入“证据可信度阈值 + 无证据拒答模板”，减少误归因。
3. 把“草案待确认状态”单独建模，实现严格串行 HITL。
4. 做一个记忆管理页（查看/编辑/删除 life_events 与 profile）。
5. 补自动化回归用例：时间解析、路由、RAG 误归因、HITL。

---

## 6. 一句话回顾

你这条研发路线的核心不是“堆功能”，而是逐步把一个能聊的应用，打磨成“可控、可解释、可持续进化”的个人智能体系统。

---

## 7. 产品研发进程（事无巨细版：加了什么 / 改了什么 / 删了什么）

> 注：以下按“功能里程碑”组织，便于你按记忆回放研发过程；每条都标注了新增、修改、删除（或废弃）的内容。

### 阶段 0：从 0 到可启动骨架
- 新增：
  - `FastAPI` 应用入口、`/health`、首页模板渲染与静态资源挂载。
  - 初版目录结构（`main/settings/db/models/ingest/retrieval/templates/static`）。
  - `.env.example`、`.gitignore`、`requirements.txt`、`README` 基础说明。
- 修改：
  - 启动脚本与 Windows 运行说明（`run.ps1`、uvicorn 命令）。
- 删除/废弃：
  - 无（纯新建阶段）。

### 阶段 1：聊天闭环 + 持久化
- 新增：
  - `/api/chat` 接口，支持 `session_id`。
  - `chat_messages` 表，落库 user/assistant 全量消息。
  - 前端 `localStorage` 持久化 `session_id`。
- 修改：
  - 聊天响应结构，统一返回 `answer/session_id`。
  - 聊天异常处理与日志输出。
- 删除/废弃：
  - 无（仅扩展）。

### 阶段 2：待办系统与 HITL 双阶段确认
- 新增：
  - `planner.py`：`detect_plan_intent`、`detect_affirmation`、`propose_plan`。
  - `/api/plan/propose`、`/api/plan/confirm`。
  - `todo_items` 表与 `/api/todos/today`、`/api/todos/upcoming`。
  - 内存态 `proposal_cache` 与 `pending_intents`。
- 修改：
  - 将“新增待办”从直接写库改成：先提案、后确认写库。
  - 前端加入“草案展示 + 确认加入待办”。
- 删除/废弃：
  - 废弃“用户一句话直接入 todo”的早期直写方式（逻辑层面）。

### 阶段 3：时间解析增强（LLM + 规则兜底）
- 新增：
  - `time_parser.py`，支持 `今天/明天/后天/大后天` 确定性解析。
  - 周几与“下周X”日期计算工具。
- 修改：
  - `planner` 中 `due_at` 生成逻辑：先解析日期，再与时间字段合并。
  - trace 输出加入时间解析可观测信息。
- 删除/废弃：
  - 弱化纯 LLM 单点时间解析（改为“LLM + deterministic guardrail”）。

### 阶段 4：本地知识库索引与 RAG
- 新增：
  - `document_chunks` 表、`/api/index`、`/api/ask`。
  - 扫描器支持递归处理 PDF/TXT/PNG(OCR)。
  - 检索返回 `citations(path/page/snippet/score)`。
- 修改：
  - `index` 增加去重（`path + page + hash`）。
  - 前端新增 Ask 区域与引用渲染。
- 删除/废弃：
  - 无（功能新增为主）。

### 阶段 5：OCR 可用性与索引可解释性
- 新增：
  - OCR 环境变量配置：`TESSERACT_CMD`、`OCR_LANG` 等。
  - 索引返回中加入统计字段（inserted/skipped/vector_*）。
- 修改：
  - OCR 参数适配中英文混合场景（`chi_sim+eng`）。
  - 文档说明补充 Windows OCR 排障路径。
- 删除/废弃：
  - 无。

### 阶段 6：从“规则流程”到 Graph 状态机
- 新增：
  - `graph_chat.py`：`load_context -> hitl_check -> decide -> run_tool -> finalize`。
  - `trace` 结构化步骤（input/llm/tool/graph_node/postprocess）。
  - `/api/ask` 复用同一 graph（`forced_action=query_knowledge`）。
- 修改：
  - `main.py` 聊天主链路改为 graph 驱动。
  - 路由器与工具执行分离，提升可维护性。
- 删除/废弃：
  - 废弃部分散落在 `main.py` 的 if/else 路由分支（逐步迁移）。

### 阶段 7：路由能力增强（RAG-first 与回退）
- 新增：
  - 事实问句识别（如“叫什么名字/是谁/单位”）。
  - `query_knowledge` 无命中时 `normal_chat_fallback`。
- 修改：
  - Router 提示词：事实/文档问题优先检索。
  - `graph_chat` 决策逻辑：规则兜底 + LLM 路由双保险。
- 删除/废弃：
  - 废弃“检索无命中直接返回 No relevant context found”的硬失败体验。

### 阶段 8：个人记忆系统（画像 + 事件 + 主动建议）
- 新增：
  - `life_events`、`user_profiles` 数据模型。
  - `extract_personal_events`、`extract_profile_facts`。
  - 主动建议决策：`score/threshold/should_add/reasons`。
- 修改：
  - 聊天链路前置：先更新画像/抽取事件，再进 graph。
  - 回复后置：按阈值决定是否追加“补充建议”。
- 删除/废弃：
  - 弱化纯关键词建议触发（改为“结构化上下文 + LLM 决策 + 阈值门控”）。

### 阶段 9：记忆范围与会话隔离策略
- 新增：
  - `PERSONAL_MEMORY_SCOPE`（`session/global`）。
  - `PERSONAL_MEMORY_GLOBAL_ID`（单用户跨会话聚合）。
- 修改：
  - 画像/事件读写按 owner_id 聚合。
  - 待办读取兼容旧库：有 `session_id` 列则按会话过滤，否则全局。
- 删除/废弃：
  - 废弃“所有场景强会话隔离”的单一策略。

### 阶段 10：建议上下文时间化（不再按固定条数）
- 新增：
  - 事件时间窗读取：`PROACTIVE_EVENT_WINDOW_HOURS`。
  - 事件字段中加入 `effective_time/recorded_at` 用于判断时效。
  - LLM 统一 runtime context（`now_local/now_utc/timezone`）。
- 修改：
  - 从 `recent_events[-N]` 改为“按时间窗口检索事件”。
  - trace 增加 `event_window_start/end` 与 `now_iso`。
- 删除/废弃：
  - 废弃“固定 30 条事件”作为建议输入的做法。

### 阶段 11：RAG 误归因修复（外部知识 != 用户事实）
- 新增：
  - `rag_answer` 提示词约束：外部指南/菜谱不得自动归因为用户已发生行为。
  - 个人近况问题识别（如“我最近饮食怎么样”）避免强制 RAG。
- 修改：
  - Router 规则细化：
    - 事实/文档问题 -> 优先检索
    - 个人近况问题 -> 优先 normal_chat（结合个人记忆）
- 删除/废弃：
  - 废弃“所有事实相关问句都强制 RAG”这一过强策略。

### 阶段 12：检索体系工程化（bm25 -> hybrid）
- 新增：
  - 多路召回：`path + bm25 + rewrite-bm25 + vector`。
  - 融合策略：`merge_rank` / `RRF`。
  - LangChain 向量索引构建与检索回调。
- 修改：
  - `search_chunks` 按 `RAG_BACKEND` 分流。
  - 索引输出附带向量统计信息。
- 删除/废弃：
  - 弱化“只靠单路关键词检索”的旧链路。

### 阶段 13：文档化与可交接能力
- 新增：
  - `docs/architecture.md`（时序图/状态图/数据流）。
  - `docs/operation-log.md`（每日操作记录）。
  - 本文件 `docs/dev-retrospective.md`（问题与思路复盘）。
- 修改：
  - README 补齐配置、运行、排障、功能验证说明。
- 删除/废弃：
  - 废弃分散、不可追溯的“口头记录式”知识传递。

---

## 8. 你在每次迭代中的固定工作模式（可复用）

1. 先定义“用户体感问题”而非技术问题（例如：为什么它说我不知道）。
2. 用 trace 定位是路由错、检索错、还是回答归因错。
3. 最小改动修一层，再做回归验证。
4. 把新增配置同步到 `.env.example` 与 README。
5. 每次改动都补“可审计信息”，保证后续可解释。

---

## 9. 给未来自己的检查清单（下次继续开发前先看）

- [ ] 当前 `RAG_BACKEND` 是什么（bm25 / hybrid / langchain）
- [ ] 索引是否 rebuild 过，文档是否真正入库
- [ ] OCR 环境是否对当前 Python 进程生效
- [ ] `PERSONAL_MEMORY_SCOPE` 是否符合当前测试目标
- [ ] trace 是否能说明“为什么走了这条路径”
- [ ] 回答中的事实是否都能追溯到证据或用户事件
