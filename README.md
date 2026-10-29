<div align="center">

# LifeOps-Agent

### 面向个人事务管理的 AI Agent

自然语言交互 · 多步任务编排 · 人机确认 · 可追溯执行

**FastAPI · LangGraph · SQLite · Qwen · Docker**

[快速开始](#快速开始) · [系统架构](#系统架构) · [测试与评估](#测试与评估) · [部署与持续集成](#部署与持续集成) · [Harness 源码](app/src/common/harness/)

</div>

---

LifeOps-Agent 将待办、记账、日历和知识库整合到统一的对话入口。模型负责理解请求与生成计划，应用负责执行约束、用户确认和数据持久化，通过 SSE 展示处理进度、回答、来源引用与执行轨迹。

项目同时提供真实模型模式与无需 API Key 的离线演示，涵盖 Agent 编排、运行时控制、回归评估和容器化部署。

![LifeOps-Agent 聊天界面](docs/screenshots/chat.png)

## 核心能力

| 能力 | 实现方式 |
| --- | --- |
| **任务编排** | LangGraph 管理多轮状态、JSON 动作路由、多步计划、执行验证与结果汇总 |
| **人机确认** | 待办先生成草案，通过 checkpoint 支持确认、修改和取消；已有账户余额变更须确认 |
| **知识库问答** | BM25 / 向量 / 混合检索，支持 RRF 融合、可选重排和来源引用 |
| **个人记忆** | 保留聊天原文，增量生成摘要，按身份隔离画像、事件与学习文件 |
| **运行控制** | 独立 Harness 管理时间、调用次数和 token 预算，拦截工具请求并记录审计 |
| **工具契约** | 统一注册参数 schema、副作用与确认要求，模型工具目录与执行校验同源，参数拒绝后停止后续计划 |
| **交互与工具** | SSE 流式交互、请求取消、待办、收支、日历、天气与可选本地语音 |

## 系统架构

```mermaid
flowchart LR
    Web[Web Client] --> API[FastAPI / SSE]
    API --> Auth[身份校验与会话队列]
    Auth --> Runtime

    subgraph Runtime[Agent Runtime]
        Graph[LangGraph 编排] --> LLM[模型调用]
        Graph --> Tools[工具执行]
        Graph <--> HITL[草案与用户确认]
        Harness[预算 / 策略 / 审计] -.-> LLM
        Harness -.-> Tools
    end

    HITL <--> CP[(Checkpoint)]
    Tools --> DB[(业务数据)]
    Tools --> RAG[检索与来源引用]
    Graph --> Result[回答 / Trace / 用量摘要]
    Result --> API
```

### 编排与确认

路由层将请求转换为 `{action, args}`，复杂任务生成多步计划，由程序分派工具并汇总执行结果。同一会话通过进程内 FIFO 队列串行处理，降低并发请求覆盖 checkpoint 的风险。

队列默认最多接收全进程 32 个任务、单会话 8 个任务（均包含运行中与排队中）。超限通过 SSE `error` 返回可重试提示，不启动该任务；空闲会话自动释放队列与 worker。关闭服务时停止接收新任务，默认排空 15 秒，随后取消剩余任务并等待工具退出，再关闭 checkpoint；这些容量限制仅对单进程有效。

主动停止会把原问题、计划进度、未完成输出和已提交操作回执存入业务库。下一条补充条件会携带这份中断上下文供模型重新规划；原请求重试返回已中止结果。停止不会撤销已提交写入，也不保证恢复模型内部推理。

待办使用 LangGraph `interrupt()` 暂停执行，保存草案后等待用户操作：

```text
描述任务 → 生成草案 → 用户修改 / 确认 / 取消 → 恢复执行 → 返回结果
```

确认后写入待办，取消则结束该流程。已有账户余额变更也需要确认；收支记录等操作遵循各自业务规则。

### 运行时 Harness

Harness 独立于模型可修改的图状态，对模型与工具调用施加统一约束：

- **预算控制**：限制墙钟时间、LLM / 工具调用次数、token 和图超级步数。
- **工具策略**：动作白名单、只读 SQL 校验、会话作用域视图与 SQLite authorizer。
- **参数边界**：[统一工具契约](app/src/domain/tool_registry.py)生成模型可见目录，校验金额、日期、枚举、查询上限及未知字段；身份由服务端注入，内部确认动作不对模型开放。
- **可观测性**：记录耗时、用量来源、异常和预算状态，输出 `harness_summary` 与元数据审计。
- **前后台隔离**：学习、记忆和文档摘要使用独立预算；等待用户确认不占用执行时间。

供应商返回 usage 时记录实际用量，缺失时标记并采用保守预留量控制预算。详细实现见 [Harness 模块](app/src/common/harness/)。

### 检索与记忆

默认 BM25 无需向量模型；可选混合链路采用 **BM25 + 向量召回 → RRF 融合 → 交叉编码器重排**，保留文档路径、页码和章节信息，供回答引用。

关键词检索使用现有 SQLite 数据库中的 **FTS5 持久化索引**，沿用中文相邻双字、英文连续字符的分词规则。文档变更通过数据库触发器记录待更新项，在入库后及查询前分批同步；查询只加载命中的候选原文，不在 Python 中保存全量 chunk 或构建全量 BM25 对象。已有知识库首次检索会自动补建索引，需要 SQLite 运行时支持 FTS5。

FTS5 按全库词频统计计算 BM25，再应用来源与摘要筛选，分数与原 `rank_bm25` 实现不直接可比；界面相关度是词面匹配的展示分，不代表回答正确率。重排与真实问题的检索质量仍需独立评估。

原始对话持续留存，摘要按游标增量更新。个人画像、事件和学习文件按身份隔离，并为历史内容附加时间信息，帮助模型区分背景记录与当前安排。

## 快速开始

### Docker 演示

在已启动 Docker 的环境中，于仓库根目录执行：

```bash
docker compose up -d --build
```

访问 **http://127.0.0.1:5001**，使用 `demo` / `lifeops-demo-2026` 登录。

> 演示使用规则模拟模型、BM25 和仓库内虚构资料，无需 API Key 或模型下载；HTTP/SSE、图执行、身份校验和数据库操作均实际运行。语音与后台学习默认关闭，天气使用虚构数据。

| 体验步骤 | 输入示例 | 观察结果 |
| --- | --- | --- |
| 创建草案 | `明天下午3点开会` | 展示草案，尚未写入待办 |
| 修改安排 | `可以改成后天吗` | 更新草案日期 |
| 确认或取消 | `确认` / `取消` | 写入待办或结束流程 |
| 查询结果 | `查询待办` | 查看当前待办 |
| 知识库问答 | `报销需要哪些材料` | 回答附带虚构制度来源 |

```bash
# 验证部署后的 HTTP/SSE、确认流程、引用与执行摘要
docker compose exec -T lifeops-demo python scripts/smoke.py --url http://127.0.0.1:5000

# 停止服务，保留数据卷
docker compose down
```

演示数据使用专用 volume。如需切换构建源，可设置 `PIP_INDEX_URL`；演示端口通过 `LIFEOPS_PORT` 配置。Windows Docker Desktop 可按本机环境显式指定 `docker --context desktop-linux compose ...`。

### 真实模型模式

建议使用 Python **3.11**。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
Copy-Item app/src/common/env/.env.example app/src/common/env/.env
```

编辑 `.env`，设置 `QWEN_API_KEY`、实际 `DOCS_DIR`，并保持 `DEMO_MODE=false`、`DEMO_SEED=false`。按需修改 `QWEN_MODEL` 与 `BASE_URL`，随后启动：

```powershell
python -m app.src
```

访问 http://localhost:5000。首次注册须从服务本机回环地址访问，并设置至少 8 位密码；后续使用同一账号登录。身份由 HttpOnly Cookie 校验。

可选向量检索与语音依赖位于 `requirements-ml.txt`，需要另外准备模型；PNG OCR 还需 Tesseract。完整配置见[环境变量模板](app/src/common/env/.env.example)。

## 测试与评估

以下为 **2026-10-27** 的验证快照：

| 验证层级 | 结果 | 覆盖范围 |
| --- | --- | --- |
| 单元与回归测试 | **285 项通过** | 编排、业务、认证、隔离、预算与日期处理 |
| 端到端流程契约 | **100 / 100 通过** | HTTP/SSE、图、checkpoint 与隔离 SQLite |
| Docker Compose | **冒烟通过** | 就绪、登录、确认/取消、来源引用与执行摘要 |
| kind | **部署及冒烟通过** | 独立命名空间、单副本、PVC 与实际 HTTP 流程 |

2026-10-29 补充验证：当前源码在 Linux 测试容器中 **391 项单元与回归测试、100/100 条模拟端到端流程契约通过**；前端 **13 项测试通过**，跨进程恢复及关联演练 **12 个检查点通过**。覆盖 FTS5 同步、工具参数拒绝、队列容量、请求幂等、数据恢复、HTTP/Agent/审计关联与监控鉴权。本次使用已有测试镜像只读挂载源码，未重建或更新运行中的 Docker / kind 部署。

[查看流程报告](app/eval/reports/e2e-contract.md) · [查看逐例结果与源码指纹](app/eval/reports/e2e-contract.json)

流程契约使用模拟模型，包含 10 类场景各 10 个变体，覆盖确认、取消、修改、多日期、记账、账户变更、引用、权限拦截、预算耗尽和供应商故障。**通过率反映程序契约，不代表真实模型准确率；报告延迟也不代表线上网络或模型性能。**

```bash
# 在隔离 Linux 镜像中复跑
docker build --target test -t lifeops-agent:test .
docker run --rm lifeops-agent:test
docker run --rm lifeops-agent:test python -m app.eval.run_e2e
```

真实模型回归须配置 Key 并显式启用付费调用：

```bash
python -m app.eval.run_e2e --mode real --allow-paid-calls --output artifacts/e2e-real.json
```

真实模式运行 80 条流程用例，排除模拟故障与模拟跨用户 SQL 的专用场景。真实模型保留集与组件消融评估仍待补充。原有[检索报告](app/eval/reports/retrieval.md)和[路由报告](app/eval/reports/task.json)依赖外部语料，作为历史记录保留。

## 部署与持续集成

| 组件 | 配置入口 | 说明 |
| --- | --- | --- |
| Docker | [Dockerfile](Dockerfile) | 多阶段构建，区分运行、测试与可选 ML 目标，非 root 用户运行 |
| Compose | [compose.yaml](compose.yaml) | 独立数据卷、只读根文件系统、本地回环端口与健康检查 |
| Kubernetes | [kind 配置](deploy/kind/demo.yaml) | 单副本 Deployment、Service、PVC 与探针 |
| Jenkins | [Jenkinsfile](Jenkinsfile) | 构建、检查、测试、报告归档、冒烟与可选 kind 部署 |
| GitHub Actions | [ci.yml](.github/workflows/ci.yml) | 容器验证与报告归档 |

Docker / kind 已完成本地验证。流水线配置已提供，Jenkins 执行节点仍需接入；可选 ML 镜像目标需单独验证。kind 使用已有 `dev` 集群和独立的 `lifeops-demo` 命名空间；Jenkins 需要具备 Docker CLI 与 daemon 访问权限的 Linux 执行节点。

## 运行诊断

每个 HTTP 请求返回服务端生成的 `X-Trace-ID`，结构化请求日志记录该标识、路由模板、状态码和完整响应耗时（包含 SSE 流和排队时间）。Agent 审计与 `harness_summary` 同时携带 HTTP 标识、`turn_id` 和哈希化的请求回执键。重试有新的 HTTP 标识，但复用原回执；缓存回答里的执行轨迹仍属于最初那次执行。

设置独立的 `OPS_METRICS_TOKEN` 后，可通过 Bearer 认证读取 Prometheus 文本格式指标；默认未配置时 `/metrics` 返回 404。Compose 会从同名环境变量传入令牌。普通用户登录 Cookie 不授予监控权限。

```bash
curl -H "Authorization: Bearer $OPS_METRICS_TOKEN" http://127.0.0.1:5001/metrics
```

| 指标 | 用途 |
| --- | --- |
| `lifeops_http_requests_total` / `lifeops_http_request_duration_seconds` | HTTP 状态和完整响应耗时分布，按路由模板聚合 |
| `lifeops_chat_active_jobs` / `lifeops_chat_pending_jobs` / `lifeops_chat_capacity` | 运行中、排队中任务及容量 |
| `lifeops_chat_rejections_total` / `lifeops_chat_queue_wait_seconds` | 全局容量、会话容量、停机拒绝，以及入队等待分布 |
| `lifeops_chat_jobs_total` / `lifeops_agent_turns_total` | 区分请求处理完成、取消、工具拒绝、预算耗尽与执行失败 |
| `lifeops_agent_calls_total` / `lifeops_agent_call_duration_seconds` | 工具与模型调用状态、耗时分布 |
| `lifeops_agent_charged_tokens_total` | 区分供应商报告的 token 与缺失 usage 时的预算预留量 |

指标是单进程内存聚合，重启会归零，需由外部采集器持久化；HTTP 200 不代表 SSE 中的 Agent 任务成功，应同时观察任务和 turn 指标。标签不包含用户名、会话 ID、问题、工具参数或任意 URL，持久化审计仍写入 `HARNESS_TRACE_DIR`。

## 备份与恢复

先停止所有应用进程和定时任务，再使用 [快照工具](scripts/backup.py)备份业务库、知识库、checkpoint 与个人记忆。三份 SQLite 之间没有分布式事务，`--service-stopped` 是操作者对已停服的明确声明，脚本不会自动停止服务；不应把此入口用于在线跨库一致性备份。

```bash
# 替换成实际配置的路径；output 的父目录须已存在，目标目录须不存在。
python -m scripts.backup create --business-db /data/lifeops.db --knowledge-db /data/knowledge.db --checkpoint-db /data/checkpointer.db --memory-dir /data/memory --output /backups/snapshot-001 --service-stopped
python -m scripts.backup verify /backups/snapshot-001
python -m scripts.backup restore /backups/snapshot-001 --output /data-restored
```

快照使用 SQLite backup API，包含 WAL 中已提交的数据，并附文件大小、SHA-256 与完整性检查。恢复只写入全新目录，不覆盖现有数据。恢复后，将数据库和个人记忆配置指向新目录，保留旧目录用于回退，再启动服务。快照包含个人数据与认证信息，应存放在受控目录；不包含环境密钥、原始参考文档、模型权重或诊断日志，部署时需分别保留这些资源。

```bash
# 全程使用临时目录与虚构资料；两个独立应用进程验证恢复，无需 API Key。
python -m scripts.recovery_drill --output artifacts/recovery-drill.json
```

演练覆盖恢复后登录、原请求回执重放不重复记账、待确认 checkpoint 恢复并创建一次待办、脱离原始文档的持久化 FTS 检索和个人记忆文件恢复。Jenkins 与 GitHub Actions 的容器验证已接入该演练。

## 项目结构

```text
app/
├── src/
│   ├── agent/              # 图编排、规划、时间解析与记忆提取
│   ├── common/harness/     # 预算、拦截器、SQL 作用域与审计
│   ├── common/config/      # 配置、模型客户端、数据库与 checkpoint
│   ├── common/static/      # Web 交互与静态资源
│   ├── common/templates/   # Web 页面模板
│   ├── controller/         # SSE、认证、业务与健康检查接口
│   ├── service/            # 业务服务、后台学习与定时任务
│   ├── domain/             # 数据模型、ORM 实体与图状态
│   ├── demo/               # 模拟模型与虚构数据初始化
│   └── util/               # 文档加载、OCR 与检索
└── eval/                   # 流程契约、模型评估与公开报告
examples/knowledge/         # 可复现的虚构语料
scripts/                    # 部署冒烟工具
tests/                      # 单元与回归测试
deploy/kind/                # Kubernetes 配置
```

## 设计边界与路线

当前采用单进程、单副本 SQLite 和进程内会话队列，适用于个人部署及本地演示。`Recreate` 更新期间会短暂停机；多副本运行需要外部数据库、共享 checkpoint、分布式协调和跨进程幂等机制。

后续重点：

- 建立真实模型保留集，覆盖歧义、否定、跨任务对话与工具失败，统计任务成功率、实际 token 和响应延迟。
- 固定语料与参数，对 BM25、向量、混合检索和重排进行消融比较。
- 完善进程异常退出恢复、数据库版本迁移和备份恢复演练，并为工具执行失败统一错误语义。

本地密钥、数据库、个人记忆、模型权重和运行产物不纳入版本控制。
