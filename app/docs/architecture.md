# LifeOps-Agent 架构图（旧版 vs LangGraph 新版）

> 本文档可直接放到仓库中用于讲解系统设计，包含：
> 1) 时序图（Sequence）
> 2) 状态图（State Machine）
> 3) 数据表流向图（Data Flow）
>
> 图示重点文件：`lifeops/main.py`、`lifeops/graph_chat.py`、`lifeops/agent_router.py`、`lifeops/planner.py`、`lifeops/retrieval.py`。

---

## 1. 总览：旧版与新版编排差异（框 + 箭头）

```mermaid
flowchart LR
    subgraph OLD[旧版：Router + if/else 编排（main.py）]
      O1[用户输入 /api/chat] --> O2[main.py::chat 保存 user 消息]
      O2 --> O3{detect_affirmation?}
      O3 -- 是 --> O4[propose_plan 生成草案]
      O3 -- 否 --> O5[route_decision LLM 选 action]
      O5 --> O6{if/elif 分支执行工具}
      O6 --> O7[保存 assistant 消息 + 摘要 + 记忆抽取]
    end

    subgraph NEW[新版：LangGraph 状态机编排（graph_chat.py）]
      N1[用户输入 /api/chat] --> N2[main.py 保存 user 消息]
      N2 --> N3[run_chat_graph]
      N3 --> N4[load_context]
      N4 --> N5[hitl_check]
      N5 --> N6{pending + affirmation?}
      N6 -- 是 --> N7[propose_from_pending]
      N6 -- 否 --> N8[decide LLM]
      N8 --> N9[run_tool]
      N7 --> N10[finalize]
      N9 --> N10
      N10 --> N11[main.py 保存 assistant + 摘要 + 记忆抽取]
    end
```

---

## 2. 时序图（Sequence）：从用户输入到回复

### 2.1 新版聊天主链（LangGraph）

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant FE as Web UI (index.html + app.js)
    participant API as main.py:/api/chat
    participant DB as SQLite
    participant G as graph_chat.py (StateGraph)
    participant R as agent_router.py (LLM decide)
    participant T as Tool层（DB查询/Planner）
    participant L as llm_qwen.py (Qwen)

    U->>FE: 输入消息
    FE->>API: POST /api/chat {session_id, message}
    API->>DB: 插入 chat_messages(role=user)
    API->>G: run_chat_graph(session_id, message)

    G->>DB: load_context(最近对话+summary)
    G->>G: hitl_check(是否确认上一轮草案)

    alt 命中确认 + 有 pending_intent
        G->>T: planner.propose_plan(pending_text)
        T->>L: LLM 生成结构化 proposal JSON
        L-->>T: proposal
        T-->>G: proposal_id + proposal
        G-->>API: answer=已生成草案，请确认
    else 普通路径
        G->>R: route_decision(context)
        R->>L: 让 LLM 输出 action/args
        L-->>R: RouterDecision(JSON)
        R-->>G: action

        alt action = query_todos
            G->>DB: 查询 todo_items(时间范围)
            DB-->>G: rows
            G->>L: 对 rows 做自然语言总结
            L-->>G: answer
        else action = ask_user_confirm_proposal/propose_todo
            G->>G: 设置 pending_intents[session_id]
            G-->>API: answer=是否生成待办草案
        else action = normal_chat
            G->>L: 普通聊天回复
            L-->>G: answer
        end
    end

    G-->>API: state(answer/proposal/trace)
    API->>DB: 插入 chat_messages(role=assistant)
    API->>DB: 可选写入 conversation_summaries / memory_candidates
    API-->>FE: ChatResponse(answer, proposal, trace)
    FE-->>U: 展示回复与 trace
```

### 2.2 HITL 写入待办（二阶段）

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant FE as Web UI
    participant C as /api/chat
    participant P as planner.py
    participant PC as proposal_cache(内存)
    participant F as /api/plan/confirm
    participant DB as todo_items

    U->>FE: 明天去理发
    FE->>C: POST /api/chat
    C-->>FE: 需要我生成待办草案吗？

    U->>FE: 需要
    FE->>C: POST /api/chat
    C->>P: propose_plan(pending_text)
    P-->>C: proposal_id + proposal
    C->>PC: 缓存草案
    C-->>FE: 返回 proposal（仍未写库）

    U->>FE: 点击“确认加入待办”
    FE->>F: POST /api/plan/confirm {proposal_id}
    F->>PC: 读取 proposal
    F->>DB: 插入 todo_items
    F-->>FE: {inserted: N}
```

---

## 3. 状态图（State Machine）：LangGraph 聊天图

```mermaid
stateDiagram-v2
    [*] --> LoadContext
    LoadContext --> HitlCheck

    HitlCheck --> ProposeFromPending: detect_affirmation && pending_text存在
    HitlCheck --> Decide: 其他情况

    Decide --> RunTool

    RunTool --> Finalize: query_todos
    RunTool --> Finalize: ask_user_confirm_proposal/propose_todo
    RunTool --> Finalize: normal_chat

    ProposeFromPending --> Finalize
    Finalize --> [*]
```

### 状态说明

- `LoadContext`：加载最近对话与会话摘要。
- `HitlCheck`：检查“确认词 + 待确认意图”是否同时成立。
- `ProposeFromPending`：从 pending 文本生成草案（不直接写入 todo）。
- `Decide`：调用 Router LLM 决策 action。
- `RunTool`：根据 action 执行工具。
- `Finalize`：兜底 answer、补 trace，回传给 `main.py`。

---

## 4. 数据表流向图（Data Flow）

```mermaid
flowchart TB
    U[用户消息] --> A[/api/chat]
    A --> B[(chat_messages)]

    A --> C[LangGraph 执行]
    C --> D{action}

    D -->|query_todos| E[(todo_items)]
    D -->|propose_todo| F[(pending_intents 内存)]
    D -->|normal_chat| G[LLM 直接回复]

    C --> H[answer/proposal/trace]
    H --> I[(chat_messages: assistant)]

    I --> J[_maybe_update_summary]
    J --> K[(conversation_summaries)]

    I --> L[memory_extractor]
    L --> M[(memory_candidates)]

    subgraph RAG[知识库链路（独立于 /api/chat）]
      X[/api/index] --> Y[扫描 docs_dir + OCR/PDF/TXT]
      Y --> Z[(document_chunks)]
      Q[/api/ask] --> S[search_chunks 三路召回]
      S --> Z
      S --> T[rag_answer]
      T --> W[answer + citations]
    end
```

---

## 5. 关键设计结论（面试可直接讲）

1. 新版将“流程控制”从 `main.py` 的分支逻辑上移到 `LangGraph`，提升可维护性。
2. HITL 被建模为显式状态转移：未确认只到 proposal，不会写入 `todo_items`。
3. trace 从“结果日志”升级为“节点级执行轨迹”，更易审计与定位问题。
4. 聊天编排链路与 RAG 链路解耦：`/api/chat` 与 `/api/ask` 可独立演进。
5. 数据层使用 SQLite：消息、待办、摘要、记忆候选、文档块均可落库追溯。

