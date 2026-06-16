// LifeOps Agent - 简化版交互逻辑

// DOM 元素
const chatLog = document.getElementById("chat-log");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");

const todosToday = document.getElementById("todos-today");
const todosUpcoming = document.getElementById("todos-upcoming");
const todosOutput = document.getElementById("todos-output");

// 待办草案相关元素
const proposalCard = document.getElementById("proposal-card");
const proposalContent = document.getElementById("proposal-content");
const closeProposal = document.getElementById("close-proposal");
const confirmProposal = document.getElementById("confirm-proposal");
const rejectProposal = document.getElementById("reject-proposal");

const ragSection = document.getElementById("rag-section");
const closeRag = document.getElementById("close-rag");
const indexRun = document.getElementById("index-run");
const indexOutput = document.getElementById("index-output");
const askInput = document.getElementById("ask-input");
const askSend = document.getElementById("ask-send");
const askAnswer = document.getElementById("ask-answer");
const askCitations = document.getElementById("ask-citations");

let sessionId = localStorage.getItem("lifeops_session_id");
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem("lifeops_session_id", sessionId);
}

let proposalId = null;
let currentTodoView = "upcoming";

// ========================================
// 工具函数
// ========================================
function _pretty(obj) {
  if (obj === undefined) return "";
  if (typeof obj === "string") return obj;
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

function _fmtValue(v, maxLen = 140) {
  if (v === undefined || v === null) return "";
  let s;
  if (typeof v === "string") s = v;
  else {
    try { s = JSON.stringify(v); } catch { s = String(v); }
  }
  s = s.replace(/\s+/g, " ").trim();
  return s.length > maxLen ? s.slice(0, maxLen - 1) + "…" : s;
}

function _fmtDetail(obj, maxLen = 220) {
  if (obj === undefined || obj === null) return "";
  if (typeof obj !== "object") return _fmtValue(obj, maxLen);
  const entries = Object.entries(obj);
  if (entries.length === 0) return "{}";
  const parts = entries
    .map(([k, v]) => `${k}=${_fmtValue(v, 100)}`)
    .filter((p) => !p.endsWith("="));
  const out = parts.join(", ");
  return out.length > maxLen ? out.slice(0, maxLen - 1) + "…" : out;
}

function buildThoughtLines(trace, result) {
  const steps = Array.isArray(trace?.steps) ? trace.steps : [];
  const routeStep = steps.find((step) =>
    ["router", "knowledge_query_guard", "bare_confirmation_guard"].includes(step.name)
  );
  const toolStep = [...steps].reverse().find((step) => step.type === "tool");
  const reasoning = [];

  if (routeStep?.output?.action) {
    reasoning.push(`我先判断这句更适合走「${routeStep.output.action}」流程。`);
  }

  if (toolStep?.name === "query_knowledge") {
    const hitInfo = toolStep.output;
    if (hitInfo?.hits > 0) {
      reasoning.push(`我在知识库检索到了 ${hitInfo.hits} 条相关证据。`);
    } else {
      reasoning.push("知识库没有命中直接证据，我改用通用建议来回答。")
    }
  } else if (toolStep?.name === "propose_from_pending") {
    reasoning.push("我已根据你的确认生成待办草案，等你确认入库。")
  } else if (toolStep?.name === "query_weather") {
    const city = toolStep.output?.city || toolStep.input?.resolved_city || toolStep.input?.city || "未知城市";
    const cached = toolStep.output?.cached ? "（命中缓存）" : "";
    if (toolStep.error) {
      reasoning.push(`查询${city}的高德天气失败：${toolStep.error}。`);
    } else {
      reasoning.push(`我查询了${city}的高德天气实况与预报${cached}。`);
    }
  } else if (toolStep?.name === "query_calendar") {
    const eventsCount = toolStep.output?.events_count;
    if (toolStep.error) {
      reasoning.push(`查询 Google Calendar 失败：${toolStep.error}。`);
    } else if (eventsCount === 0) {
      reasoning.push("我从 Google Calendar 查了这段时间，没有事件。");
    } else if (typeof eventsCount === "number") {
      reasoning.push(`我从 Google Calendar 拉取了 ${eventsCount} 条事件。`);
    } else {
      reasoning.push("我从 Google Calendar 拉取了这段时间的事件。");
    }
  } else if (toolStep?.name) {
    reasoning.push(`本轮执行了「${toolStep.name}」步骤。`);
  }

  if (Array.isArray(result?.citations) && result.citations.length > 0) {
    reasoning.push(`答案引用了 ${result.citations.length} 条知识库片段。`);
  }

  const summary = reasoning.length > 0 ? reasoning : ["我已完成意图判断并生成本轮回答。"];
  const lines = summary.map((text) => ({ text, className: "" }));

  if (steps.length > 0) {
    lines.push({ text: `—— 完整过程（共 ${steps.length} 步） ——`, className: "separator" });
    steps.forEach((step, idx) => {
      const tag = `[${idx + 1}] ${step.type || "?"} / ${step.name || "?"}`;
      const when = step.ts ? `  (${step.ts})` : "";
      lines.push({ text: `${tag}${when}`, className: "detail" });
      if (step.input !== undefined) lines.push({ text: `    输入: ${_fmtDetail(step.input)}`, className: "detail" });
      if (step.output !== undefined) lines.push({ text: `    输出: ${_fmtDetail(step.output)}`, className: "detail" });
      if (step.metadata !== undefined) lines.push({ text: `    元信息: ${_fmtDetail(step.metadata)}`, className: "detail" });
      if (step.error) lines.push({ text: `    错误: ${_fmtValue(step.error, 240)}`, className: "detail error" });
    });
    if (trace?.meta) {
      lines.push({ text: `—— 元数据: ${_fmtDetail(trace.meta)}`, className: "detail" });
    }
  } else {
    lines.push({ text: "—— 本轮未记录详细步骤 ——", className: "separator" });
  }

  return lines;
}

function renderCitations(citations) {
  if (!askCitations) return;
  askCitations.innerHTML = "";
  if (!citations || citations.length === 0) return;
  
  citations.forEach((cite, idx) => {
    const div = document.createElement("div");
    div.className = "citation";
    div.textContent = `${cite.path} (相似度: ${(cite.score * 100).toFixed(0)}%)`;
    askCitations.appendChild(div);
  });
}

function appendChat(role, text) {
  const div = document.createElement("div");
  div.className = `chat-line ${role}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

function appendAssistantTurn() {
  const wrapper = document.createElement("div");
  wrapper.className = "chat-line assistant assistant-turn";

  const thoughtHeader = document.createElement("div");
  thoughtHeader.className = "assistant-thought-header";

  const thoughtLabel = document.createElement("span");
  thoughtLabel.className = "assistant-thought-label";
  thoughtLabel.textContent = "思考";

  const thoughtToggle = document.createElement("button");
  thoughtToggle.type = "button";
  thoughtToggle.className = "assistant-thought-toggle";
  thoughtToggle.textContent = "收起";

  const thoughtDiv = document.createElement("div");
  thoughtDiv.className = "assistant-thought";
  setThoughtLines(thoughtDiv, ["正在分析你的问题..."]);

  const answerDiv = document.createElement("div");
  answerDiv.className = "assistant-answer";
  answerDiv.textContent = "回答：";

  let collapsed = true;
  thoughtToggle.addEventListener("click", () => {
    collapsed = !collapsed;
    thoughtDiv.classList.toggle("collapsed", collapsed);
    thoughtToggle.textContent = collapsed ? "展开" : "收起";
  });

  thoughtHeader.appendChild(thoughtLabel);
  thoughtHeader.appendChild(thoughtToggle);

  wrapper.appendChild(thoughtHeader);
  wrapper.appendChild(thoughtDiv);
  wrapper.appendChild(answerDiv);
  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
  return { wrapper, thoughtDiv, answerDiv, thoughtToggle };
}

function setThoughtLines(thoughtDiv, lines) {
  if (!thoughtDiv) return;
  const safeLines = Array.isArray(lines) && lines.length > 0 ? lines : ["我已完成意图判断并生成本轮回答。"];
  thoughtDiv.innerHTML = "";
  safeLines.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "thought-line";
    if (entry && typeof entry === "object") {
      row.textContent = entry.text ?? "";
      if (entry.className) row.classList.add(...entry.className.split(/\s+/).filter(Boolean));
    } else {
      row.textContent = String(entry);
    }
    thoughtDiv.appendChild(row);
  });
}

// ========================================
// 待办草案功能
// ========================================
function showProposal(proposalId, proposal) {
  if (!proposalCard || !proposalContent) return;
  
  // 保存提案ID
  window.currentProposalId = proposalId;
  
  // 格式化显示草案内容
  let content = "";
  if (Array.isArray(proposal)) {
    proposal.forEach((item, idx) => {
      content += `${idx + 1}. ${item.title}\n`;
      if (item.due_at) {
        const dueDate = new Date(item.due_at);
        content += `   截止时间: ${dueDate.toLocaleString('zh-CN')}\n`;
      }
      content += "\n";
    });
  } else {
    content = _pretty(proposal);
  }
  
  proposalContent.textContent = content;
  proposalCard.classList.remove("hidden");
  
  // 滚动到底部
  setTimeout(() => {
    proposalCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, 100);
}

function hideProposal() {
  if (!proposalCard) return;
  proposalCard.classList.add("hidden");
  window.currentProposalId = null;
}

// 关闭草案卡片
if (closeProposal) {
  closeProposal.addEventListener("click", () => {
    hideProposal();
  });
}

// 确认待办草案
if (confirmProposal) {
  confirmProposal.addEventListener("click", async () => {
    const proposalId = window.currentProposalId;
    if (!proposalId) {
      alert("没有可确认的草案");
      return;
    }
    
    try {
      const res = await fetch("/api/plan/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          proposal_id: proposalId,
          session_id: sessionId 
        }),
      });
      
      const data = await res.json();
      
      if (res.ok) {
        appendChat("assistant", "✅ 待办已成功加入！");
        hideProposal();
        // 自动刷新待办列表
        loadTodos("/api/todos/upcoming?days=7");
      } else {
        appendChat("assistant", `❌ 确认失败: ${data.detail || "未知错误"}`);
      }
    } catch (error) {
      appendChat("assistant", `❌ 网络错误: ${error.message}`);
    }
  });
}

// 拒绝待办草案
if (rejectProposal) {
  rejectProposal.addEventListener("click", () => {
    appendChat("assistant", "已取消待办草案");
    hideProposal();
  });
}

// ========================================
// 追踪面板开关
// ========================================
const toggleRag = document.getElementById("toggle-rag");

if (toggleRag) {
  toggleRag.addEventListener("click", () => {
    if (ragSection) ragSection.classList.toggle("hidden");
    // On mobile, also open the side panel
    var sidePanel = document.getElementById("side-panel");
    var sideOverlay = document.getElementById("side-panel-overlay");
    if (sidePanel && window.innerWidth <= 768) {
      sidePanel.classList.add("open");
      if (sideOverlay) sideOverlay.classList.add("visible");
    }
  });
}

// ========================================
// RAG 区域控制
// ========================================
function showRag() {
  if (ragSection) ragSection.classList.remove("hidden");
}

if (closeRag) {
  closeRag.addEventListener("click", () => {
    if (ragSection) ragSection.classList.add("hidden");
  });
}

// RAG 标签切换
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    const targetTab = document.getElementById(`tab-${tab}`);
    if (targetTab) targetTab.classList.add("active");
  });
});

// ========================================
// 聊天功能
// ========================================
if (chatSend) {
  chatSend.addEventListener("click", async () => {
    const message = chatInput.value.trim();
    if (!message) return;
    
    appendChat("user", message);
    chatInput.value = "";
    const assistantTurn = appendAssistantTurn();
    
    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId }),
      });
      if (!res.ok || !res.body) {
        throw new Error(`请求失败: ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let finalResult = null;
      let assistantText = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          const line = part.split("\n").find((item) => item.startsWith("data: "));
          if (!line) continue;
          const payload = JSON.parse(line.slice(6));

          if (payload.type === "delta") {
            assistantText += payload.delta || "";
            if (assistantTurn?.answerDiv) assistantTurn.answerDiv.textContent = `回答：${assistantText}`;
          } else if (payload.type === "status") {
            if (assistantTurn?.thoughtDiv) {
              let firstLine = assistantTurn.thoughtDiv.querySelector(".thought-line");
              if (!firstLine) {
                setThoughtLines(assistantTurn.thoughtDiv, [payload.message || "正在思考..."]);
              } else {
                firstLine.textContent = payload.message || "正在思考...";
              }
            }
          } else if (payload.type === "final") {
            finalResult = payload.result;
          }
        }
      }

      if (finalResult) {
        const finalAnswer = finalResult.answer || assistantText || "";
        if (assistantTurn?.answerDiv) assistantTurn.answerDiv.textContent = `回答：${finalAnswer}`;
        if (assistantTurn?.thoughtDiv) setThoughtLines(assistantTurn.thoughtDiv, buildThoughtLines(finalResult.trace, finalResult));

        if (finalResult.proposal_id && finalResult.proposal && finalResult.used_tool === "plan_proposal") {
          showProposal(finalResult.proposal_id, finalResult.proposal);
        }

        if (Array.isArray(finalResult.citations) && finalResult.citations.length > 0) {
          showRag();
          const askTabBtn = document.querySelector('[data-tab="ask"]');
          if (askTabBtn) askTabBtn.click();
          if (askAnswer) {
            askAnswer.classList.remove("hidden");
            askAnswer.textContent = finalResult.answer;
          }
          renderCitations(finalResult.citations);
        }
      } else if (assistantTurn?.answerDiv) {
        assistantTurn.answerDiv.textContent = `回答：${assistantText || "未收到回复"}`;
      }
    } catch (error) {
      if (assistantTurn?.thoughtDiv) setThoughtLines(assistantTurn.thoughtDiv, ["请求失败"]);
      if (assistantTurn?.answerDiv) assistantTurn.answerDiv.textContent = `回答：错误: ${error.message}`;
    }
  });
}

if (chatInput) {
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (chatSend) chatSend.click();
    }
  });
}

// ========================================
// 待办查询
// ========================================
function _formatTodoDue(dueAt) {
  if (!dueAt) return "无截止";
  return new Date(dueAt).toLocaleString("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function updateTodoStatus(todoId, completed) {
  const res = await fetch(`/api/todos/${todoId}/status`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ completed, session_id: sessionId }),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "更新状态失败");
  }
  return data;
}

function renderTodoItems(items, { emptyText = "暂无待办", canToggle = true } = {}) {
  if (!todosOutput) return;
  if (!items || items.length === 0) {
    const emptyDiv = document.createElement("div");
    emptyDiv.className = "todo-empty";
    emptyDiv.textContent = emptyText;
    todosOutput.appendChild(emptyDiv);
    return;
  }

  items.forEach((todo) => {
    const row = document.createElement("label");
    row.className = `todo-row ${todo.is_completed ? "completed" : "pending"}`;

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = Boolean(todo.is_completed);
    checkbox.disabled = !canToggle;
    checkbox.className = "todo-checkbox";

    checkbox.addEventListener("change", async (event) => {
      try {
        checkbox.disabled = true;
        await updateTodoStatus(todo.id, event.target.checked);
        if (currentTodoView === "today") {
          await loadTodos("/api/todos/today");
        } else {
          await loadTodos("/api/todos/upcoming?days=7");
        }
      } catch (error) {
        checkbox.checked = !event.target.checked;
        alert(`更新失败: ${error.message}`);
      } finally {
        checkbox.disabled = false;
      }
    });

    const content = document.createElement("div");
    content.className = "todo-content";

    const title = document.createElement("div");
    title.className = "todo-title";
    title.textContent = todo.title;

    const due = document.createElement("div");
    due.className = "todo-due";
    due.textContent = _formatTodoDue(todo.due_at);

    content.appendChild(title);
    content.appendChild(due);

    row.appendChild(checkbox);
    row.appendChild(content);
    todosOutput.appendChild(row);
  });
}

function renderTodayTodos(pending, completed) {
  if (!todosOutput) return;
  todosOutput.innerHTML = "";

  const pendingHeader = document.createElement("div");
  pendingHeader.className = "todo-group-title";
  pendingHeader.textContent = "未完成";
  todosOutput.appendChild(pendingHeader);
  renderTodoItems(pending, { emptyText: "今日无未完成待办" });

  const completedHeader = document.createElement("div");
  completedHeader.className = "todo-group-title";
  completedHeader.textContent = "已完成";
  todosOutput.appendChild(completedHeader);
  renderTodoItems(completed, { emptyText: "今日暂无已完成待办" });
}

async function loadTodos(url) {
  try {
    // 添加session_id参数
    const separator = url.includes('?') ? '&' : '?';
    const urlWithSession = `${url}${separator}session_id=${sessionId}`;
    
    const res = await fetch(urlWithSession);
    const data = await res.json();

    const result = data.result || data.data || {};
    if (url.includes("/api/todos/today")) {
      currentTodoView = "today";
      const pending = result.pending || [];
      const completed = result.completed || [];
      renderTodayTodos(pending, completed);
    } else {
      currentTodoView = "upcoming";
      const todos = result.todos || data.items || [];
      if (todosOutput) {
        todosOutput.innerHTML = "";
      }
      renderTodoItems(todos, { emptyText: "7日内暂无未完成待办" });
    }
  } catch (error) {
    todosOutput.textContent = `加载失败: ${error.message}`;
  }
}

if (todosToday) {
  todosToday.addEventListener("click", () => {
    todosToday.classList.add("active");
    if (todosUpcoming) todosUpcoming.classList.remove("active");
    loadTodos("/api/todos/today");
  });
}

if (todosUpcoming) {
  todosUpcoming.addEventListener("click", () => {
    todosUpcoming.classList.add("active");
    if (todosToday) todosToday.classList.remove("active");
    loadTodos("/api/todos/upcoming?days=7");
  });
}

loadTodos("/api/todos/upcoming?days=7");

// ========================================
// 索引功能
// ========================================
if (indexRun) {
  indexRun.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/index", { method: "POST" });
      const data = await res.json();
      indexOutput.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
      indexOutput.textContent = `错误: ${error.message}`;
    }
  });
}

// ========================================
// 知识库问答
// ========================================
if (askSend) {
  askSend.addEventListener("click", async () => {
    const question = askInput.value.trim();
    if (!question) return;
    
    if (askAnswer) {
      askAnswer.classList.remove("hidden");
      askAnswer.textContent = "查询中...";
    }
    
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      
      const data = await res.json();
      if (askAnswer) askAnswer.textContent = data.answer;
      renderCitations(data.citations);
    } catch (error) {
      if (askAnswer) askAnswer.textContent = `错误: ${error.message}`;
    }
  });
}

if (askInput) {
  askInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (askSend) askSend.click();
    }
  });
}

console.log("✅ LifeOps Agent UI 已加载");
