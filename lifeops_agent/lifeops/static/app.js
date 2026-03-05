// 页面初始化：先拿到所有会用到的 DOM 节点。
const chatLog = document.getElementById("chat-log");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");

const planOutput = document.getElementById("plan-output");
const planConfirm = document.getElementById("plan-confirm");

const todosToday = document.getElementById("todos-today");
const todosUpcoming = document.getElementById("todos-upcoming");
const todosOutput = document.getElementById("todos-output");

const indexRun = document.getElementById("index-run");
const indexOutput = document.getElementById("index-output");

const askInput = document.getElementById("ask-input");
const askSend = document.getElementById("ask-send");
const askAnswer = document.getElementById("ask-answer");
const askCitations = document.getElementById("ask-citations");

const traceOutput = document.getElementById("trace-output");

// 会话 ID：
// - 通过 localStorage 持久化在浏览器本地；
// - 刷新页面后仍能保持同一 session，后端可继续关联上下文。
let sessionId = localStorage.getItem("lifeops_session_id");
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem("lifeops_session_id", sessionId);
}

// proposalId 用于记录“当前待确认草案”的唯一标识。
// 初始没有草案，所以确认按钮默认禁用。
let proposalId = null;
planConfirm.disabled = true;

// 渲染待办草案区域，并启用确认按钮。
function renderProposal(proposal, id) {
  if (!proposal || !proposal.items || proposal.items.length === 0) return;
  proposalId = id;
  planOutput.textContent = JSON.stringify(proposal, null, 2);
  planConfirm.disabled = false;
}

// 把任意对象转成可展示字符串，避免 trace 渲染时因类型差异报错。
function _pretty(obj) {
  if (obj === undefined) return "";
  if (typeof obj === "string") return obj;
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

// 渲染右侧 trace 面板：把后端返回的步骤按时间顺序展开。
function renderTrace(trace) {
  if (!traceOutput) return;
  if (!trace) {
    traceOutput.textContent = "";
    return;
  }

  const lines = [];
  const meta = trace.meta || {};
  if (Object.keys(meta).length > 0) {
    lines.push(`[trace] graph=${meta.graph || "-"} started_at=${meta.started_at || "-"}`);
  }

  // steps 里每一项通常代表一个节点：input/router/tool/llm/finalize。
  const steps = Array.isArray(trace.steps) ? trace.steps : [];
  steps.forEach((step, idx) => {
    const stepName = step.name ? `${step.type}:${step.name}` : step.type;
    const ts = step.ts ? ` @ ${step.ts}` : "";
    lines.push(`\n${idx + 1}. ${stepName}${ts}`);

    if (step.user_message) {
      lines.push(`   input.user_message: ${step.user_message}`);
    }
    if (step.input !== undefined) {
      lines.push(`   input: ${_pretty(step.input)}`);
    }
    if (step.output !== undefined) {
      lines.push(`   output: ${_pretty(step.output)}`);
    }
  });

  // 同时保留原始 JSON，便于开发者完整排查。
  lines.push("\n--- raw trace json ---");
  lines.push(_pretty(trace));
  traceOutput.textContent = lines.join("\n");
}

// 渲染知识库引用（citations）。
function renderCitations(citations) {
  askCitations.innerHTML = "";
  (citations || []).forEach((cite) => {
    const div = document.createElement("div");
    div.className = "citation";
    const reason = cite.reason ? ` | reason: ${cite.reason}` : "";
    div.textContent = `${cite.path} | page: ${cite.page ?? "-"} | score: ${cite.score}${reason} | ${cite.snippet}`;
    askCitations.appendChild(div);
  });
}

// 往聊天框追加一行消息。
function appendChat(role, text) {
  const div = document.createElement("div");
  div.className = "chat-line";
  div.textContent = `${role}: ${text}`;
  chatLog.appendChild(div);
  // 自动滚动到最新消息。
  chatLog.scrollTop = chatLog.scrollHeight;
}

// Chat 发送：
// 1) 把用户消息先显示到 UI；
// 2) 请求 /api/chat；
// 3) 渲染 assistant 回复 + trace + 可选草案。
chatSend.addEventListener("click", async () => {
  const message = chatInput.value.trim();
  if (!message) return;
  appendChat("user", message);
  chatInput.value = "";
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  const data = await res.json();
  appendChat("assistant", data.answer);
  renderTrace(data.trace);
  if (Array.isArray(data.citations) && data.citations.length > 0) {
    renderCitations(data.citations);
  }

  // 如果本轮触发了 HITL 草案，显示草案并允许点击确认。
  if (data.proposal_id && data.proposal) {
    appendChat("assistant", "检测到待办意图，已生成草案，请确认加入待办。\n" + JSON.stringify(data.proposal, null, 2));
    renderProposal(data.proposal, data.proposal_id);
  }
});

// 点击“确认加入待办”：调用 /api/plan/confirm 真正写入数据库。
planConfirm.addEventListener("click", async () => {
  if (!proposalId) return;
  const res = await fetch("/api/plan/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ proposal_id: proposalId }),
  });
  const data = await res.json();
  planOutput.textContent = `Inserted: ${data.inserted}`;
  // 提交后清空当前确认状态，防止重复提交。
  planConfirm.disabled = true;
});

// 查询今日待办。
todosToday.addEventListener("click", async () => {
  const res = await fetch("/api/todos/today");
  const data = await res.json();
  todosOutput.textContent = JSON.stringify(data, null, 2);
});

// 查询未来 7 天待办。
todosUpcoming.addEventListener("click", async () => {
  const res = await fetch("/api/todos/upcoming?days=7");
  const data = await res.json();
  todosOutput.textContent = JSON.stringify(data, null, 2);
});

// 执行文档索引（会扫描 DOCS_DIR 并更新知识库）。
indexRun.addEventListener("click", async () => {
  const res = await fetch("/api/index", { method: "POST" });
  const data = await res.json();
  indexOutput.textContent = JSON.stringify(data, null, 2);
});

// 独立知识库问答入口：调用 /api/ask，渲染答案与引用。
askSend.addEventListener("click", async () => {
  const question = askInput.value.trim();
  if (!question) return;
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await res.json();
  askAnswer.textContent = data.answer;
  renderCitations(data.citations);
});
