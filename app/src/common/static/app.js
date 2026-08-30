// LifeOps Agent - 简化版交互逻辑

// DOM 元素
const chatLog = document.getElementById("chat-log");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");

const todosToday = document.getElementById("todos-today");
const todosUpcoming = document.getElementById("todos-upcoming");
const todosAll = document.getElementById("todos-all");
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
const indexReset = document.getElementById("index-reset");
const indexOutput = document.getElementById("index-output");
const askInput = document.getElementById("ask-input");
const askSend = document.getElementById("ask-send");
const askAnswer = document.getElementById("ask-answer");
const askCitations = document.getElementById("ask-citations");

const loginSection = document.getElementById("login-section");
const loginInput = document.getElementById("login-input");
const loginBtn = document.getElementById("login-btn");
const logoutBtn = document.getElementById("logout-btn");
const appSection = document.getElementById("app-section");

let sessionId = localStorage.getItem("lifeops_session_id") || "";

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

function _stepSummary(step) {
  const name = step.name || "?";
  if (step.error) return `[${name}] 失败: ${_fmtValue(step.error, 100)}`;
  if (name === "query_weather") {
    const city = step.output?.city || step.input?.resolved_city || step.input?.city || "?";
    const cached = step.output?.cached ? " (缓存)" : "";
    return `[天气] ${city}${cached}`;
  }
  if (name === "weather_summarize") {
    return `[天气总结] ${_fmtValue(step.output, 140)}`;
  }
  if (name === "query_knowledge") {
    const hits = step.output?.hits;
    if (hits === 0) return "[知识库] 无匹配";
    return `[知识库] ${hits}条匹配`;
  }
  if (name === "rag_answer" || name === "normal_chat") {
    return `[回答] ${_fmtValue(step.output, 140)}`;
  }
  if (name === "normal_chat_fallback") {
    return `[降级回答] ${_fmtValue(step.output?.answer || step.output, 140)}`;
  }
  if (name === "schedule_summarize") {
    return `[日程总结] ${_fmtValue(step.output, 140)}`;
  }
  if (name === "query_calendar") {
    const cnt = step.output?.events_count;
    return `[日历] ${cnt != null ? cnt + "条事件" : "完成"}`;
  }
  if (name === "query_todos") {
    const items = step.output;
    const count = Array.isArray(items) ? items.length : 0;
    return `[待办] ${count}条`;
  }
  if (name === "propose_from_text" || name === "propose_from_pending") {
    const items = step.output?.items;
    const count = Array.isArray(items) ? items.length : 0;
    return `[待办草案] ${count}项`;
  }
  if (name === "decompose") {
    const plan = step.output?.plan;
    const count = Array.isArray(plan) ? plan.length : 0;
    if (count === 0) return "[分解] 单步可完成";
    const actions = plan.map(s => s.action).join(" → ");
    return `[分解] ${count}步: ${actions}`;
  }
  if (name === "load_context") return `[加载] 已加载上下文`;
  if (name === "profile_context") return `[画像] ${step.output?.profile_changed ? "已变更" : "无变更"}`;
  if (name === "personal_events") return `[事件] ${step.output?.inserted || 0}条新增`;
  if (name === "pre_graph_hint") return `[预分析] 完成`;
  if (name === "pending_reminders") return `[提醒] ${step.output?.count || 0}条待提醒`;
  if (name === "finalize") return `[汇总] 已生成最终回答`;
  if (name === "quality_gate") return `[质量审查] ${step.output?.passed ? "通过" : "已修正"}`;
  if (name === "verify_step") return `[执行审查] ${step.output?.passed ? "通过" : "未通过"}`;
  if (name === "proactive_advice_decision") {
    const added = step.output?.added;
    const score = step.output?.score;
    return `[主动建议] score=${score} ${added ? "已追加" : "不追加"}`;
  }
  if (name === "router_decision") {
    return `[路由] ${step.output?.action || "?"}`;
  }
  if (name === "plan_step") {
    const action = step.output?.action || step.input?.step?.action || "";
    return `[执行计划] ${action}`;
  }
  return `[${name}]`;
}

function buildThoughtLines(trace, result) {
  const steps = Array.isArray(trace?.steps) ? trace.steps : [];
  const lines = [];

  // 第一步：紧凑且有信息量的步骤总结
  const summaryLines = [];
  steps.forEach((step, idx) => {
    const label = step.ts ? step.ts.slice(11, 19) : "";
    summaryLines.push(`${label} ${_stepSummary(step)}`);
  });

  // 引用信息
  const citeCount = Array.isArray(result?.citations) ? result.citations.length : 0;
  if (citeCount > 0) {
    summaryLines.push(`📎 ${citeCount}条引用`);
  }

  if (summaryLines.length > 0) {
    summaryLines.forEach(s => lines.push({ text: s, className: "" }));
  } else {
    lines.push({ text: "我已完成意图判断并生成本轮回答。", className: "" });
  }

  // 第二步：完整过程详情（默认折叠）
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

  // 图标已移到 header，此处不再创建

  const thoughtLabel = document.createElement("span");
  thoughtLabel.className = "assistant-thought-label";
  thoughtLabel.textContent = "思考";

  const thoughtToggle = document.createElement("button");
  thoughtToggle.type = "button";
  thoughtToggle.className = "assistant-thought-toggle";
  thoughtToggle.textContent = "收起";

  const stopBtn = document.createElement("button");
  stopBtn.type = "button";
  stopBtn.className = "assistant-stop-btn";
  stopBtn.textContent = "停止";
  stopBtn.hidden = true;
  let stopPending = false;
  stopBtn.addEventListener("click", () => {
    abortActiveStream();
    if (stopPending) return;
    stopPending = true;
    stopBtn.disabled = true;
    stopBtn.textContent = "停止中…";
    fetch("/api/chat/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    }).catch(() => {});
  });

  const thoughtDiv = document.createElement("div");
  thoughtDiv.className = "assistant-thought collapsed";
  setThoughtLines(thoughtDiv, ["正在分析你的问题..."]);

  const answerDiv = document.createElement("div");
  answerDiv.className = "assistant-answer";
  answerDiv.textContent = "回答：正在生成中...";

  let collapsed = true;
  thoughtToggle.textContent = "展开";
  thoughtToggle.addEventListener("click", () => {
    collapsed = !collapsed;
    thoughtDiv.classList.toggle("collapsed", collapsed);
    thoughtToggle.textContent = collapsed ? "展开" : "收起";
  });

  function expandThought() {
    thoughtDiv.classList.remove("collapsed");
    thoughtToggle.textContent = "收起";
    collapsed = false;
  }

  function collapseThought() {
    thoughtDiv.classList.add("collapsed");
    thoughtToggle.textContent = "展开";
    collapsed = true;
  }

  thoughtHeader.appendChild(thoughtLabel);
  thoughtHeader.appendChild(thoughtToggle);
  thoughtHeader.appendChild(stopBtn);

  wrapper.appendChild(thoughtHeader);
  wrapper.appendChild(thoughtDiv);
  wrapper.appendChild(answerDiv);
  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
  return {
    wrapper,
    thoughtDiv,
    answerDiv,
    thoughtToggle,
    expandThought,
    collapseThought,
    stopBtn,
    setStopVisible(visible) {
      stopBtn.hidden = !visible;
      stopBtn.disabled = stopPending;
      stopBtn.textContent = stopPending ? "停止中…" : "停止";
    },
  };
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
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "确认", session_id: sessionId }),
      });
      if (!res.ok || !res.body) {
        alert("确认失败，请重试");
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        buffer = buffer.split("\n\n").pop() || "";
      }

      appendChat("assistant", "✅ 待办已确认！");
      hideProposal();
      loadTodos("/api/todos/upcoming?days=7");
    } catch (error) {
      alert(`确认失败: ${error.message}`);
    }
  });
}

// ========================================
// 提醒功能
// ========================================
const remindersOutput = document.getElementById("reminders-output");

function _formatRemindTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  const now = new Date();
  const diffMs = d - now;
  if (diffMs <= 0) return "已到期";
  const hours = Math.floor(diffMs / 3600000);
  const minutes = Math.floor((diffMs % 3600000) / 60000);
  if (hours > 48) return `${Math.round(hours / 24)}天后`;
  if (hours > 0) return `${hours}小时${minutes}分后`;
  return `${minutes}分后`;
}

async function dismissReminder(reminderId, rowEl) {
  try {
    const res = await fetch(`/api/reminders/${reminderId}/dismiss`, { method: "POST" });
    const data = await res.json();
    if (res.ok) {
      rowEl.remove();
      const container = remindersOutput;
      if (container && container.children.length === 0) {
        container.innerHTML = '<div class="reminder-empty">暂无待提醒事项</div>';
      }
    } else {
      alert(`取消失败: ${data.detail || "未知错误"}`);
    }
  } catch (error) {
    alert(`网络错误: ${error.message}`);
  }
}

function renderReminders(reminders) {
  if (!remindersOutput) return;
  remindersOutput.innerHTML = "";
  if (!reminders || reminders.length === 0) {
    remindersOutput.innerHTML = '<div class="reminder-empty">暂无待提醒事项</div>';
    return;
  }

  // 标题
  const header = document.createElement("div");
  header.className = "reminder-group-title";
  header.textContent = "提醒";
  remindersOutput.appendChild(header);

  reminders.forEach((rem) => {
    const row = document.createElement("div");
    row.className = "reminder-row";

    const content = document.createElement("div");
    content.className = "reminder-content";

    const title = document.createElement("div");
    title.className = "reminder-title";
    title.textContent = rem.title;

    const meta = document.createElement("div");
    meta.className = "reminder-meta";
    const nextStr = _formatRemindTime(rem.next_remind_at);
    meta.textContent = `已提醒${rem.remind_count}次 · 下次${nextStr}`;

    content.appendChild(title);
    content.appendChild(meta);

    const dismissBtn = document.createElement("button");
    dismissBtn.className = "reminder-dismiss";
    dismissBtn.textContent = "×";
    dismissBtn.title = "忽略此提醒";
    dismissBtn.addEventListener("click", () => dismissReminder(rem.id, row));

    row.appendChild(content);
    row.appendChild(dismissBtn);
    remindersOutput.appendChild(row);
  });
}

async function loadReminders() {
  try {
    const res = await fetch(`/api/reminders/pending?session_id=${sessionId}`);
    const data = await res.json();
    const result = data.result || data.data || {};
    renderReminders(result.reminders || []);
  } catch (error) {
    if (remindersOutput) remindersOutput.textContent = `加载失败: ${error.message}`;
  }
}

// 在加载待办后同时加载提醒
const originalLoadTodos = loadTodos;
loadTodos = async function(url) {
  await originalLoadTodos(url);
  await loadReminders();
};

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
// 顶部居中状态动画（GIF 状态机：待机/思考/知识库检索思考）
// ========================================
const ANIM_GIFS = {
  idle: "/static/待机.gif",
  think_start: "/static/思考开始.gif",
  thinking: "/static/思考中.gif",
  think_end: "/static/思考结束.gif",
  kb_search_start: "/static/知识库搜索开始.gif",
  kb_searching: "/static/知识库搜索中.gif",
  kb_found: "/static/知识库搜索到思索.gif",
  kb_think_start: "/static/知识库思考开始.gif",
  kb_thinking: "/static/知识库思考.gif"
};
// 过渡动画固定播放时长，播完切到后续阶段；循环类阶段持续到下一次请求
const ANIM_TRANSITION_MS = 1800;
const ANIM_SUCCESSOR = {
  think_start: "thinking",
  think_end: "idle",
  kb_search_start: "kb_searching",
  kb_found: "kb_think_start",
  kb_think_start: "kb_thinking",
  think_resume: "thinking"
};

let animTimer = null;
let animPendingStage = null;

// 启动即预加载全部 GIF，避免切换时卡顿/闪白
Object.values(ANIM_GIFS).forEach((src) => { const img = new Image(); img.src = src; });

function _applyAnim(stage) {
  const el = document.getElementById("think-anim");
  const img = document.getElementById("think-img");
  animPendingStage = null;
  if (!el || !img) return;
  el.style.display = "";
  const src = ANIM_GIFS[stage];
  if (src && img.getAttribute("src") !== src) img.src = src;
  const successor = ANIM_SUCCESSOR[stage];
  if (successor) {
    animTimer = setTimeout(() => {
      animTimer = null;
      _applyAnim(animPendingStage || successor);
    }, ANIM_TRANSITION_MS);
  }
}

function requestAnim(stage) {
  if (!ANIM_GIFS[stage]) return;
  if (animTimer) {
    // 正在播过渡动画：先记下新请求，播完再切换
    animPendingStage = stage;
    return;
  }
  _applyAnim(stage);
}

// ========================================
// 登录
// ========================================
async function handleLogin() {
  const username = loginInput.value.trim();
  if (!username) {
    loginInput.focus();
    return;
  }
  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username }),
    });
    if (!res.ok) throw new Error("登录失败");
    const data = await res.json();
    sessionId = data.session_id;
    localStorage.setItem("lifeops_session_id", sessionId);
    loginSection.style.display = "none";
    appSection.style.display = "";
    requestAnim("idle");
    loadTodos("/api/todos/upcoming?days=7");
    loadReminders();
  } catch (err) {
    console.error("Login error:", err);
  }
}

if (sessionId) {
  loginSection.style.display = "none";
  appSection.style.display = "";
  requestAnim("idle");
  loadTodos("/api/todos/upcoming?days=7");
  loadReminders();
} else {
  loginSection.style.display = "";
  appSection.style.display = "none";
}

if (loginBtn) {
  loginBtn.addEventListener("click", handleLogin);
}
if (loginInput) {
  loginInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleLogin();
    }
  });
}
if (logoutBtn) {
  logoutBtn.addEventListener("click", () => {
    localStorage.removeItem("lifeops_session_id");
    sessionId = "";
    appSection.style.display = "none";
    loginSection.style.display = "";
    loginInput.value = "";
    loginInput.focus();
  });
}

// ========================================
// 聊天功能
// ========================================
let activeStream = null;

function abortActiveStream() {
  if (activeStream && !activeStream.aborted) {
    activeStream.aborted = true;
    try { activeStream.controller.abort(); } catch (_) { /* noop */ }
  }
}

async function sendChatMessage() {
  const message = chatInput.value.trim();
  if (!message) return;

  appendChat("user", message);
  chatInput.value = "";
  const assistantTurn = appendAssistantTurn();

  const controller = new AbortController();
  activeStream = { controller, aborted: false };

  let stopped = false;
  let queued = false;
  let finalResult = null;
  let assistantText = "";
  let thinkingStarted = false;
  let usedKbFlow = false;
  let streamLine = null;

  function showStopIfActive() {
    if (stopped || finalResult) return;
    assistantTurn.setStopVisible(!queued);
  }
  function hideStop() {
    assistantTurn.setStopVisible(false);
  }

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) {
      requestAnim("think_end");
      throw new Error(`请求失败: ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      let chunk;
      try {
        chunk = await reader.read();
      } catch (err) {
        if (err && (err.name === "AbortError") && activeStream && activeStream.aborted) {
          stopped = true;
          break;
        }
        throw err;
      }
      const { value, done } = chunk;
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const part of parts) {
        const line = part.split("\n").find((item) => item.startsWith("data: "));
        if (!line) continue;
        let payload;
        try {
          payload = JSON.parse(line.slice(6));
        } catch (_) {
          continue;
        }

        if (payload.type === "anim") {
          if (!thinkingStarted) {
            thinkingStarted = true;
            assistantTurn.expandThought();
          }
          const stage = payload.stage || "";
          if (stage.startsWith("kb_")) usedKbFlow = true;
          requestAnim(ANIM_GIFS[stage] ? stage : "thinking");
          continue;
        }

        if (!thinkingStarted && (payload.type === "status" || payload.type === "step" || payload.type === "delta")) {
          thinkingStarted = true;
          requestAnim("think_start");
          assistantTurn.expandThought();
        }

        if (payload.type === "delta") {
          assistantText += payload.delta || "";
          if (assistantTurn?.thoughtDiv) {
            if (!streamLine) {
              streamLine = document.createElement("div");
              streamLine.className = "thought-line";
              streamLine.dataset.stream = "true";
              streamLine.style.color = "#555";
              assistantTurn.thoughtDiv.appendChild(streamLine);
            }
            streamLine.textContent = assistantText;
          }
        } else if (payload.type === "status") {
          const msg = payload.message || "";
          if (msg === "（已停止）") {
            stopped = true;
            hideStop();
          } else if (msg.startsWith("排队中")) {
            queued = true;
            if (assistantTurn?.thoughtDiv) setThoughtLines(assistantTurn.thoughtDiv, [msg]);
          } else {
            queued = false;
            showStopIfActive();
            if (assistantTurn?.thoughtDiv) {
              let firstLine = assistantTurn.thoughtDiv.querySelector(".thought-line");
              if (!firstLine) {
                setThoughtLines(assistantTurn.thoughtDiv, [msg || "正在思考..."]);
              } else {
                firstLine.textContent = msg || "正在思考...";
              }
            }
          }
        } else if (payload.type === "step") {
          queued = false;
          showStopIfActive();
          if (assistantTurn?.thoughtDiv) {
            const lines = assistantTurn.thoughtDiv.querySelectorAll(".thought-line");
            const lastLine = lines[lines.length - 1];
            const text = payload.message || "";
            if (lastLine && lastLine.textContent === text) continue;
            const newLine = document.createElement("div");
            newLine.className = "thought-line";
            newLine.textContent = text;
            if (lastLine && lastLine.textContent.startsWith("正在") && lastLine.dataset.stream !== "true") {
              lastLine.textContent = text;
            } else {
              assistantTurn.thoughtDiv.appendChild(newLine);
            }
          }
        } else if (payload.type === "error") {
          hideStop();
          requestAnim("think_end");
          if (assistantTurn?.answerDiv) assistantTurn.answerDiv.textContent = `回答：错误: ${payload.message || "未知错误"}`;
        } else if (payload.type === "final") {
          finalResult = payload.result;
          hideStop();
          // 知识库流程不播结束动画直接回待机；普通流程播思考结束
          requestAnim(usedKbFlow ? "idle" : "think_end");
        }
      }
    }

    if (stopped) {
      hideStop();
      requestAnim("idle");
      if (assistantTurn?.thoughtDiv) setThoughtLines(assistantTurn.thoughtDiv, ["（已停止）"]);
      if (assistantTurn?.answerDiv) assistantTurn.answerDiv.textContent = `回答：已停止`;
    } else if (finalResult) {
      const finalAnswer = finalResult.answer || assistantText || "";
      assistantTurn.collapseThought();
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
    hideStop();
    requestAnim("think_end");
    if (assistantTurn?.thoughtDiv) setThoughtLines(assistantTurn.thoughtDiv, ["请求失败"]);
    if (assistantTurn?.answerDiv) assistantTurn.answerDiv.textContent = `回答：错误: ${error.message}`;
  } finally {
    activeStream = null;
  }
}

if (chatSend) {
  chatSend.addEventListener("click", sendChatMessage);
}

document.addEventListener("keydown", function(e) {
  if (e.key === "Enter" && !e.shiftKey && document.activeElement === chatInput) {
    e.preventDefault();
    if (chatSend) chatSend.click();
  }
});

// ========================================
// 语音输入（MediaRecorder 录音 + 本地 Whisper 识别，需 HTTPS 或 localhost）
// ========================================
const chatMic = document.getElementById("chat-mic");
let mediaRecorder = null;
let mediaStream = null;
let micChunks = [];
let micListening = false;
let micAutoStopTimer = null;

function _resetMicUI() {
  micListening = false;
  if (chatMic) {
    chatMic.classList.remove("listening");
    chatMic.classList.remove("disabled");
    chatMic.title = "语音输入";
  }
}

function _stopMicAutoStopTimer() {
  if (micAutoStopTimer) {
    clearTimeout(micAutoStopTimer);
    micAutoStopTimer = null;
  }
}

function _stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  _stopMicAutoStopTimer();
}

if (chatMic && navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder) {
  function _pickMimeType() {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ];
    for (const type of candidates) {
      if (window.MediaRecorder.isTypeSupported(type)) return type;
    }
    return "";
  }

  async function _uploadSpeech(blob) {
    chatMic.classList.add("disabled");
    chatMic.title = "识别中...";
    try {
      const resp = await fetch("/api/speech", {
        method: "POST",
        headers: { "Content-Type": blob.type || "audio/webm" },
        body: blob,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const text = data && data.result && data.result.text ? data.result.text : "";
      if (text) {
        chatInput.value = text;
        chatInput.focus();
        if (chatSend) chatSend.click();
      } else {
        appendChat("assistant", "没有听清，请再说一次");
      }
    } catch (err) {
      appendChat("assistant", `语音识别失败: ${err.message}`);
    } finally {
      _resetMicUI();
    }
  }

  chatMic.addEventListener("click", () => {
    if (micListening) {
      _stopRecording();
      return;
    }
    if (!chatInput.value.trim()) {
      chatInput.value = "";
    }
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then((stream) => {
        mediaStream = stream;
        const mimeType = _pickMimeType();
        micChunks = [];
        mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
        mediaRecorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) micChunks.push(event.data);
        };
        mediaRecorder.onstop = () => {
          const blob = new Blob(micChunks, { type: mimeType || "audio/webm" });
          _resetMicUI();
          _uploadSpeech(blob);
        };
        mediaRecorder.onerror = () => {
          _resetMicUI();
          appendChat("assistant", "语音录制失败");
        };
        mediaRecorder.start();
        micListening = true;
        chatMic.classList.add("listening");
        chatMic.classList.add("disabled");
        chatMic.title = "正在聆听...";
        micAutoStopTimer = setTimeout(() => {
          if (micListening) _stopRecording();
        }, 20000);
      })
      .catch((err) => {
        _resetMicUI();
        appendChat("assistant", `无法访问麦克风: ${err.name || err.message}`);
      });
  });
} else if (chatMic) {
  chatMic.style.display = "none";
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

function renderGroupedTodos(pending) {
  if (!todosOutput) return;
  todosOutput.innerHTML = "";

  const now = new Date();
  const pendingHeader = document.createElement("div");
  pendingHeader.className = "todo-group-title";
  pendingHeader.textContent = "待完成";
  todosOutput.appendChild(pendingHeader);

  if (!pending || pending.length === 0) {
    const empty = document.createElement("div");
    empty.className = "todo-empty";
    empty.textContent = "暂无待办";
    todosOutput.appendChild(empty);
    return;
  }

  const groups = {};
  pending.forEach(t => {
    const d = t.due_at ? new Date(t.due_at) : null;
    let label = "未设置日期";
    if (d) {
      const month = d.getMonth() + 1;
      const day = d.getDate();
      const weekday = ["日", "一", "二", "三", "四", "五", "六"][d.getDay()];
      const diff = Math.floor((d - now) / 86400000);
      if (diff < 1 && d.toDateString() === now.toDateString()) label = "今天";
      else if (diff < 2 && d.toDateString() === new Date(now.getTime() + 86400000).toDateString()) label = "明天";
      else label = `${month}月${day}日 周${weekday}`;
    }
    if (!groups[label]) groups[label] = [];
    groups[label].push(t);
  });
  const sortedLabels = Object.keys(groups).sort((a, b) => {
    if (a === "今天") return -1;
    if (b === "今天") return 1;
    if (a === "明天") return -1;
    if (b === "明天") return 1;
    if (a === "未设置日期") return 1;
    if (b === "未设置日期") return -1;
    return a.localeCompare(b, "zh");
  });
  sortedLabels.forEach(label => {
    const groupHeader = document.createElement("div");
    groupHeader.className = "todo-group-title";
    groupHeader.textContent = label;
    todosOutput.appendChild(groupHeader);
    renderTodoItems(groups[label], { canToggle: true });
  });
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
    } else if (url.includes("/api/todos/all")) {
      currentTodoView = "all";
      renderGroupedTodos(result.pending || []);
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



// 初始加载
if (todosUpcoming) todosUpcoming.classList.add("active");
loadTodos("/api/todos/upcoming?days=7");

function _setActiveTab(activeBtn) {
  [todosToday, todosUpcoming, todosAll].forEach(btn => {
    if (btn) btn.classList.remove("active");
  });
  if (activeBtn) activeBtn.classList.add("active");
}

if (todosToday) {
  todosToday.addEventListener("click", () => {
    _setActiveTab(todosToday);
    loadTodos("/api/todos/today");
  });
}

if (todosUpcoming) {
  todosUpcoming.addEventListener("click", () => {
    _setActiveTab(todosUpcoming);
    loadTodos("/api/todos/upcoming?days=7");
  });
}

if (todosAll) {
  todosAll.addEventListener("click", () => {
    _setActiveTab(todosAll);
    loadTodos("/api/todos/all");
  });
}

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

if (indexReset) {
  indexReset.addEventListener("click", async () => {
    if (!confirm("确定要重置索引吗？将清空所有存量数据后全量重录。")) return;
    const startTime = new Date();
    try {
      indexOutput.textContent = `重置中...（开始时间: ${startTime.toLocaleTimeString()}，可能需数分钟）`;
      const res = await fetch("/api/index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reset: true }),
      });
      const elapsed = ((new Date() - startTime) / 1000).toFixed(1);
      const data = await res.json();
      data.elapsed_seconds = parseFloat(elapsed);
      indexOutput.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
      indexOutput.textContent = `错误（耗时 ${((new Date() - startTime) / 1000).toFixed(1)}s）: ${error.message}`;
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
