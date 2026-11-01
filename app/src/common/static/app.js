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

function readIdentityHint() {
  try { return localStorage.getItem("lifeops_session_id") || ""; }
  catch (_) { return ""; }
}
function rememberIdentityHint(value) {
  try {
    if (value) localStorage.setItem("lifeops_session_id", value);
    else localStorage.removeItem("lifeops_session_id");
  } catch (_) { /* Cookie authentication works without a local identity hint. */ }
}
let sessionId = readIdentityHint();
let identityRevision = 0;
let todoLoadSequence = 0;
const pendingChatRequests = new Map();

function pendingStorageKey(owner) {
  return `lifeops_pending_requests:${encodeURIComponent(owner)}`;
}

function readPendingRequests(owner) {
  try {
    const records = JSON.parse(sessionStorage.getItem(pendingStorageKey(owner)) || "[]");
    return Array.isArray(records) ? records.filter(payload =>
      payload && payload.session_id === owner && typeof payload.message === "string" &&
      typeof payload.request_id === "string" && /^[A-Za-z0-9_-]{1,128}$/.test(payload.request_id)
    ) : [];
  } catch (_) { return []; }
}

function persistPendingRequest(request, remove = false) {
  try {
    const payload = request.payload;
    const records = readPendingRequests(payload.session_id).filter(item => item.request_id !== payload.request_id);
    if (!remove) records.push(payload);
    if (records.length) sessionStorage.setItem(pendingStorageKey(payload.session_id), JSON.stringify(records));
    else sessionStorage.removeItem(pendingStorageKey(payload.session_id));
  } catch (_) { /* Storage may be disabled/full; same-page retry still works. */ }
}

function restorePendingRequests(owner) {
  for (const payload of readPendingRequests(owner)) {
    if (pendingChatRequests.has(payload.request_id)) continue;
    const request = createChatRequest(payload.message, payload.proposal_id || null, payload);
    request.turn.answerDiv.textContent = "上次请求的结果未确认，可点击“重试这条消息”查询或继续原请求。";
    request.turn.setRetryVisible(true);
    if (payload.proposal_id) {
      proposalRequests.set(JSON.stringify([owner, payload.proposal_id, payload.message]), request);
    }
  }
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
    const details = document.createElement("details");
    details.className = "citation";
    details.classList.add(`cite-${cite.relatedness || "low"}`);
    const summ = document.createElement("summary");
    const page = cite.page ? ` · 第 ${cite.page} 页` : "";
    const relLabel = { strong: "强相关", weak: "弱相关", low: "低参考" }[cite.relatedness || "low"] || "低参考";
    const section = cite.section_path || cite.section_title;
    const sectionLabel = section ? ` · ${section}` : "";
    summ.textContent = `${cite.path}${page}${sectionLabel}（相关度: ${Math.round(cite.score || 0)} · ${relLabel}）`;
    const body = document.createElement("div");
    body.className = "citation-body";
    body.textContent = cite.snippet || "(无正文)";
    details.appendChild(summ);
    details.appendChild(body);
    askCitations.appendChild(details);
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

function appendAssistantTurn(requestId, turnSessionId, retry) {
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
  const retryBtn = document.createElement("button");
  retryBtn.type = "button";
  retryBtn.className = "assistant-retry-btn";
  retryBtn.textContent = "重试这条消息";
  retryBtn.hidden = true;
  retryBtn.addEventListener("click", () => retry());
  let stopPending = false;
  stopBtn.addEventListener("click", () => {
    if (stopPending) return;
    stopPending = true;
    stopBtn.disabled = true;
    stopBtn.textContent = "停止中…";
    fetch("/api/chat/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: turnSessionId, request_id: requestId }),
    }).then(async (response) => {
      if (!response.ok) throw new Error("停止失败，请重试");
    }).catch((error) => {
      stopPending = false;
      stopBtn.disabled = false;
      stopBtn.textContent = "停止";
      console.error(error);
    });
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
  thoughtHeader.appendChild(retryBtn);

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
    retryBtn,
    setRetryVisible(visible) {
      retryBtn.hidden = !visible;
      retryBtn.disabled = !visible;
    },
    resetAttempt() {
      stopPending = false;
      stopBtn.disabled = false;
      stopBtn.textContent = "停止";
      setThoughtLines(thoughtDiv, ["正在处理这条消息..."]);
      answerDiv.textContent = "回答：正在生成中...";
    },
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
      if (item.repeat || item.repeat_rule) {
        const rawRule = item.repeat || item.repeat_rule;
        const ruleText = rawRule === "daily" ? "每日" : rawRule === "weekly" ? "每周" : rawRule;
        const start = item.due_at ? String(item.due_at).slice(0, 10) : "";
        const until = item.repeat_until ? String(item.repeat_until).slice(0, 10) : "";
        let periodText = ruleText;
        if (start && until) periodText = `${ruleText}（${start} 至 ${until}）`;
        else if (until) periodText = `${ruleText}（至 ${until}）`;
        content += `   周期: ${periodText}\n`;
      }
      if (item.due_at) {
        const dueStr = String(item.due_at);
        if (dueStr.length > 10) {
          content += `   截止时间: ${new Date(item.due_at).toLocaleString('zh-CN')}\n`;
        } else {
          content += `   开始时间: ${new Date(item.due_at).toLocaleDateString('zh-CN')}\n`;
        }
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

// 草案确认/取消也通过同一请求生命周期执行，失败后复用原请求。
let confirmBusy = false;
const proposalRequests = new Map();

async function sendProposalDecision(message) {
  const proposalId = window.currentProposalId;
  if (confirmBusy || !proposalId) return;
  const key = JSON.stringify([sessionId, proposalId, message]);
  let request = proposalRequests.get(key);
  if (!request) {
    request = createChatRequest(message, proposalId);
    proposalRequests.set(key, request);
  }
  confirmBusy = true;
  confirmProposal.disabled = true;
  rejectProposal.disabled = true;
  try {
    return await executeChatRequest(request);
  } finally {
    confirmBusy = false;
    confirmProposal.disabled = false;
    rejectProposal.disabled = false;
    if (request.completed || request.cancelled) proposalRequests.delete(key);
  }
}

if (confirmProposal) {
  confirmProposal.addEventListener("click", () => sendProposalDecision("确认"));
}
if (rejectProposal) {
  rejectProposal.addEventListener("click", () => sendProposalDecision("取消"));
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
let loginPending = false;
async function handleLogin() {
  if (loginPending) return;
  const username = loginInput.value.trim();
  if (!username) {
    loginInput.focus();
    return;
  }
  loginPending = true;
  if (loginBtn) loginBtn.disabled = true;
  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password: document.getElementById("login-password").value }),
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(typeof error.detail === "string" ? error.detail : "登录失败，请检查密码（至少8位）");
    }
    const data = await res.json();
    if (typeof data.session_id !== "string" || !data.session_id.trim()) throw new Error("登录响应异常，请重试。");
    identityRevision++;
    sessionId = data.session_id;
    document.getElementById("login-password").value = "";
    document.getElementById("login-error").textContent = "";
    rememberIdentityHint(sessionId);
    loginSection.style.display = "none";
    appSection.style.display = "";
    requestAnim("idle");
    loadTodos("/api/todos/upcoming?days=7");
    restorePendingRequests(sessionId);
  } catch (err) {
    document.getElementById("login-error").textContent = err.message;
    console.error("Login error:", err);
  } finally {
    loginPending = false;
    if (loginBtn) loginBtn.disabled = false;
  }
}

if (sessionId) {
  const verifyingSession = sessionId;
  const verifyingRevision = identityRevision;
  loginSection.style.display = "none";
  appSection.style.display = "";
  requestAnim("idle");
  loadTodos("/api/todos/upcoming?days=7");
  fetch("/api/auth/me").then(async (response) => {
    if (!response.ok && response.status !== 401) throw new Error("Session check unavailable");
    const user = response.ok ? await response.json() : null;
    if (sessionId !== verifyingSession || identityRevision !== verifyingRevision) return;
    if (!user || user.session_id !== sessionId) {
      rememberIdentityHint("");
      identityRevision++;
      sessionId = "";
      appSection.style.display = "none";
      loginSection.style.display = "";
    } else {
      restorePendingRequests(sessionId);
    }
  }).catch(() => {
    if (sessionId !== verifyingSession || identityRevision !== verifyingRevision) return;
    appSection.style.display = "none";
    loginSection.style.display = "";
    document.getElementById("login-error").textContent = "暂时无法验证登录状态，请检查网络后重新登录。";
  });
} else {
  loginSection.style.display = "";
  appSection.style.display = "none";
}

if (loginBtn) {
  loginBtn.addEventListener("click", handleLogin);
}
if (loginInput) {
  loginInput.addEventListener("keydown", (e) => {
    if (e.isComposing || e.keyCode === 229) return;
    if (e.key === "Enter") {
      e.preventDefault();
      handleLogin();
    }
  });
}
if (logoutBtn) {
  logoutBtn.addEventListener("click", async () => {
    let logoutConfirmed = false;
    try {
      const response = await fetch("/api/auth/logout", { method: "POST" });
      logoutConfirmed = response.ok;
    } catch (_) { /* Local data is still cleared below. */ }
    try { sessionStorage.removeItem(pendingStorageKey(sessionId)); } catch (_) { /* unavailable */ }
    for (const [id, request] of pendingChatRequests) {
      if (request.payload.session_id === sessionId) {
        request.turn.setRetryVisible(false);
        pendingChatRequests.delete(id);
      }
    }
    for (const [key, request] of proposalRequests) {
      if (request.payload.session_id === sessionId) proposalRequests.delete(key);
    }
    hideProposal();
    chatLog.innerHTML = "";
    if (indexOutput) indexOutput.textContent = "";
    if (askAnswer) askAnswer.textContent = "";
    if (todosOutput) todosOutput.innerHTML = "";
    renderCitations([]);
    rememberIdentityHint("");
    identityRevision++;
    sessionId = "";
    appSection.style.display = "none";
    loginSection.style.display = "";
    loginInput.value = "";
    document.getElementById("login-error").textContent = logoutConfirmed ? ""
      : "本地内容已清除，但未能确认服务器退出。请恢复网络后重新登录并退出。";
    loginInput.focus();
  });
}

document.getElementById("login-password").addEventListener("keydown", (event) => {
  if (event.isComposing || event.keyCode === 229) return;
  if (event.key === "Enter") handleLogin();
});

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
  if (Array.from(message).length > 1000) {
    alert("正文不能超过1000个字符，请缩短后发送。");
    return;
  }
  chatInput.value = "";
  return executeChatRequest(createChatRequest(message));
}

function createChatRequest(message, proposalId = null, originalPayload = null) {
  const payload = originalPayload ? { ...originalPayload } : { message, session_id: sessionId, request_id: crypto.randomUUID() };
  if (proposalId) payload.proposal_id = proposalId;
  const request = {
    payload: Object.freeze(payload),
    body: JSON.stringify(payload),
    running: false,
    completed: false,
    cancelled: false,
  };
  appendChat("user", message);
  request.turn = appendAssistantTurn(payload.request_id, payload.session_id, () => executeChatRequest(request));
  pendingChatRequests.set(payload.request_id, request);
  persistPendingRequest(request);
  return request;
}

async function executeChatRequest(request) {
  if (request.running || request.completed || request.cancelled) return null;
  const { payload, turn: assistantTurn } = request;
  if (!sessionId || sessionId !== payload.session_id) {
    assistantTurn.answerDiv.textContent = "请先登录发送这条消息的账号，再重试。";
    return null;
  }
  request.running = true;
  assistantTurn.setRetryVisible(false);
  assistantTurn.resetAttempt();

  const controller = new AbortController();
  const stream = { controller, aborted: false };
  activeStream = stream;

  let stopped = false;
  let queued = false;
  let finalResult = null;
  let assistantText = "";
  let thinkingStarted = false;
  let usedKbFlow = false;
  let streamLine = null;
  let serverError = null;
  let terminalError = null;

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
      body: request.body,
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
        if (err && (err.name === "AbortError") && stream.aborted) {
          stopped = true;
          break;
        }
        throw err;
      }
      const { value, done } = chunk;
      if (sessionId !== request.payload.session_id) {
        try { await reader.cancel(); } catch (_) { /* old identity's response is discarded */ }
        return null;
      }
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split(/\r?\n\r?\n/);
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
          serverError = payload.message || "请求未能完成";
          if (payload.retryable === false) terminalError = payload;
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
      if (finalResult || stopped || serverError) {
        try { await reader.cancel(); } catch (_) { /* terminal event already received */ }
        break;
      }
    }

    if (stopped || finalResult?.request_status === "cancelled") {
      request.cancelled = true;
      hideStop();
      requestAnim("idle");
      if (assistantTurn?.thoughtDiv) setThoughtLines(assistantTurn.thoughtDiv, ["（已停止）"]);
      if (assistantTurn?.answerDiv) assistantTurn.answerDiv.textContent = "回答：已停止。可以发送补充条件，继续处理上一轮任务。";
    } else if (finalResult) {
      request.completed = true;
      const finalAnswer = finalResult.answer || assistantText || "";
      assistantTurn.collapseThought();
      if (assistantTurn?.answerDiv) {
        const outcome = finalResult.tool_error?.outcome;
        const status = finalResult.tool_error ? ({
          some_operations_committed: "部分操作已提交。请先核对记录，已提交的操作不会自动撤销。",
          unknown: "写入结果不确定。请先核对记录，不要重复提交相同操作。",
          not_applied: "当前操作未应用。",
          not_completed: "本轮任务未完成。",
        }[outcome] || "本轮执行未成功完成。") : "";
        assistantTurn.answerDiv.textContent = status ? `${status}\n\n回答：${finalAnswer}` : `回答：${finalAnswer}`;
      }
      if (assistantTurn?.thoughtDiv) setThoughtLines(assistantTurn.thoughtDiv, buildThoughtLines(finalResult.trace, finalResult));

      if (finalResult.tool_error?.invalidated_proposal_id === window.currentProposalId && window.currentProposalId) {
        hideProposal();
      }

      if (finalResult.proposal_id && finalResult.proposal && ["plan_proposal", "revise_proposal"].includes(finalResult.used_tool)) {
        showProposal(finalResult.proposal_id, finalResult.proposal);
      }

      if (payload.proposal_id && ["plan_gate_confirm", "plan_gate_cancel"].includes(finalResult.used_tool)) {
        if (window.currentProposalId === payload.proposal_id) hideProposal();
        loadTodos("/api/todos/upcoming?days=7");
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
    } else if (terminalError) {
      request.completed = true;
      assistantTurn.collapseThought();
      if (assistantTurn?.answerDiv) assistantTurn.answerDiv.textContent = `回答：${serverError}`;
      if (terminalError.invalidated_proposal_id === window.currentProposalId && window.currentProposalId) hideProposal();
    } else {
      throw new Error(serverError || "连接已结束，但未收到最终结果");
    }
  } catch (error) {
    hideStop();
    requestAnim("think_end");
    if (stream.aborted && error.name === "AbortError") {
      request.cancelled = true;
      assistantTurn.answerDiv.textContent = "回答：已停止接收，请核对实际处理结果。";
    } else if (!finalResult) {
      setThoughtLines(assistantTurn.thoughtDiv, [error.message || "请求失败"]);
      assistantTurn.answerDiv.textContent = "结果未确认，可点击“重试这条消息”继续确认原请求；无需重新发送。";
      assistantTurn.setRetryVisible(true);
    }
  } finally {
    request.running = false;
    if (activeStream === stream) activeStream = null;
    hideStop();
    if (request.completed || request.cancelled) {
      pendingChatRequests.delete(payload.request_id);
      persistPendingRequest(request, true);
      if (payload.proposal_id) proposalRequests.delete(JSON.stringify([payload.session_id, payload.proposal_id, payload.message]));
    }
  }
  return finalResult || null;
}

if (chatSend) {
  chatSend.addEventListener("click", sendChatMessage);
}

document.addEventListener("keydown", function(e) {
  if (e.isComposing || e.keyCode === 229) return;
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

    if (todo.repeat_rule) {
      const badge = document.createElement("span");
      badge.className = "todo-repeat-badge";
      const ruleText = todo.repeat_rule === "daily" ? "每日" : todo.repeat_rule === "weekly" ? "每周" : todo.repeat_rule;
      let startTag = "";
      let untilTag = "";
      if (todo.due_at) { try { startTag = new Date(todo.due_at).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" }); } catch (_) {} }
      if (todo.repeat_until) { try { untilTag = new Date(todo.repeat_until).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" }); } catch (_) {} }
      badge.textContent = startTag && untilTag ? `${ruleText}·${startTag}~${untilTag}` : `${ruleText}${untilTag ? `·至${untilTag}` : ""}`;
      title.appendChild(badge);
    }

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
  if (!sessionId) return;
  const owner = sessionId;
  const revision = identityRevision;
  const sequence = ++todoLoadSequence;
  const current = () => owner === sessionId && revision === identityRevision && sequence === todoLoadSequence;
  try {
    // 添加session_id参数
    const separator = url.includes('?') ? '&' : '?';
    const urlWithSession = `${url}${separator}session_id=${encodeURIComponent(owner)}`;
    
    const res = await fetch(urlWithSession);
    const data = await res.json();
    if (!current()) return;
    if (!res.ok || data.success === false) throw new Error("待办加载未完成，请重试。");

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
    if (current() && todosOutput) todosOutput.textContent = "待办加载失败，请检查登录状态或网络后重试。";
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
let indexing = false;
  async function runIndex(reset = false) {
    if (indexing || !indexOutput) return;
    if (reset && !confirm("确定要重建知识库索引吗？")) return;
    const owner = sessionId;
    const revision = identityRevision;
    indexing = true;
    if (indexRun) indexRun.disabled = true;
    if (indexReset) indexReset.disabled = true;
    indexOutput.textContent = "正在建立索引，请等待，可能需要数分钟…";
    try {
      const res = await fetch("/api/index", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reset }),
      });
      const data = await res.json();
      if (owner !== sessionId || revision !== identityRevision) return;
      if (!res.ok || data.success === false) {
        const messages = {
          409: "已有索引任务正在运行，请等待后重试。",
          422: "文档超过导入限制，请检查文件大小、页数或图片尺寸。",
          401: "登录已失效，请重新登录。",
        };
        indexOutput.textContent = messages[res.status] || "索引未完整完成，请检查日志并核对现有索引后重试。";
        return;
      }
      const result = data.result;
      if (!result || typeof result !== "object") throw new Error("Invalid response");
      const status = result.indexing_status === "degraded"
        ? "部分完成：词法索引可用，向量索引失败，请修复后重建。"
        : "索引建立完成。";
      const summary = result.summary_generation === "background" ? "文档摘要将在后台继续生成。" : "";
      indexOutput.textContent = `${status} 新增片段：${result.inserted ?? 0}，未变化片段：${result.skipped ?? 0}。${summary}`;
    } catch (_) {
      if (owner !== sessionId || revision !== identityRevision) return;
      indexOutput.textContent = "连接中断或响应异常，无法确认索引结果。请先核对状态，再手动重试。";
    } finally {
      indexing = false;
      if (indexRun) indexRun.disabled = false;
      if (indexReset) indexReset.disabled = false;
    }
  }
  if (indexRun) indexRun.addEventListener("click", () => runIndex());
  if (indexReset) indexReset.addEventListener("click", () => runIndex(true));

let askSequence = 0;
async function askKnowledge() {
  const question = askInput.value.trim();
  if (!question) return;
  const sequence = ++askSequence;
  if (Array.from(question).length > 1000) {
    if (askAnswer) {
      askAnswer.classList.remove("hidden");
      askAnswer.textContent = "问题不能超过1000个字符，请缩短后查询。";
    }
    renderCitations([]);
    return;
  }
  const owner = sessionId;
  const revision = identityRevision;
  const current = () => sequence === askSequence && owner === sessionId && revision === identityRevision;
  if (askAnswer) {
    askAnswer.classList.remove("hidden");
    askAnswer.textContent = "查询中...";
  }
  renderCitations([]);
  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!current()) return;
    if (!res.ok || data.success === false) {
      const messages = {
        503: "知识库查询繁忙，请稍后手动重试。",
        401: "登录已失效，请重新登录后查询。",
        422: "问题格式或长度不符合要求，请修改后查询。",
      };
      if (askAnswer) askAnswer.textContent = messages[res.status] || "查询未完成，请稍后重试。";
      return;
    }
    const result = data.result;
    if (!result || typeof result.answer !== "string") throw new Error("查询响应异常，请重试。");
    if (askAnswer) askAnswer.textContent = result.answer;
    renderCitations(Array.isArray(result.citations) ? result.citations : []);
  } catch (_) {
    if (current() && askAnswer) askAnswer.textContent = "查询未完成，请检查登录状态或网络后重试。";
  }
}
if (askSend) askSend.addEventListener("click", askKnowledge);

if (askInput) {
  askInput.addEventListener("keydown", (e) => {
    if (e.isComposing || e.keyCode === 229) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (askSend) askSend.click();
    }
  });
}

console.log("✅ LifeOps Agent UI 已加载");
