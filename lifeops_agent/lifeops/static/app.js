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

let sessionId = localStorage.getItem("lifeops_session_id");
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem("lifeops_session_id", sessionId);
}

let proposalId = null;
planConfirm.disabled = true;

function renderProposal(proposal, id) {
  if (!proposal || !proposal.items || proposal.items.length === 0) return;
  proposalId = id;
  planOutput.textContent = JSON.stringify(proposal, null, 2);
  planConfirm.disabled = false;
}

function appendChat(role, text) {
  const div = document.createElement("div");
  div.className = "chat-line";
  div.textContent = `${role}: ${text}`;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

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
  if (data.proposal_id && data.proposal) {
    appendChat("assistant", "检测到待办意图，已生成草案，请确认加入待办。\n" + JSON.stringify(data.proposal, null, 2));
    renderProposal(data.proposal, data.proposal_id);
  }
});

planConfirm.addEventListener("click", async () => {
  if (!proposalId) return;
  const res = await fetch("/api/plan/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ proposal_id: proposalId }),
  });
  const data = await res.json();
  planOutput.textContent = `Inserted: ${data.inserted}`;
  planConfirm.disabled = true;
});

todosToday.addEventListener("click", async () => {
  const res = await fetch("/api/todos/today");
  const data = await res.json();
  todosOutput.textContent = JSON.stringify(data, null, 2);
});

todosUpcoming.addEventListener("click", async () => {
  const res = await fetch("/api/todos/upcoming?days=7");
  const data = await res.json();
  todosOutput.textContent = JSON.stringify(data, null, 2);
});

indexRun.addEventListener("click", async () => {
  const res = await fetch("/api/index", { method: "POST" });
  const data = await res.json();
  indexOutput.textContent = JSON.stringify(data, null, 2);
});

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
  askCitations.innerHTML = "";
  data.citations.forEach((cite) => {
    const div = document.createElement("div");
    div.className = "citation";
    div.textContent = `${cite.path} | page: ${cite.page ?? "-"} | score: ${cite.score} | ${cite.snippet}`;
    askCitations.appendChild(div);
  });
});
