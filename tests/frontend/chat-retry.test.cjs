const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Execute the actual browser script. Only browser surfaces and HTTP are faked.
class Element {
  constructor() {
    this.children = [];
    this.listeners = {};
    this.style = {};
    this.dataset = {};
    this.value = '';
    this.className = '';
    this.classList = {
      add: (...names) => { this.className += ' ' + names.join(' '); },
      remove: (...names) => { this.className = this.className.split(/\s+/).filter(n => !names.includes(n)).join(' '); },
      contains: name => this.className.split(/\s+/).includes(name),
      toggle: name => this.classList.contains(name) ? this.classList.remove(name) : this.classList.add(name),
    };
  }
  set textContent(value) { this.text = value; this.children = []; }
  get textContent() { return this.text || ''; }
  set innerHTML(value) { this.children = []; }
  appendChild(child) { this.children.push(child); return child; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  click() { return this.disabled ? undefined : this.listeners.click?.(); }
  querySelectorAll(selector) {
    const result = [];
    for (const child of this.children) {
      if (selector.startsWith('.') && child.classList.contains(selector.slice(1))) result.push(child);
      result.push(...child.querySelectorAll(selector));
    }
    return result;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  getAttribute(name) { return this[name]; }
  setAttribute(name, value) { this[name] = value; }
  scrollIntoView() {}
  focus() {}
}

function browser(fetchChat, storage = new Map(), fetchIndex = null) {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  let nextId = 0;
  const calls = [];
  const context = vm.createContext({
    document: { getElementById: element, createElement: () => new Element(),
      querySelectorAll: () => [], querySelector: () => null, addEventListener() {} },
    window: { addEventListener() {} },
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    sessionStorage: {
      getItem: key => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key),
    },
    Image: class {}, navigator: {},
    setTimeout: () => 0, clearTimeout() {},
    crypto: { randomUUID: () => `request-${++nextId}` },
    AbortController, TextDecoder,
    alert() {}, confirm: () => true, console: { log() {}, error() {} },
    fetch: async (url, options) => {
      if (url === '/api/index' && fetchIndex) return fetchIndex(JSON.parse(options.body));
      if (url === '/api/chat/stream') {
        const payload = JSON.parse(options.body);
        calls.push(payload);
        return fetchChat(payload, calls.length);
      }
      return { ok: true, json: async () => ({ success: true, result: { pending: [], completed: [], todos: [] } }) };
    },
  });
  const source = readFileSync(path.resolve(__dirname, '../../app/src/common/static/app.js'), 'utf8');
  vm.runInContext(source, context);
  vm.runInContext('sessionId = "alice"', context);
  return { context, calls, element, retry: () => element('chat-log').querySelector('.assistant-retry-btn') };
}

function response(events, { broken = false, separator = '\n\n' } = {}) {
  // Split UTF-8 mid-character/frame, as a real response stream is allowed to do.
  const bytes = Buffer.from(events.map(event => `data: ${JSON.stringify(event)}${separator}`).join(''));
  let offset = 0;
  return { ok: true, body: { getReader: () => ({
    async read() {
      if (offset < bytes.length) {
        const value = bytes.subarray(offset, offset + 7);
        offset += value.length;
        return { value, done: false };
      }
      if (broken) throw new TypeError('connection interrupted');
      return { done: true };
    },
    async cancel() {},
  }) } };
}

const final = (answer = '已记录', used_tool = 'record_expense') => ({ type: 'final', result: { answer, used_tool } });

test('lost response retries the identical request without a second user bubble or mutation', async () => {
  const receipts = new Map();
  let mutations = 0;
  const b = browser((payload, attempt) => {
    if (!receipts.has(payload.request_id)) { receipts.set(payload.request_id, final()); mutations++; }
    return attempt === 1 ? response([{ type: 'delta', delta: '部分输出' }], { broken: true })
      : response([receipts.get(payload.request_id)]);
  });
  b.element('chat-input').value = '我花了10块钱';
  await b.context.sendChatMessage();
  assert.equal(b.retry().hidden, false);
  const bubbles = b.element('chat-log').children.length;
  await b.retry().click();
  assert.deepEqual(b.calls[1], b.calls[0]);
  assert.equal(mutations, 1);
  assert.equal(b.element('chat-log').children.length, bubbles);
  assert.equal(b.retry().hidden, true);
  assert.match(b.element('chat-log').querySelector('.assistant-answer').textContent, /已记录/);
});

test('two deliberately sent identical messages have distinct IDs', async () => {
  const b = browser(() => response([final()]));
  for (let i = 0; i < 2; i++) {
    b.element('chat-input').value = '我花了10块钱';
    await b.context.sendChatMessage();
  }
  assert.equal(b.calls.length, 2);
  assert.notEqual(b.calls[0].request_id, b.calls[1].request_id);
});

test('double retry is single-flight', async () => {
  let release;
  const b = browser(async (_, attempt) => {
    if (attempt === 1) throw new TypeError('offline');
    await new Promise(resolve => { release = resolve; });
    return response([final()]);
  });
  const request = b.context.createChatRequest('我花了10块钱');
  await b.context.executeChatRequest(request);
  const pending = b.context.executeChatRequest(request);
  await b.context.executeChatRequest(request);
  assert.equal(b.calls.length, 2);
  release();
  await pending;
});

test('EOF without final, SSE error and HTTP errors expose manual retry only', async () => {
  for (const failure of [response([]), response([{ type: 'error', message: 'provider error' }]), { ok: false, status: 503 }]) {
    const b = browser(() => failure);
    b.element('chat-input').value = '我花了10块钱';
    await b.context.sendChatMessage();
    assert.equal(b.calls.length, 1);
    assert.equal(b.retry().hidden, false);
    assert.match(b.element('chat-log').querySelector('.assistant-answer').textContent, /结果未确认/);
  }
});

test('a stopped request is not replayed', async () => {
  const b = browser(() => response([{ type: 'status', message: '（已停止）' }]));
  const request = b.context.createChatRequest('我花了10块钱');
  await b.context.executeChatRequest(request);
  assert.equal(request.cancelled, true);
  assert.equal(b.retry().hidden, true);
  await b.context.executeChatRequest(request);
  assert.equal(b.calls.length, 1);
});

test('a persisted cancellation returned on retry invites a new supplement without replay', async () => {
  const b = browser(() => response([{ type: 'final', result: {
    request_status: 'cancelled', answer: '上一轮已中止',
  } }]));
  const request = b.context.createChatRequest('安排周末');
  await b.context.executeChatRequest(request);
  assert.equal(request.cancelled, true);
  assert.equal(b.retry().hidden, true);
  assert.match(request.turn.answerDiv.textContent, /补充条件/);
  await b.context.executeChatRequest(request);
  assert.equal(b.calls.length, 1);
});

for (const currentId of ['failed-draft', 'newer-draft']) {
  test(`failed confirmation invalidates its own card and preserves ${currentId}`, async () => {
    const b = browser(() => response([{ type: 'final', result: {
      answer: '确认未完成，请先核对记录。', used_tool: 'propose_todo',
      tool_error: { kind: 'execution', outcome: 'unknown', invalidated_proposal_id: 'failed-draft' },
    } }]));
    b.context.window.currentProposalId = currentId;
    const request = b.context.createChatRequest('确认', 'failed-draft');
    await b.context.executeChatRequest(request);
    assert.equal(b.context.window.currentProposalId, currentId === 'failed-draft' ? null : 'newer-draft');
    assert.equal(request.completed, true);
    assert.equal(b.retry().hidden, true);
    await b.context.executeChatRequest(request);
    assert.equal(b.calls.length, 1);
    assert.equal(b.calls[0].proposal_id, 'failed-draft');
  });
}

test('known terminal request rejection clears pending storage and exposes no retry', async () => {
  const storage = new Map();
  const b = browser(() => response([{ type: 'error', code: 'proposal_expired', retryable: false,
    invalidated_proposal_id: 'old', message: '草案已失效，请使用最新草案' }]), storage);
  b.context.window.currentProposalId = 'old';
  const request = b.context.createChatRequest('确认', 'old');
  await b.context.executeChatRequest(request);
  assert.equal(request.completed, true);
  assert.equal(b.retry().hidden, true);
  assert.equal(storage.size, 0);
  assert.equal(b.context.window.currentProposalId, null);
  assert.match(request.turn.answerDiv.textContent, /草案已失效/);
  await b.context.executeChatRequest(request);
  assert.equal(b.calls.length, 1);
});

test('retry cannot be sent using another logged-in identity', async () => {
  const b = browser(() => { throw new TypeError('offline'); });
  const request = b.context.createChatRequest('我花了10块钱');
  await b.context.executeChatRequest(request);
  vm.runInContext('sessionId = "bob"', b.context);
  await b.retry().click();
  assert.equal(b.calls.length, 1);
  assert.match(request.turn.answerDiv.textContent, /发送这条消息的账号/);
});

test('proposal confirm and cancel reuse their original proposal and request IDs on retry', async () => {
  for (const [message, action] of [['确认', 'plan_gate_confirm'], ['取消', 'plan_gate_cancel']]) {
    const b = browser((_, attempt) => attempt === 1 ? response([], { broken: true }) : response([final('已处理', action)]));
    b.context.window.currentProposalId = 'proposal-original';
    await b.context.sendProposalDecision(message);
    // The same card button is also a retry path, not a fresh request.
    await b.context.sendProposalDecision(message);
    assert.equal(b.calls[0].message, message);
    assert.equal(b.calls[0].proposal_id, 'proposal-original');
    assert.deepEqual(b.calls[1], b.calls[0]);
    assert.equal(b.context.window.currentProposalId, null);
  }
});

test('final-only cached response succeeds with CRLF event delimiters', async () => {
  const b = browser(() => response([final('缓存结果')], { separator: '\r\n\r\n' }));
  const request = b.context.createChatRequest('我花了10块钱');
  await b.context.executeChatRequest(request);
  assert.equal(request.completed, true);
  assert.equal(b.retry().hidden, true);
  assert.match(request.turn.answerDiv.textContent, /缓存结果/);
});

test('page reload restores an unresolved request without automatically resending it', async () => {
  const storage = new Map();
  const firstPage = browser(() => { throw new TypeError('offline'); }, storage);
  firstPage.element('chat-input').value = '我花了10块钱';
  await firstPage.context.sendChatMessage();
  assert.equal(storage.size, 1);
  const nextPage = browser(() => response([final()]), storage);
  nextPage.context.restorePendingRequests('alice');
  nextPage.context.restorePendingRequests('alice');
  assert.equal(nextPage.calls.length, 0);
  assert.equal(nextPage.element('chat-log').children.length, 2);
  assert.equal(nextPage.retry().hidden, false);
  await nextPage.retry().click();
  assert.deepEqual(nextPage.calls[0], firstPage.calls[0]);
  assert.equal(storage.size, 0);
});

test('pending requests are isolated by identity and cleared on logout', async () => {
  const storage = new Map();
  const b = browser(() => { throw new TypeError('offline'); }, storage);
  const request = b.context.createChatRequest('我花了10块钱');
  await b.context.executeChatRequest(request);
  const anotherUser = browser(() => response([final()]), storage);
  vm.runInContext('sessionId = "bob"', anotherUser.context);
  anotherUser.context.restorePendingRequests('bob');
  assert.equal(anotherUser.element('chat-log').children.length, 0);
  await b.element('logout-btn').click();
  assert.equal(storage.size, 0);
  assert.equal(request.turn.retryBtn.hidden, true);
});

test('unavailable browser storage does not break same-page retry', async () => {
  const b = browser((_, attempt) => attempt === 1 ? response([]) : response([final()]));
  b.context.sessionStorage.setItem = () => { throw new Error('quota exceeded'); };
  b.element('chat-input').value = '我花了10块钱';
  await b.context.sendChatMessage();
  await b.retry().click();
  assert.deepEqual(b.calls[1], b.calls[0]);
});

test('late response after an identity switch does not display an old proposal', async () => {
  let release;
  const b = browser(async () => {
    await new Promise(resolve => { release = resolve; });
    return response([{ type: 'final', result: {
      answer: '旧用户草案', used_tool: 'plan_proposal', proposal_id: 'old-private', proposal: [],
    } }]);
  });
  const request = b.context.createChatRequest('明天开会');
  const pending = b.context.executeChatRequest(request);
  vm.runInContext('sessionId = "bob"', b.context);
  release();
  await pending;
  assert.equal(b.context.window.currentProposalId, undefined);
  assert.equal(request.completed, false);
});


for (const [outcome, expected] of [
  ['some_operations_committed', /部分操作已提交/],
  ['unknown', /写入结果不确定/],
  ['not_completed', /本轮任务未完成/],
]) {
  test(`structured failure outcome ${outcome} remains visible without replay`, async () => {
    const b = browser(() => response([{type: 'final', result: {
      answer: '执行说明', tool_error: {outcome},
    }}]));
    const request = b.context.createChatRequest('执行任务');
    await b.context.executeChatRequest(request);
    assert.match(request.turn.answerDiv.textContent, expected);
    assert.equal(b.retry().hidden, true);
    await b.context.executeChatRequest(request);
    assert.equal(b.calls.length, 1);
  });
}


test('index submission is single-flight and restores both buttons', async () => {
  let release;
  let calls = 0;
  const b = browser(() => {}, new Map(), async () => {
    calls++;
    await new Promise(resolve => { release = resolve; });
    return { ok: true, json: async () => ({ success: true, result: { inserted: 2, skipped: 1 } }) };
  });
  const pending = b.context.runIndex();
  assert.equal(b.element('index-run').disabled, true);
  assert.equal(b.element('index-reset').disabled, true);
  await b.context.runIndex(true);
  assert.equal(calls, 1);
  release();
  await pending;
  assert.equal(b.element('index-run').disabled, false);
  assert.equal(b.element('index-reset').disabled, false);
  assert.match(b.element('index-output').textContent, /新增片段：2/);
});

for (const [status, expected] of [[409, /正在运行/], [422, /超过导入限制/], [401, /重新登录/], [500, /未完整完成/]]) {
  test(`index HTTP ${status} displays a failure and permits manual retry`, async () => {
    const b = browser(() => {}, new Map(), async () => ({ ok: false, status, json: async () => ({ detail: 'internal detail' }) }));
    await b.context.runIndex();
    assert.match(b.element('index-output').textContent, expected);
    assert.doesNotMatch(b.element('index-output').textContent, /internal detail/);
    assert.equal(b.element('index-run').disabled, false);
  });
}

test('index degradation is visible and connection failure never retries automatically', async () => {
  let calls = 0;
  const b = browser(() => {}, new Map(), async () => {
    if (++calls === 1) return { ok: true, json: async () => ({ result: { indexing_status: 'degraded', summary_generation: 'background' } }) };
    throw new TypeError('offline');
  });
  await b.context.runIndex();
  assert.match(b.element('index-output').textContent, /部分完成/);
  assert.match(b.element('index-output').textContent, /后台继续/);
  await b.context.runIndex();
  assert.equal(calls, 2);
  assert.match(b.element('index-output').textContent, /无法确认/);
  assert.equal(b.element('index-reset').disabled, false);
});

test('late indexing response cannot overwrite another identity screen', async () => {
  let release;
  const b = browser(() => {}, new Map(), async () => {
    await new Promise(resolve => { release = resolve; });
    return { ok: true, json: async () => ({ result: { inserted: 99 } }) };
  });
  const pending = b.context.runIndex();
  vm.runInContext('sessionId = "bob"', b.context);
  b.element('index-output').textContent = 'new identity';
  release();
  await pending;
  assert.equal(b.element('index-output').textContent, 'new identity');
  assert.equal(b.element('index-run').disabled, false);
});

test('indexing response is discarded after a login revision changes', async () => {
  let release;
  const b = browser(() => {}, new Map(), async () => {
    await new Promise(resolve => { release = resolve; });
    throw new TypeError('old connection');
  });
  const pending = b.context.runIndex();
  vm.runInContext('identityRevision++', b.context);
  b.element('index-output').textContent = '';
  release();
  await pending;
  assert.equal(b.element('index-output').textContent, '');
});
