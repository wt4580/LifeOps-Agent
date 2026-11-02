"""Real Chromium + offline HTTP/SSE acceptance; no personal browser or paid model."""
import argparse
import asyncio
import json
import secrets
import urllib.request
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from playwright.async_api import async_playwright, expect


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://localhost:5000')
    parser.add_argument('--output', default='/artifacts/browser-acceptance')
    args = parser.parse_args()
    origin = urlsplit(args.base_url)
    if origin.hostname not in {'localhost', '127.0.0.1'} or origin.scheme != 'http':
        raise ValueError('Only the disposable loopback HTTP fixture is supported')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    run_id = secrets.token_hex(3)
    started_at = datetime.now(timezone.utc).isoformat()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        fixture_info = {}
        def checkpoint(complete=False):
            report = {'run_id': run_id, 'browser': browser.version, 'engine': 'Chromium',
                      'started_at': started_at,
                      'finished_at': datetime.now(timezone.utc).isoformat() if complete else None,
                      'scope': 'Linux headless; isolated offline fixture',
                      'paid_model_calls': 0, 'complete': complete, 'cases': results,
                      'fixture': fixture_info, 'not_yet_covered': [],
                      'limitations': ['IME uses browser composition events, not a native Windows input method',
                                      'Linux Chromium headless; no physical screen-reader or other-browser claim']}
            (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            return report
        checkpoint()
        probe = await playwright.request.new_context()
        response = await probe.get(args.base_url + '/__acceptance/info', timeout=15000)
        assert response.ok, 'Only the dedicated offline browser fixture may be tested'
        fixture_info = await response.json()
        assert fixture_info.get('fixture') == 'lifeops-offline-browser' and fixture_info.get('demo_mode') is True
        await probe.dispose()
        last_login_at = None
        for case_id in [f'B{index:02d}' for index in range(1, 15)]:
            if case_id == 'B11' and last_login_at is not None:
                # The real service allows ten logins per peer per minute.
                # Keep it enabled and let the first batch expire before identity tests.
                delay = max(0, 61 - (time.monotonic() - last_login_at))
                print(f'Waiting {delay:.1f}s for login rate window', flush=True)
                await asyncio.sleep(delay)
            context = await browser.new_context(viewport={'width': 1365, 'height': 900}, timezone_id='Asia/Shanghai')
            page = await context.new_page()
            page.set_default_timeout(15000)
            errors, requests = [], []
            page.on('pageerror', lambda error: errors.append(str(error)[:300]))
            page.on('request', lambda request: requests.append(request.url))
            async def restrict(route):
                url = urlsplit(route.request.url)
                if (url.scheme, url.hostname, url.port) == (origin.scheme, origin.hostname, origin.port):
                    await route.continue_()
                else:
                    await route.abort()
            await context.route('**/*', restrict)
            username = f'browser-{run_id}-{case_id.lower()}'
            try:
                await page.goto(args.base_url, wait_until='domcontentloaded')
                assert not errors, f'Browser startup errors: {errors}'
                await page.locator('#login-input').fill(username)
                await page.locator('#login-password').fill('offline-browser-test-password')
                await page.locator('#login-password').press('Enter')
                if case_id == 'B01':
                    await page.keyboard.press('Enter')
                await expect(page.locator('#login-section')).to_be_hidden()
                last_login_at = time.monotonic()
                await expect(page.locator('#chat-input')).to_be_visible()

                async def send(text):
                    await page.locator('#chat-input').fill(text)
                    async with page.expect_response(lambda response: '/api/chat/stream' in response.url) as pending:
                        await page.locator('#chat-send').click()
                    response = await pending.value
                    assert response.ok, f'Chat HTTP status {response.status}'
                    # A fetch/SSE response may remain open after its final event.
                    # Observe the UI's terminal state with a bounded wait instead.
                    await expect(page.locator('.assistant-stop-btn').last).to_be_hidden(timeout=15000)
                    await expect(page.locator('.assistant-retry-btn').last).to_be_hidden()
                    await expect(page.locator('#chat-input')).to_have_value('')

                async def api(path):
                    separator = '&' if '?' in path else '?'
                    response = await context.request.get(args.base_url + path + separator + 'session_id=' + username)
                    assert response.ok, f'API status {response.status}'
                    data = await response.json()
                    return data.get('result', data.get('data'))

                async def index():
                    if not await page.locator('#rag-section').is_visible():
                        await page.locator('#toggle-rag').click()
                    await page.locator('[data-tab="index"]').click()
                    await page.locator('#index-run').click()
                    await expect(page.locator('#index-run')).to_be_enabled(timeout=30000)

                async def ask():
                    await page.locator('[data-tab="ask"]').click()
                    await page.locator('#ask-input').fill('报销需要哪些材料')
                    await page.locator('#ask-send').click()

                if case_id == 'B01':
                    assert sum('/api/auth/login' in url for url in requests) == 1
                elif case_id == 'B02':
                    before = sum('/api/chat/stream' in url for url in requests)
                    await page.locator('#chat-input').fill('中文输入验证')
                    await page.locator('#chat-input').dispatch_event('keydown', {'key': 'Enter', 'isComposing': True, 'keyCode': 229})
                    assert await page.locator('#chat-input').input_value() == '中文输入验证'
                    assert sum('/api/chat/stream' in url for url in requests) == before
                    await page.locator('#chat-input').press('Shift+Enter')
                    assert '\n' in await page.locator('#chat-input').input_value()
                    async with page.expect_response(lambda response: '/api/chat/stream' in response.url):
                        await page.locator('#chat-input').press('Enter')
                    await expect(page.locator('.assistant-stop-btn').last).to_be_hidden(timeout=15000)
                    assert sum('/api/chat/stream' in url for url in requests) == before + 1
                elif case_id == 'B03':
                    await send('我花了10元买午饭')
                    await expect(page.locator('#chat-log')).to_contain_text('已记录支出')
                    rows = await api('/api/transactions')
                    assert len(rows) == 1 and rows[0]['amount'] == -10
                elif case_id == 'B04':
                    await send('明天下午3点开会')
                    await expect(page.locator('#proposal-card')).to_be_visible()
                    assert not (await api('/api/todos/all'))['pending']
                    await page.locator('#reject-proposal').click()
                    await expect(page.locator('#proposal-card')).to_be_hidden()
                    assert not (await api('/api/todos/all'))['pending']
                    await send('明天下午3点开会')
                    await expect(page.locator('#proposal-card')).to_be_visible()
                    await page.locator('#confirm-proposal').click()
                    await expect(page.locator('#proposal-card')).to_be_hidden()
                    await expect(page.locator('#chat-log')).to_contain_text('已添加到待办')
                    assert len((await api('/api/todos/all'))['pending']) == 1
                elif case_id == 'B05':
                    attempts = []
                    async def lose_ack(route):
                        attempts.append(route.request.post_data_json)
                        if len(attempts) != 1:
                            await route.continue_()
                            return
                        headers = await route.request.all_headers()
                        url, body = route.request.url, route.request.post_data.encode('utf-8')
                        assert url.startswith(args.base_url + '/')
                        def commit_then_disconnect():
                            request = urllib.request.Request(url, data=body, headers={
                                'Content-Type': 'application/json', 'Cookie': headers.get('cookie', ''),
                                'Origin': args.base_url})
                            with urllib.request.urlopen(request, timeout=15) as stream:
                                for _ in range(300):
                                    line = stream.readline()
                                    if not line:
                                        break
                                    if line.startswith(b'data: ') and json.loads(line[6:]).get('type') == 'final':
                                        return
                            raise AssertionError('Server did not commit a final result')
                        await asyncio.wait_for(asyncio.to_thread(commit_then_disconnect), timeout=20)
                        await route.abort()
                    await page.route('**/api/chat/stream', lose_ack)
                    await page.locator('#chat-input').fill('我花了10元买午饭')
                    await page.locator('#chat-send').click()
                    await expect(page.locator('.assistant-retry-btn').last).to_be_visible(timeout=25000)
                    assert len(await api('/api/transactions')) == 1
                    await page.locator('.assistant-retry-btn').last.click()
                    await expect(page.locator('#chat-log')).to_contain_text('已记录支出')
                    assert len(attempts) == 2 and attempts[0]['request_id'] == attempts[1]['request_id']
                    assert len(await api('/api/transactions')) == 1
                    assert await page.locator('#chat-log').get_by_text('我花了10元买午饭', exact=True).count() == 1
                elif case_id == 'B06':
                    payloads = []
                    def capture(request):
                        if '/api/chat/stream' in request.url:
                            payloads.append(request.post_data_json)
                    page.on('request', capture)
                    await page.locator('#chat-input').fill('明天下午3点开会')
                    await page.locator('#chat-send').click()
                    await expect(page.locator('.assistant-stop-btn').last).to_be_visible()
                    await page.locator('.assistant-stop-btn').last.click()
                    await expect(page.locator('#chat-log')).to_contain_text('已停止', timeout=15000)
                    await send('后天下午4点开会')
                    await expect(page.locator('#proposal-card')).to_be_visible()
                    await page.locator('#confirm-proposal').click()
                    await expect(page.locator('#chat-log')).to_contain_text('已添加到待办')
                    assert len((await api('/api/todos/all'))['pending']) == 1
                    assert len(payloads) >= 2 and payloads[0]['request_id'] != payloads[1]['request_id']
                elif case_id in {'B07', 'B08'}:
                    await index()
                    if case_id == 'B07':
                        await send('报销需要哪些材料')
                    else:
                        await ask()
                        await expect(page.locator('#ask-answer')).to_contain_text('演示回答')
                    await expect(page.locator('#ask-citations summary').first).to_contain_text('参考资料')
                    await page.locator('#ask-citations summary').first.click()
                    await expect(page.locator('#ask-citations .citation-body').first).to_be_visible()
                    assert 'undefined' not in await page.locator('#ask-answer').inner_text()
                elif case_id == 'B09':
                    started, release = asyncio.Event(), asyncio.Event()
                    count = 0
                    async def hold_index(route):
                        nonlocal count
                        count += 1
                        started.set()
                        response = await route.fetch(timeout=30000)
                        await asyncio.wait_for(release.wait(), timeout=10)
                        await route.fulfill(response=response)
                    await page.route('**/api/index', hold_index)
                    await page.locator('#toggle-rag').click()
                    await page.locator('#index-run').click()
                    await asyncio.wait_for(started.wait(), timeout=5)
                    await expect(page.locator('#index-run')).to_be_disabled()
                    await expect(page.locator('#index-reset')).to_be_disabled()
                    await page.locator('#index-run').evaluate('(button) => button.click()')
                    release.set()
                    await expect(page.locator('#index-run')).to_be_enabled(timeout=30000)
                    await expect(page.locator('#index-reset')).to_be_enabled()
                    assert count == 1
                    await page.unroute('**/api/index', hold_index)
                    async def degraded_index(route):
                        response = await route.fetch(timeout=30000)
                        data = await response.json()
                        data['result']['indexing_status'] = 'degraded'
                        await route.fulfill(response=response, json=data)
                    await page.route('**/api/index', degraded_index)
                    await page.locator('#index-run').click()
                    await expect(page.locator('#index-output')).to_contain_text('部分完成')
                    await expect(page.locator('#index-output')).not_to_contain_text('索引建立完成')
                elif case_id == 'B10':
                    for text in ['明天下午3点开会唯一甲', '后天下午4点开会唯一乙']:
                        await send(text)
                        await expect(page.locator('#proposal-card')).to_be_visible()
                        await page.locator('#confirm-proposal').click()
                        await expect(page.locator('#proposal-card')).to_be_hidden()
                    started, release, done = asyncio.Event(), asyncio.Event(), asyncio.Event()
                    async def hold_today(route):
                        response = await route.fetch()
                        started.set()
                        await asyncio.wait_for(release.wait(), timeout=10)
                        await route.fulfill(response=response)
                        done.set()
                    await page.route('**/api/todos/today*', hold_today)
                    await page.locator('#todos-today').click()
                    await asyncio.wait_for(started.wait(), timeout=5)
                    await page.locator('#todos-all').click()
                    await expect(page.locator('#todos-output')).to_contain_text('唯一乙')
                    release.set()
                    await asyncio.wait_for(done.wait(), timeout=5)
                    await page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
                    await expect(page.locator('#todos-output')).to_contain_text('唯一乙')
                    await expect(page.locator('#todos-output')).to_contain_text('唯一甲')
                elif case_id == 'B11':
                    await index()
                    started, release, done = asyncio.Event(), asyncio.Event(), asyncio.Event()
                    async def hold_answer(route):
                        response = await route.fetch(timeout=30000)
                        started.set()
                        await asyncio.wait_for(release.wait(), timeout=15)
                        await route.fulfill(response=response)
                        done.set()
                    await page.route('**/api/ask', hold_answer)
                    await ask()
                    await asyncio.wait_for(started.wait(), timeout=10)
                    await page.locator('#logout-btn').click()
                    await expect(page.locator('#login-section')).to_be_visible()
                    await page.locator('#login-input').fill(username + '-other')
                    await page.locator('#login-password').fill('offline-browser-test-password')
                    await page.locator('#login-password').press('Enter')
                    await expect(page.locator('#login-section')).to_be_hidden()
                    release.set()
                    await asyncio.wait_for(done.wait(), timeout=5)
                    await page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
                    await expect(page.locator('#ask-answer')).to_have_text('')
                    assert await page.locator('#ask-citations details').count() == 0
                elif case_id == 'B12':
                    before = sum('/api/chat/stream' in url for url in requests)
                    text = '😀' * 1001
                    await page.locator('#chat-input').fill(text)
                    await page.locator('#chat-send').click()
                    assert await page.locator('#chat-input').input_value() == text
                    assert sum('/api/chat/stream' in url for url in requests) == before
                    await send('😀' * 1000)
                    assert sum('/api/chat/stream' in url for url in requests) == before + 1
                elif case_id == 'B13':
                    await page.screenshot(path=str(output / 'desktop.png'), full_page=True)
                    await page.set_viewport_size({'width': 390, 'height': 844})
                    await expect(page.locator('#chat-send')).to_be_in_viewport()
                    await expect(page.locator('#mobile-menu-toggle')).to_be_visible()
                    assert await page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                    await page.locator('#mobile-menu-toggle').click()
                    await expect(page.locator('#todos-all')).to_be_in_viewport()
                    await page.screenshot(path=str(output / 'mobile-panel.png'), full_page=True)
                    await page.locator('#close-side-panel').click()
                    await expect(page.locator('#chat-send')).to_be_in_viewport()
                    await expect(page.locator('#mobile-menu-toggle')).to_have_attribute('aria-expanded', 'false')
                    await page.locator('#mobile-menu-toggle').click()
                    await page.keyboard.press('Escape')
                    await expect(page.locator('#mobile-menu-toggle')).to_have_attribute('aria-expanded', 'false')
                    await send('明天下午3点开会')
                    await expect(page.locator('#confirm-proposal')).to_be_in_viewport()
                    await expect(page.locator('#reject-proposal')).to_be_in_viewport()
                    await page.locator('#confirm-proposal').click()
                    await expect(page.locator('#chat-log')).to_contain_text('已添加到待办')
                    await page.locator('#chat-input').fill('你好')
                    await page.locator('#chat-send').click()
                    await expect(page.locator('.assistant-stop-btn').last).to_be_in_viewport()
                    await page.locator('.assistant-stop-btn').last.click()
                    await expect(page.locator('#chat-log')).to_contain_text('已停止')
                    await page.screenshot(path=str(output / 'mobile.png'), full_page=True)
                elif case_id == 'B14':
                    assert await page.locator('#chat-input').get_attribute('aria-label')
                    assert await page.locator('#ask-answer').get_attribute('aria-live') == 'polite'
                    found = False
                    for _ in range(20):
                        await page.keyboard.press('Tab')
                        if await page.locator('#chat-input').evaluate('(element) => element === document.activeElement'):
                            found = True
                            break
                    assert found, 'Chat input was not reachable by keyboard'
                    await index()
                    await ask()
                    await expect(page.locator('#ask-citations summary').first).to_be_visible()
                    await page.keyboard.press('Tab')
                    await expect(page.locator('#ask-citations summary').first).to_be_focused()
                    assert await page.locator('#ask-citations summary').first.evaluate(
                        '(element) => getComputedStyle(element).outlineStyle !== "none"')
                    await page.keyboard.press('Enter')
                    assert await page.locator('#ask-citations details').first.get_attribute('open') is not None
                assert not errors, f'Browser errors: {errors}'
                results.append({'id': case_id, 'status': 'passed'})
            except Exception as error:
                results.append({'id': case_id, 'status': 'failed', 'error': str(error)[:1200]})
                await page.screenshot(path=str(output / f'{case_id}-failed.png'), full_page=True)
            finally:
                checkpoint()
                await asyncio.wait_for(context.close(), timeout=10)
        report = checkpoint(complete=True)
        print(json.dumps(report, ensure_ascii=False))
        await browser.close()
    return int(any(result['status'] != 'passed' for result in results))


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
