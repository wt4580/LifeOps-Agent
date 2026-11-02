"""Real Chromium + offline HTTP/SSE acceptance; no personal browser or paid model."""
import argparse
import asyncio
import json
import secrets
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
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        def checkpoint(complete=False):
            report = {'run_id': run_id, 'browser': browser.version, 'engine': 'Chromium',
                      'scope': 'Linux headless; isolated offline fixture',
                      'paid_model_calls': 0, 'complete': complete, 'cases': results,
                      'not_yet_covered': ['B02', 'B05', 'B06', 'B09', 'B10', 'B11']}
            (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            return report
        checkpoint()
        for case_id in ['B01', 'B03', 'B04', 'B07', 'B08', 'B12', 'B13', 'B14']:
            context = await browser.new_context(viewport={'width': 1365, 'height': 900})
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

                if case_id == 'B01':
                    assert sum('/api/auth/login' in url for url in requests) == 1
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
                elif case_id in {'B07', 'B08'}:
                    await page.locator('#toggle-rag').click()
                    await page.locator('#index-run').click()
                    await expect(page.locator('#index-run')).to_be_enabled(timeout=30000)
                    if case_id == 'B07':
                        await send('报销需要哪些材料')
                    else:
                        await page.locator('[data-tab="ask"]').click()
                        await page.locator('#ask-input').fill('报销需要哪些材料')
                        await page.locator('#ask-send').click()
                        await expect(page.locator('#ask-answer')).to_contain_text('演示回答')
                    await expect(page.locator('#ask-citations summary').first).to_contain_text('参考资料')
                    await page.locator('#ask-citations summary').first.click()
                    await expect(page.locator('#ask-citations .citation-body').first).to_be_visible()
                    assert 'undefined' not in await page.locator('#ask-answer').inner_text()
                elif case_id == 'B12':
                    before = sum('/api/chat/stream' in url for url in requests)
                    text = '😀' * 1001
                    await page.locator('#chat-input').fill(text)
                    await page.locator('#chat-send').click()
                    assert await page.locator('#chat-input').input_value() == text
                    assert sum('/api/chat/stream' in url for url in requests) == before
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
