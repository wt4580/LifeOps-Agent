# Chromium 浏览器验收

使用真正的 Linux Chromium headless shell、当前应用、HTTP/SSE 和临时 SQLite。应用运行离线演示模型，浏览器与应用共享无外网的网络命名空间，只访问 loopback 服务；不需要个人浏览器或 API Key。

在仓库根目录执行：

```bash
docker compose -p lifeops-browser-acceptance -f compose.browser-test.yaml up --build --abort-on-container-exit --exit-code-from browser
docker compose -p lifeops-browser-acceptance -f compose.browser-test.yaml down --volumes
```

首次构建需要下载 Chromium 及系统依赖；Python 包与浏览器固定为 Playwright 1.64.1。可通过 `PIP_INDEX_URL` 选择 HTTPS Python 镜像。

结果及截图写入被忽略的 `artifacts/browser-acceptance/`。非零退出表示有检查失败。`report.json` 逐项记录结果和未覆盖项，截图使用虚构测试账号及数据，不输出 Cookie、API Key 或网络请求体。

当前 B01–B14 已实际运行并全部通过，详见[逐项报告](results/chromium-acceptance.json)及[完整验收说明](../../BROWSER_ACCEPTANCE.md)。覆盖登录防重入、合成IME事件、记账、确认/取消、提交后响应丢失重试、主动停止续问、RAG引用、索引防重入/降级提示、迟到响应、账号切换、Unicode边界、窄屏及键盘操作。

`serve.py` 使用生产应用入口，仅在隔离测试环境为演示模型增加 0.6 秒延迟并提供来源指纹端点；不修改模型结果。脚本先验证专用离线夹具身份后才执行写操作，逐案例落盘进度，SSE 终态和资源关闭均有有限等待。登录限流保持启用，案例在需要时等待真实的一分钟窗口。浏览器结果不代表真实模型质量、Windows 输入法、实际读屏器或生产吞吐。
