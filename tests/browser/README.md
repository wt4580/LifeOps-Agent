# Chromium 浏览器验收

使用真正的 Linux Chromium headless shell、当前应用、HTTP/SSE 和临时 SQLite。应用运行离线演示模型，浏览器与应用共享无外网的网络命名空间，只访问 loopback 服务；不需要个人浏览器或 API Key。

在仓库根目录执行：

```bash
docker compose -p lifeops-browser-acceptance -f compose.browser-test.yaml up --build --abort-on-container-exit --exit-code-from browser
docker compose -p lifeops-browser-acceptance -f compose.browser-test.yaml down --volumes
```

首次构建需要下载 Chromium 及系统依赖；Python 包与浏览器固定为 Playwright 1.64.1。可通过 `PIP_INDEX_URL` 选择 HTTPS Python 镜像。

结果及截图写入被忽略的 `artifacts/browser-acceptance/`。非零退出表示有检查失败。`report.json` 逐项记录结果和未覆盖项，截图使用虚构测试账号及数据，不输出 Cookie、API Key 或网络请求体。

当前脚本包含 B01、B03、B04、B07、B08、B12、B13、B14 的初始检查；B02、B05、B06、B09、B10、B11 尚未覆盖，B14 的引用键盘操作仍待补充。新增脚本尚需实际运行验证，不等同于全部浏览器验收通过。

`serve.py` 仅在这个隔离测试环境为演示模型增加 0.6 秒延迟，便于观测异步状态；不修改模型结果或生产入口。浏览器结果不代表真实模型质量、Windows 输入法、实际读屏器或生产吞吐。
