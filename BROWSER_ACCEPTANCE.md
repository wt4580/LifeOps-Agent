# 浏览器端到端验收

本流程用于项目发布前的真实浏览器检查。2026-11-02 的隔离 Linux Chromium 自动化已通过下表 14 项检查；原生 Windows 输入法、其他浏览器和真实读屏器不在该验证范围。

## 隔离环境

使用当前源码构建独立演示镜像，避免连接个人数据卷或付费模型：

```bash
docker build --target production -t lifeops-agent:browser-check .
docker run --rm --name lifeops-browser-check --read-only --tmpfs /tmp:rw,size=64m --tmpfs /data:rw,size=128m,uid=10001,gid=10001 -p 127.0.0.1:5012:5000 -e DEMO_MODE=true -e DEMO_SEED=true -e BACKGROUND_LEARNING_ENABLED=false -e SCHEDULER_ENABLED=false lifeops-agent:browser-check
```

在浏览器打开 `http://127.0.0.1:5012`，使用演示账号 `demo`、密码 `lifeops-demo-2026`。上面命令用于手动演示；自动化验收使用下述独立环境。验收结束后终止本次容器；临时数据随容器退出丢弃。

记录实际 Git 提交、镜像 ID、浏览器与版本、屏幕尺寸、时间和每项结果。桌面与窄屏分别检查；不要只写“页面正常”。

## 自动化复现与最新结果

```bash
docker compose -p lifeops-browser-acceptance -f compose.browser-test.yaml up --build --abort-on-container-exit --exit-code-from browser
docker compose -p lifeops-browser-acceptance -f compose.browser-test.yaml down --volumes
```

[逐项报告](tests/browser/results/chromium-acceptance.json)记录 Chromium 156.0.8078.4、开始/结束时间、应用源码及静态资源指纹，结果为 **14/14**。桌面视口 1365×900、窄屏 390×844；本地结果和截图位于被忽略的 `artifacts/browser-acceptance/`。报告指纹已与当前应用源码核对一致。

应用代码基线为 `f82768c`。本轮实际应用镜像 ID 为 `sha256:9c6026fca4e71deb986e8726bf192275f187af14f2e5fa571829b2a25ce5a6a7`，浏览器镜像 ID 为 `sha256:d246bb16c25ed4702373cef5c3b0535130b9b2ee10f4d8a3e3006bcca068684a`；镜像标识是本地运行记录，不表示已发布到远程仓库。

测试使用实际 HTML/CSS/JavaScript、HTTP/SSE 和临时业务数据库。应用使用离线规则模型，增加 0.6 秒响应延迟以稳定触发停止等操作；不消耗模型额度。B05由受控代理在服务端产生最终结果后丢弃响应，核对相同请求 ID 重试后仅一条账目和一个用户气泡；B09注入降级响应元数据检查前端提示；B10/B11延迟真实响应并核对最终视图与身份。登录限流保持启用，案例按窗口分批执行。

## 检查清单

| 编号 | 操作 | 应观察到的结果 | 本轮状态 |
| --- | --- | --- | --- |
| B01 | 登录；重复点击登录或重复按 Enter | 同时只有一个登录请求，按钮恢复，账号正确 | 通过（Chromium） |
| B02 | 输入中文并按 Enter 选词，再正常发送 | 选词不发送；完成输入后正常发送；Shift+Enter换行 | 合成IME事件通过；原生IME未验 |
| B03 | 发送“我花了10元买午饭” | 展示进度、正文和最终结果；实际记录一次 | 通过（Chromium） |
| B04 | 请求“明天下午3点开会”，先取消，再重新提出并确认 | 确认前不写待办；取消不写；确认仅创建一次 | 通过（Chromium） |
| B05 | 发送后中断网络，恢复网络并手动重试原请求 | 不重复增加用户气泡或业务写入，恢复原请求结果；不自动重试写操作 | 通过（Chromium） |
| B06 | 主动停止回答，随后补充条件 | 停止状态可见；补充作为新请求，已提交操作不重复执行 | 通过（Chromium） |
| B07 | 问“报销需要哪些材料” | 答案及引用显示；章节和参考资料标签可读，展开后可看正文 | 通过（Chromium） |
| B08 | 在知识库问答面板发送同一问题 | 正确显示统一响应中的答案和引用，无 undefined 文本 | 通过（Chromium） |
| B09 | 运行索引，并在进行中重复点击运行或重置 | 两个按钮禁用；结束后恢复；降级状态不得显示为完整成功 | 通过（Chromium） |
| B10 | 快速切换待办“今天/全部/未来7天” | 最终列表与最后选择的标签一致，迟到响应不覆盖当前视图 | 通过（Chromium） |
| B11 | 问答或索引未返回时退出，再登录另一演示身份 | 旧内容清空，旧响应不出现在新账号页面 | 通过（Chromium） |
| B12 | 输入1001个字符，包含emoji边界 | 本地提示超限，不发送；聊天保留原文；1000个Unicode字符可提交 | 通过（Chromium） |
| B13 | 检查约390px宽度及桌面宽度 | 输入、发送、停止、确认及侧栏入口均可操作，无关键按钮被遮挡 | 通过（Chromium） |
| B14 | 仅用键盘操作登录、聊天、引用展开 | 焦点可见且顺序合理；输入有可访问名称；动态状态可被感知 | 键盘与ARIA检查通过 |

B05/B06需要在请求仍运行时操作。演示响应可能过快；无法实际触发时记为“未验证”，不能记为通过。并发过载、依赖故障等自动化验证另见统一验收报告；本清单不要求为了制造错误而修改现有部署。

## 结果与通过条件

每项记录：通过、失败、未验证或不适用，并附复现输入及观察结果。关键项 B01–B11有失败时不宣称端到端验收通过；不适用需说明原因，未验证项保持开放。

截图仅使用虚构演示资料，放入被忽略的 `artifacts/browser-acceptance/`。浏览器控制台异常、HTTP状态、重复写入和错误身份展示均应单独记录。分享日志前检查Cookie、密码、API Key及个人资料。

本流程验证浏览器行为与演示链路，不衡量真实模型完成率、语义准确性、生产负载或长期稳定性。

## Windows 浏览器工具尝试（历史）

版本 `44f7454` 已构建隔离镜像 `lifeops-agent:browser-44f7454`，临时容器使用内存数据目录、只读根文件系统、离线演示模型，映射至 `127.0.0.1:5012`；容器健康检查通过。用户已确认在 Edge 手动登录并保持页面打开。

Computer Use 两次无法可靠识别当前浏览器 URL，强制停止控制；手动登录后再次观察仍被同一检查停止。该次工具尝试没有完成 B01–B14，用户登录确认也不等于验收通过。随后采用上文记录的独立 Linux Chromium 环境完成自动化检查；没有继续控制用户 Edge 窗口。
