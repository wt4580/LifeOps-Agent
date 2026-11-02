# 评估记录与验证边界

下列记录保留执行时的版本与范围。最新结果见[项目 README](../../../README.md#测试与评估)；历史记录中的待办不代表当前状态。所有命令从仓库根目录运行。

### 最新验证范围

2026-11-01 当前源码 `6b20053` 的补充验收：Windows 真实回环 TCP 8并发、32个操作全部成功，各用户唯一请求与写入数一致；四项过载恢复检查通过。两轮源码指纹相同且未变化，报告为本地 `artifacts/load-current-6b20053.json` 与 `artifacts/overload-current-6b20053.json`。首段正文与最终结果 P95约653毫秒、1.49秒，仅衡量离线演示链路。本轮同时通过40项前端脚本测试，包含索引/问答响应、身份隔离、中文输入法和退出失败反馈；日志为 `artifacts/frontend-current-6b20053.log`。未重新执行完整七阶段验收。

2026-11-01 当前源码 `d2edb4e` 完整回归：Windows 后端 **661 项通过、1 项跳过**，含1478条依赖弃用等警告；前端脚本 **43 项通过**，语法检查通过。覆盖查询输入边界、HTTP拒绝前无执行、后台生命周期和前端异步身份隔离。报告保存在本地 `artifacts/current-d2edb4e-unit.xml`、`artifacts/current-d2edb4e-unit.log`、`artifacts/current-d2edb4e-frontend.log`。本次未复跑完整七阶段验收，不替代下表各项原有证据。

以下结果来自 **2026-10-31 本地执行**，每项对应自己的源码版本与环境，不混算为同一轮验收。

| 验证层级 | 结果与范围 | 证据位置 |
| --- | --- | --- |
| 当前 production 镜像 | 2026-11-01 源码 `f88c3da` 新构建；非root、只读、无网络、临时数据下8项HTTP/SSE冒烟通过，含确认重放仅写一次 | 本地 `artifacts/production-f88c3da-*` |
| Linux 七阶段验收 | 2026-11-01 源码 `3d6e9d0` 新构建test镜像：672项通过、1项跳过、6条警告；七阶段全部通过，源码未变化 | 本地 `artifacts/image-verification-3d6e9d0/` |
| 模拟 Agent 流程 | 同轮 100/100；覆盖确认、取消、修改、多日期、记账、账户变更、引用、权限、预算与供应商故障 | 同轮 `e2e.json` |
| 跨进程恢复 | 同轮 14 个检查点通过，含响应保存失败恢复、回执重放和待确认 checkpoint 恢复 | 同轮 `recovery.json` |
| TCP 并发与过载 | 同轮 4 并发、16 个操作全部成功；按用户核对唯一写入，四项过载与容量恢复检查通过 | 同轮 `concurrency.json`、`overload.json` |
| 浏览器请求生命周期 | 源码 `d2edb4e`：43 项测试通过，覆盖重试、停止、身份隔离、索引/问答、输入法和存储容错 | 本地 `artifacts/current-d2edb4e-frontend.log` |
| 真实 FAISS 集成 | Windows 中文路径下保存、加载、召回、重建、空库发布和过期候选拦截通过；连同发布故障测试共 10 项 | `tests/test_vector_integration.py`、`tests/test_vector_publication.py` |

Linux 验收使用已有镜像 `lifeops-agent:test-d4678f5` 只读挂载当前源码，未重建该镜像或更新部署；统一报告为 `complete=true, passed=true, source_unchanged=true`，源码指纹 `0cca8d0d6c5704f66872cdf9b588f402577f225289fe4acd4437cfab9834c45e`。核心镜像缺少可选 FAISS 依赖，该用例跳过；Windows 真实 FAISS 测试使用确定性本地向量，不下载模型，也不衡量语义质量。

`artifacts/` 是本地生成且被忽略的验收产物，新克隆仓库需运行上述命令生成。仓库内的[流程报告](e2e-contract.md)、[逐例结果](e2e-contract.json)、[检索报告](retrieval.md)和[路由报告](task.json)属于历史快照，不能代替当前版本验收。

production 验收镜像为 `lifeops-agent:acceptance-f88c3da`，实际镜像标识 `sha256:1a44cc41d61c310de63dd99f5e8acd51fc5ac0874f8baf9cab6bb0554bb4ffcc`。构建使用清华 HTTPS PyPI 源，运行用户为 `lifeops`；临时容器已清理，未修改既有部署或数据卷。本轮为真实 HTTP/SSE 冒烟，不代替完整七阶段、浏览器、kind 或真实模型验收。

当前test镜像验收标识为 `sha256:f7808f61f0f2e0bfbd02f1e9a5be0590427467d9972dd9e030dba2f59eef670f`，完整报告 `complete=true, passed=true, source_unchanged=true`，容器退出码0，报告导出后临时容器与匿名卷已清理。本轮直接执行镜像内源码；test目标继承了运行时HTTP健康探针，批处理测试期间没有常驻5000端口，因此容器Health记录为unhealthy，此项不用于判定测试进程或production健康，后续已在test目标显式设置 `HEALTHCHECK NONE`，重新构建后检查实际镜像元数据确认生效；production仍保留readiness探针。探针修正后的镜像本轮仅完成构建与元数据验证，未重新跑七阶段。

## 模型与检索评估记录

```bash
# 虚构语料与人工相关性标签，无需 API Key
python -m app.eval.run_reference --output artifacts/reference.json

# 同案例、同模式与同语料比较，出现成功转失败时非零退出
python -m app.eval.compare artifacts/baseline.json artifacts/candidate.json --output artifacts/comparison.json

# 先预览8类场景的小规模计划，不启动应用、不调用模型
python -m app.eval.run_e2e --mode real --plan-only --limit 8 --turn-max-llm-calls 8 --turn-max-tokens 20000 --output artifacts/real-plan.json

# 实际真实模型模式会产生费用，必须显式授权
python -m app.eval.run_e2e --mode real --allow-paid-calls --output artifacts/e2e-real.json
```

[参考语料](../../../examples/reference_eval/)含 5 份虚构资料，[开发集](../../../app/eval/reference_cases.json)含 15 个有相关资料的问题与 2 个库外问题。已测文档 Recall@1 为 **0.90**、Recall@5 为 **1.00**，两个库外问题均无结果；门槛为 Recall@5 ≥ 0.80 且库外问题无结果。指标按文件去重，版本 2 区分 Recall 与 HitRate；合成开发集不能证明真实用户问题或生成答案质量。

逐例比较要求模式、模型、语料、检索后端、案例内容指纹和案例 ID 集合一致；案例含义改变后需建立新基线。模拟 E2E 验证程序契约；真实模式运行 80 条流程用例，排除模拟故障与模拟跨用户 SQL 场景。`--turn-max-llm-calls` 和 `--turn-max-tokens` 可覆盖评估中每轮的Harness预算，计划及执行报告都会记录；它们不是总token、供应商重试次数或货币账单上限。真实模型保留集、向量/混合检索消融和生产负载验收仍待完成。

报告中每个案例的 `execution` 保留轮次终止状态、是否触发预算及 SSE 错误码，便于区分业务断言失败与执行提前终止。离线演示模型不返回供应商 token 用量，Harness 会保守扣除预留额度：相同首批 8 类案例在每轮 20,000 token 上限下为 7/8，在默认 120,000 上限下为 8/8（每轮调用上限均为 8）。这说明缩小预算可能截断多步流程，不能把此处的预留扣账当作真实模型消耗，也不能将离线通过率当作模型准确率。

首轮 [qwen3.8-flash 真实模型报告](real-flash-pilot.json)覆盖 8 类、15 条消息：通过 7/8，观察到 17 次模型调用、供应商报告 66,190 token；案例总耗时 P50 为 52.7 秒，P95 为 113.2 秒（包含多轮确认，不是单次请求延迟）。多日期待办只创建一条，且未触发预算；已针对纯待办多步计划补充合并草案及统一确认。此报告保留修复前失败结果，不表示修复后的完整评估已通过，也不衡量答案语义质量或生产 SLO。

可用 `--case-id multi-0` 仅复测指定案例，多次传入可选多项；未知 ID 或真实模式中的模拟专用案例会在启动应用前被拒绝。单案例复测不能代替完整回归。

多待办合并修复后，100 条离线流程契约全部通过；补充的完整图验证覆盖确认前零写入、一次确认写入两条不同日期待办、取消后零写入。此验证证明批量确认的程序行为，真实模型复测结果需单独判断。

2026-11-02 的 [多日期单案例真实复测](real-flash-multi-timeout.json)未通过：两轮均达到 120 秒上限，最终零条待办；无供应商用量返回，40,000 是保守预留扣账，不是已测实际 token。该结果不能证明修复后的真实模型任务成功，模型服务延迟与配置仍需排查。

支持混合思考的 Qwen 模型可显式设置 `LLM_ENABLE_THINKING=false` 评估交互延迟；省略此配置时不向供应商发送该字段。`true` 表示启用思考，仅支持思考的模型不可关闭，其他兼容服务需确认参数支持情况。流式用量可设置 `LLM_STREAM_INCLUDE_USAGE=true`。参数按[供应商文档](https://www.alibabacloud.com/help/en/model-studio/deep-thinking)通过 `extra_body` 发送；执行报告记录配置，配置或预算不一致时拒绝直接作为同条件回归比较，需另建基线。

关闭思考并启用流式用量后，[修复后首批真实回归](real-flash-no-thinking-pilot.json)为 **7/8**：多日期待办通过，账户余额变更未通过（第二轮工具失败，第三轮仍待确认）。观察到 19 次调用、48,912 个供应商报告 token；案例总耗时 P50/P95 为 4.8/11.4 秒。该配置变化和代码修复同时存在，不能据此单独归因性能改善。原文保留修复后的离线契约再次 **100/100** 通过；完整真实回归、重复采样稳定性和账户失败定位仍开放。

补充脱敏工具失败诊断后，[账户确认定向抽样](real-flash-account-sample.json)为 3/3 通过，尚未复现此前失败；这不是账户偶发问题已解决的证据。执行诊断只保留动作名、错误码和异常类型，不保存工具参数及原始错误消息。

完整 [80 条真实流程基线](real-flash-full-baseline.json)通过 **65/80（81.25%）**，观察到 207 次模型调用、504,049 个供应商报告 token；案例总耗时 P50/P95 为 5.7/14.2 秒。失败案例均有规划输出无效或 JSON 解析失败记录，尚未达到发布验收要求。该集合是公开合成流程开发集，不是保留集，也不是实际用户任务准确率。

`LLM_JSON_MODE_ENABLED=true` 可为主规划、待办草案、日期解析和独立路由启用 `response_format={"type":"json_object"}`；普通回答不启用。默认关闭，以适配不同供应商；使用 Qwen 时需确认模型支持并设置非思考模式。JSON 模式只保证合法 JSON，不保证字段、权限或业务正确性，服务端校验继续执行；不支持的服务返回错误时不会悄悄降级或重试写操作。详见[结构化输出文档](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)。

启用 JSON 约束后，[原失败子集复测](real-flash-json-failure-retest.json)为 **14/15 通过**，观察到 43 次调用、102,512 个供应商报告 token；`account-6` 仍因主规划结构校验失败而未完成余额更新。开启该配置的 100 条离线契约全部通过。本次只复测原失败子集，不能拼接旧基线中的通过案例，声称新版已通过 79/80 或完整回归；输出结构稳定性及完整真实验收仍未完成。

规划失败现提供固定枚举 `validation_reason`，区分 JSON 语法、规划信封、模式、助手消息、动作、参数及策略校验，不保存原始模型输出。加入分类后的[五个账户定向样本](real-flash-account-classified-sample.json)为 5/5 通过，但未复现此前偶发失败，不能据此认定已修复。

主规划在工具执行前遇到 JSON/信封/字段结构或参数校验失败时，默认允许一次格式修正；第二次仍失败则终止。修正依据原始用户请求和工具契约，不把无效响应当作已执行操作，之后仍须通过全量计划校验。未知或内部动作、权限策略拒绝、供应商故障及业务写入不会因此自动重试。修正共用本轮模型调用、token 与时间预算，跟踪记录 `planner_format_repair`；可通过 `PLANNER_FORMAT_REPAIR_ENABLED=false` 关闭，评估报告记录开关。

加入受限格式修正后，相关测试 **144 项通过**，开启 JSON 模式的离线契约 **100/100 通过**。[8 类真实小批回归](real-flash-repair-pilot.json)为 **8/8 通过**，观察到 18 次调用、40,114 个供应商报告 token，用量均有返回；案例总耗时 P50/P95 为 5.3/7.9 秒。该结果仅覆盖首批 8 类合成流程，不能代替新版完整 80 条回归、保留集或重复采样稳定性验收，也不能单独证明格式修正导致成功率提升。

当前完整后端测试为 **703 项通过、1 项跳过**。随后 [Flash 新版完整回归](real-flash-repair-full-quota-limited.json)记录 47/80 通过，但运行期间账号额度耗尽：32 个失败案例含服务调用错误，另 1 个触发单轮预算。该轮未完成健康依赖条件下的验收，不以 47/80 作为模型准确率，也不将错误响应压低后的耗时指标作为性能结论。后续模型切换需建立独立基线。

切换至 `qwen3.8-27b` 并保持非思考、JSON 模式、单次格式修正及原预算后，[首批 8 类真实流程](real-27b-repair-pilot.json)为 **8/8 通过**，案例总耗时 P50/P95 为 5.6/8.9 秒。不同模型不作同条件回归比较；新版完整真实验收仍待完成。

该模型在版本 `44f7454` 的[完整 80 条基线](real-27b-full-baseline.json)为 **77/80（96.25%）**，观察到 200 次调用、347,473 个供应商报告 token；案例总耗时 P50/P95 为 4.7/9.7 秒。三条失败均为修改待办场景的数量断言失败；后续诊断发现首次草案可能已包含额外事项，不能仅归因于修改步骤。该结果仍不是完整通过或保留集质量证明。

后续修复将独立及纯待办批量草案统一到用户原话，保留显式周期、结束日期和时长，避免把路由改写的地点或编号当作额外任务。`revise_proposal` 改为仅服务端已有草案上下文可调用的内部动作。版本 `9bcb43e` 的[原失败三场景复测](real-27b-revision-boundary-retest.json)为 **3/3 通过**，相关测试 **187 项通过**、离线契约 **100/100 通过**；仍不能拼接旧基线宣称新版完整 80/80。来源修复阶段版本 `1637750` 的完整后端测试为 **706 项通过、1 项跳过**。

修复后的 [qwen3.8-27b 完整真实回归](real-27b-grounded-full.json)在相同模型、案例、语料、配置和每轮预算下达到 **80/80 通过**；案例总耗时 P50/P95 为 4.6/9.4 秒。[逐例比较](real-27b-grounded-comparison.json)记录三个改善、零回退、零剩余失败。该结果来自单次公开合成开发集评估，不能证明保留集准确率、长期稳定性或生产 SLO；浏览器验收另行记录。

