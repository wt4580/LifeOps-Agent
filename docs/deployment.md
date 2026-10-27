# 本地部署与持续集成

## 镜像与依赖

Dockerfile 使用 Python 3.11 Debian slim，多阶段构建。`production` / `runtime` 提供 BM25、文本/PDF 解析、真实 API 或离线演示；`test` 增加 pytest、Ruff 和测试代码；`ml` 才安装向量检索与语音依赖。默认镜像不下载 HuggingFace 模型，不包含 API Key、个人数据库或本地记忆。OCR 还需自行安装 Tesseract 系统程序。

`requirements-core.txt` 固定主要运行依赖，`requirements-test.txt` 加入测试依赖，`requirements-ml.txt` 为可选重依赖。`requirements.txt` 包含 core + ML；只体验演示时无需安装它。当前 SQLite checkpoint 2.0.11 需要 aiosqlite 0.21.0，0.22.1 删除的线程 API 会导致真实会话失败，因此显式固定兼容版本。

容器使用 UID/GID 10001；业务库、知识库、checkpoint、记忆和审计均写 `/data`。Compose / kind 使用只读根文件系统和独立 `/tmp`。运行时不允许修改源码目录。Dockerfile 的系统证书与时区检查依赖官方 slim 基础镜像。

## Compose

```powershell
# 按本机实际 context 选择；本机验证使用 desktop-linux。
$env:PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple'
docker --context desktop-linux compose up -d --build
docker --context desktop-linux compose ps
docker --context desktop-linux compose exec -T lifeops-demo python scripts/smoke.py --url http://127.0.0.1:5000
```

访问 http://127.0.0.1:5001，演示账号 `demo` / `lifeops-demo-2026`。端口可通过 `LIFEOPS_PORT` 修改。`DEMO_PASSWORD` 只适用于首次创建演示账号；修改它不会自动重置已有 volume 中的密码。

`docker compose stop` 停止服务，`docker compose down` 移除服务但保留演示数据。不要随意使用 `down -v`，它会删除该项目的演示数据卷。演示初始化拒绝包含其他已注册用户的数据库，但仍必须使用专用数据目录。

`/health` 表示进程存活并给出演示/真实模式；`/health/ready` 验证两套数据库与图初始化。前者不表示模型服务可用，后者也不执行付费模型探测。

## 测试与报告

```powershell
docker --context desktop-linux build --target test --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t lifeops-agent:test .
docker --context desktop-linux create --name lifeops-check lifeops-agent:test sh -ec 'python -m ruff check --select E9,F63,F7,F82 app tests scripts; python -m pytest tests -q --junitxml=/reports/unit.xml; python -m app.eval.run_e2e --output /reports/e2e.json'
docker --context desktop-linux start -a lifeops-check
docker --context desktop-linux inspect -f '{{.State.ExitCode}}' lifeops-check
docker --context desktop-linux cp lifeops-check:/reports/. artifacts/
docker --context desktop-linux rm lifeops-check
```

`docker start -a` 本身不能代替容器退出码检查，Jenkinsfile 中也显式检查 `State.ExitCode`。Ruff 只把语法和未定义变量等严重错误作为门禁；未宣称整个遗留仓库通过全部风格规则。

100 条合成案例由 10 类场景各 10 个变体组成，不是 100 个独立真实用户任务。默认规则模拟模型；真实图、SSE、权限、确认与数据库断言仍实际执行。报告包含逐例失败、延迟、模型调用数、用量缺失、语料与源码摘要。延迟为单进程 TestClient 场景总耗时，不是网络性能或真实模型延迟。没有供应商 usage 的调用只记录预算预留量，不能当成真实 token 账单。

如需真实模型回归，在已配置 Key 的本地环境执行 `python -m app.eval.run_e2e --mode real --allow-paid-calls --output artifacts/e2e-real.json`。它会使用独立临时数据库，但产生 API 费用；本次未运行。真实模式排除模拟故障和模拟跨用户 SQL 的 20 条专用用例，保留 80 条流程用例。该数据集仍需人工扩充并划分开发集和保留集。

## kind

复用已有 `dev` 集群，仅操作 `lifeops-demo` 命名空间：

```powershell
$env:DOCKER_CONTEXT='desktop-linux'
kind load docker-image lifeops-agent:demo --name dev
kubectl --context kind-dev apply -f deploy/kind/demo.yaml
kubectl --context kind-dev -n lifeops-demo rollout status deployment/lifeops-demo --timeout=120s
kubectl --context kind-dev -n lifeops-demo exec deployment/lifeops-demo -- python scripts/smoke.py --url http://127.0.0.1:5000
kubectl --context kind-dev -n lifeops-demo port-forward service/lifeops-demo 5002:5000 --address 127.0.0.1
```

最后一个命令保持运行后，可访问 http://127.0.0.1:5002。PVC 使用集群默认 StorageClass，需要支持动态卷。单副本、`Recreate` 更新避免 SQLite 多写者；这是本地演示方案，更新期间会短暂停机。生产部署还需要外部数据库、分布式队列/锁、备份、TLS 与密钥管理。

若复用同一个镜像 tag，重新加载后需执行 `kubectl --context kind-dev -n lifeops-demo rollout restart deployment/lifeops-demo`。Jenkins 使用每次构建唯一 tag，避免新构建继续使用旧 Pod。

## Jenkins

仓库根目录 `Jenkinsfile` 包含：checkout → 构建测试镜像 → 严重静态错误检查 / 单元测试 / 100 条流程契约 → 构建运行镜像 → 真实 HTTP 冒烟 → 可选 kind 部署。始终尝试归档 JUnit XML、JSON / Markdown 报告和演示日志。无需 Docker Pipeline 插件，使用 Docker CLI；需要 Pipeline、Git、JUnit、Credentials 等常规插件。

流水线要求一个标记为 `docker` 的 **Linux 执行节点**，具备 Git、Docker CLI 和对应 daemon 访问权限。测试镜像包含源码和测试，报告通过 `docker cp` 取回，避免 Jenkins 容器路径和 Docker daemon 宿主机路径不一致。启用 `DEPLOY_KIND` 时，该节点还需 kind、kubectl、同一 Docker daemon，以及能访问 `kind-dev` 的 kubeconfig。kubeconfig 属于执行节点配置，不进入镜像或仓库。

在 Jenkins 创建 “Pipeline script from SCM” 任务，选择本项目 Git 仓库与实际分支，Script Path 填 `Jenkinsfile`，首次构建可先关闭 `DEPLOY_KIND`。本地未提交的文件不会自动进入 SCM 构建；应先检查 diff 再自行提交/推送。

本机已有 `my-jenkins` 控制器容器，已启动；检查时登录页仍返回 HTTP 503，尚不能确认 Jenkins 已就绪。它有 Git，缺少 Docker CLI 与 Docker socket，不能直接执行此流水线。推荐单独配置执行节点，避免让控制器承担构建。Jenkins 的 agent secret / 登录凭据需要用户在管理页面配置，本次没有读取或修改认证设置，没有宣称 Jenkins 任务已构建成功。

`.github/workflows/ci.yml` 提供同等 Docker 测试与报告归档，便于公开仓库展示；未推送前不存在远端 CI 成功记录。

本次构建并验证了 runtime / production 和 test 目标，未安装或验证体积较大的 ML 目标。向量检索和语音的依赖、模型路径仍需在实际目标环境单独验证。

## 真实模型部署

另建独立 volume，设置 `DEMO_MODE=false`、`DEMO_SEED=false`、`QWEN_API_KEY`、实际 `DOCS_DIR` 与数据路径，通过 `--env-file` 或 secret 注入，不能修改当前演示 volume。默认账号自动注册仅允许本机回环来源；在容器中首次注册可通过容器内回环请求完成。公开演示凭据不应用于真实数据服务。

工程依据：[Docker 多阶段构建](https://docs.docker.com/build/building/multi-stage/)、[Jenkins Docker 流水线](https://www.jenkins.io/doc/book/pipeline/docker/)、[kind 加载本地镜像](https://kind.sigs.k8s.io/docs/user/quick-start/)。
