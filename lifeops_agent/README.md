# LifeOps-Agent（文档入口）

本目录的详细说明已统一维护在仓库根文档：

- 请优先阅读：`../README.md`
- 项目操作日志：`docs/operation-log.md`

这样做是为了避免出现两份 README 长期不一致的问题。

## 你在本目录通常只需要做三件事

1. 安装依赖
2. 配置 `.env`
3. 启动服务并访问网页

```powershell
cd E:\pycharm\PythonProject\lifeops\lifeops_agent
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn lifeops.main:app --reload
```

打开：`http://127.0.0.1:8000`

## 常用提醒

- 若切换混合检索，请在 `.env` 里设置 `RAG_BACKEND=hybrid`，并重建索引。
- 若要回退稳定模式，设置 `RAG_BACKEND=bm25` 并重启服务。
- 如果 `/api/index` 返回 `vector_error: No module named 'docx2txt'`，请执行 `pip install -r requirements.txt` 后重新运行索引。
