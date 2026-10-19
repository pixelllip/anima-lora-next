# anima_lora_webui

`anima_daemon` 的自有 Web 界面：零运行时依赖（只用 Python 标准库），
常驻内存 20~30MB，启动 <1s。它随 `anima-lora-next` 一起发布，不是独立仓库。

## 为什么不用上游那个 Qt GUI

上游自带 PySide6 GUI 功能很全，但有 bug，且启动后常驻 160MB 上下。
这里只要一个浏览器页面 + 一个极薄的本地服务器：**训练内核照用上游，界面自己写**。
`anima_daemon` 的 REST 已经是稳定面（`POST /jobs`、`GET /jobs/{id}`、
`/jobs/{id}/progress`、`/jobs/{id}/{stop,pause,resume}`、`/queue/{start,pause}`、
`GET /tools` 自描述清单），前端只消费它。

## 跑起来

```powershell
# 在引擎仓库根目录（有 tasks.py 的那一层）执行
<引擎>\.venv\Scripts\python.exe -m anima_lora_webui.server --root <引擎>
# 浏览器打开 http://127.0.0.1:8810
```

守护进程没起也没关系：界面会显示「守护进程未运行」，在引擎目录跑一次
`python tasks.py daemon`（或提交任意作业时它会自己拉起来 —— 上游行为）即可。

参数：`--root`（引擎目录，也可用 `ANIMA_LORA_ROOT`）、`--host`（默认 127.0.0.1）、
`--port`（默认 8810）。`ANIMA_DAEMON_PIDFILE` 可覆盖 pidfile 位置。
`ANIMA_WEBUI_VERBOSE=1` 打开访问日志。

## 界面能干什么

| 面板 | 内容 |
| --- | --- |
| 作业列表 | 全部作业（倒序），状态色标；点一条看详情。3 秒自动刷新（页面隐藏时停） |
| 作业详情 | state / kind / method / preset / 耗时 / 错误；**停止 / 暂停 / 恢复**按钮 |
| 事件流 | `progress.jsonl` 事件表（step / ckpt / log / run_start / run_end）+ **loss 曲线**（canvas） |
| stdout 日志 | 该作业的合并 stdout+stderr（`GET /jobs/{id}/logs` 的 SSE，实时追加） |
| 提交训练 | method / preset / methods_subdir + 常用 overrides（output_name、epochs、steps、lr、dim、alpha、swap）+ extra 参数 → `POST /jobs` |
| 提交命令 | 任意 `argv` 命令作业（例如 `tasks.py preprocess`） |

## 架构（一句话）

```
浏览器 ── 同源 ──> anima_lora_webui(标准库 HTTP server, :8810)
                        ├── /            静态页面（static/）
                        └── /api/*       白名单反代 ──> anima_daemon(127.0.0.1:8765)
```

**为什么要反代**：`anima_daemon` 不带 CORS 头，浏览器不能跨源直连它；
反代同时把可访问路径**白名单化**（不是开放代理）—— GET 只放
`/health` `/tools` `/jobs` `/events`，POST 只放 `/jobs` `/queue/*` `/shutdown`。

白名单写在 `server.py` 的 `GET_ALLOW` / `POST_ALLOW`；契约测试
`tests/test_webui_server.py::test_proxy_whitelist_matches_daemon_routes`
盯着"白名单里的每条前缀，在 `anima_daemon/server.py` 的对应方法里都有路由"——
守护进程改了端点而这里没跟上，测试就红。

`/api/_env` 是本服务自己的端点：报引擎目录、守护进程端口与 URL、本进程的解释器。

## 测试

```powershell
python -m pytest tests/test_webui_server.py -q
```

用例会起一个**假守护进程**（stdlib）+ 真起 WebUI，覆盖：静态文件、代理白名单
（含"不该放行的路径必须 404"）、`POST /jobs` 带 body 转发、pidfile 发现与覆盖、
反代白名单与 `anima_daemon` 路由的契约。只用标准库，不需要 torch / PySide6。

## 许可

本目录的代码是我们的自有面，随 `anima-lora-next` 一起按 MIT 发布；
它只发 HTTP 请求，不复制上游代码。
