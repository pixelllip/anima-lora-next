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
`--port`（默认 8810）、`--token`（可选访问令牌）。`ANIMA_DAEMON_PIDFILE` 可覆盖
pidfile 位置。`ANIMA_WEBUI_VERBOSE=1` 打开访问日志。

**给手机用**：ComfyHub 起 WebUI 时会 `--host 0.0.0.0 --token <随机>`。首次打开
`http://<桌面 IP>:<端口>/?token=…`，服务会 302 到 `/` 并种一个 HttpOnly Cookie；
之后静态资源 / `/api/*` / SSE 全部自动带上。没有 token 的请求一律 401 ——
训练台会写数据集、起几小时的 GPU 作业，不能是“局域网谁都能点”。
手动 `--host 0.0.0.0` 而不给 `--token` 时行为与旧版相同（不鉴权），只建议在可信网络里用。

## 界面能干什么

| 面板 | 内容 |
| --- | --- |
| 训练项目 | 左侧项目列表 + 新建（名称 / 触发词 / 方法 / 硬件档）；五个阶段圆点一眼看出进度 |
| 数据集 | **取训练集**（本机文件夹 / URL → 待筛选）、缩略图网格、勾选批量**接受 / 拒绝 / 移回 / 删除**、点图编辑 caption；窄屏 2 列、宽屏自适应 |
| 打标（可选） | `caption-autotag` 的 missing / merge / overwrite 三档；caption 覆盖率；一键补触发词 |
| 预处理 | dataset / resized / latent / TE 计数；"预处理完自动开训"开关；生成的 scoped 配置预览 |
| **参数** | **复现 Qt GUI 的 Config 页**：方法 / 硬件档 + 基础字段 + 可折叠的高级分组；字段、分组、basic/advanced 划分与中文帮助全部来自 `gui.core.config_io` + `_fields.json`，不是前端手抄的第二份。只写用户改过的 overrides，`null` = 恢复预设；保存后重新生成 `train_config.toml` 并显示快照。支持 `?tab=config` 深链 |
| 训练 | 使用「参数」页保存的配置提交；产物列表（`output/ckpt/<项目>`） |
| **Krea-2** | 独立训练线的 `TrainRequest` 全表（身份 / 数据与叠加 / 训练预算 / 显存与运行 / 保存与日志），含 style-in-the-loop 的 `stack_lora_*`；保存为 `training_projects/<项目>/krea2.toml`，校验 cache/model 后以 `scripts/krea2/train.py --config` 提交到同一串行队列；产物列表（`output/ckpt_krea2/`）。支持 `?tab=krea2` 深链 |
| **合并** | 复现 Qt GUI 的 Merge 页：适配器目录 / 文件列表 / safetensors 烘焙体检；三种模式（烘焙进 DiT / 融合多个 LoRA / 从两个检查点提取 ΔW LoRA）+ 只读干扰分析；全部走同一串行队列，日志用 job SSE |
| **EasyControl** | 方法变体（`gui-methods/*.toml`）+ 描述符变体（`configs/easycontrol/*.toml`，colorize / inpaint / sanitize / region）；表单只写改动字段，tomlkit 保留注释与 `[[datasets]]` 蓝图；预处理（`EASYADAPTER=…`）与训练都提交到同一串行队列 |
| **TensorBoard** | 守护进程起停 TensorBoard（6006 起自动避让），列出 `output/logs` 里真有 event 文件的 run、可单 run 查看 / 删除；内嵌 iframe |
| **Anime Tools** | 引擎自带策展面板（`anime_tools.gui`）的起停与内嵌；面板只绑回环，手机上不显示 iframe |
| **实验** | Turbo 蒸馏（`configs/methods/turbo.toml` → `tasks.py turbo`）与 ΔW-soup（`configs/soup/soup.toml` → `python -m scripts.soup.pipeline`）；sectioned TOML 表单保留注释 |
| 作业 | 全局队列（含**暂停 / 恢复队列**）+ 详情 + 事件表 + **loss 曲线** + stdout SSE；停止 / 暂停 / 恢复单个作业 |

与 Qt GUI 的差异只剩 **MOSS 页**（`gui/lines` + `library/moss`）与 App 内安装；
其余页签都已对齐。TensorBoard 绑 `0.0.0.0`（手机可看）且本身无令牌，只读日志；
Anime Tools 面板只绑回环，手机端只有状态、不嵌 iframe。

手机与电脑是同一份状态：手机连桌面的 WebUI 时，筛选结果、caption、参数、作业进度立刻同步。
AI 工作台调的是同一批 `/studio/*`（MCP 工具 `anima_studio_*`），人机共用一套事实。

参数相关端点（都挂在守护进程上，WebUI 只反代）：

```
GET  /studio/schema                        方法 / 硬件档 / 分组
GET  /studio/projects/{slug}/config        合并后的字段 + 覆盖值 + 生成快照
POST /studio/projects/{slug}/config        {variant?, preset?, output_name?, overrides?}
GET  /studio/projects/{slug}/krea2         TrainRequest 表 + cache/model 检查 + 产物
POST /studio/projects/{slug}/krea2         {values}
POST /studio/projects/{slug}/krea2/train   校验后提交 scripts/krea2/train.py
```

`overrides` 的值按**合并链里的原值类型**转换（bool / int / float / list / dict / string），
类型不对直接 400，**不许静默忽略**；`use_valid` / `validation_split_num` /
`repeat_by_folder_name` 是虚拟键，保存时写进第一条 `[[datasets]]`（复用
`gui.core.validation`，与 Qt GUI 同一份语义）。

## 架构（一句话）

```
浏览器 ── 同源 ──> anima_lora_webui(标准库 HTTP server, :8810)
                        ├── /            静态页面（static/）
                        └── /api/*       白名单反代 ──> anima_daemon(127.0.0.1:8765)
```

**为什么要反代**：`anima_daemon` 不带 CORS 头，浏览器不能跨源直连它；
反代同时把可访问路径**白名单化**（不是开放代理）—— GET 只放
`/health` `/tools` `/jobs` `/events` `/studio`，POST 只放 `/jobs` `/queue/*` `/shutdown` `/studio`。

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
