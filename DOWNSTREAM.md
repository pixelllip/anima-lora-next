# DOWNSTREAM.md — 本仓库是什么、上游更新怎么跟、哪些文件是我们的

> 本仓库（`anima-lora-next`）是 **`sorryhyun/anima_lora` 的下游整合版**：
> 上游训练/推理/守护进程核心 + 我们的中文显示补丁 + 计划中的 Krea-2 line。
> 上游的 `README.md` / `CLAUDE.md` / `docs/` 原样保留，它们描述的仍然是上游行为。

## 0. 一句话架构

```
upstream/main (sorryhyun/anima_lora)      ← 基线，只读
        │  git merge（自动开 PR，见 §2）
        ▼
main (本仓库) = 上游 + downstream/** + 少数"补丁文件"（见 §3）
        │
        ├── library/krea2/  scripts/krea2/  configs/krea2/   ← 计划中的 Krea-2 line（按上游 line 风格）
        └── gui/i18n/cn_overlay.json + gui/i18n/overlay.py   ← 中文显示补丁（只对 cn 生效）
```

**为什么不做整包 fork-and-forget**：上游迭代很快（我们拉基线的那天 `main` 就走了两个提交），
一次性复制一份代码等于把维护成本全扛下来。这里的做法是「上游为基线 + 极小的自有面」，
让 `git merge upstream/main` 长期保持干净。

## 1. 许可与归属（发布前必读）

| 来源 | 许可 | 本仓库怎么处理 |
| --- | --- | --- |
| `sorryhyun/anima_lora`（上游：训练/推理/GUI/守护进程） | MIT（`LICENSE`，Copyright (c) 2026 Seunghyun Ji）+ 部分派生自 `kohya-ss/sd-scripts` 的代码为 Apache-2.0（`LICENSE-APACHE`） | 两个文件原样保留，**不许删**；`NOTICE` 里补上我们这一节的来源与改动说明 |
| `scvxzf1/krea2-webui`（Krea-2 支持、Web UI、本机配方与探针） | MIT（用户 2026-10-18 确认；其 `LICENSE` 沿用上游同样的 MIT 文本） | 迁移/参考其代码时在 `NOTICE` 里注明来源；**不要**把它的个人数据集/输出/权重带进来 |
| `krea-ai/krea-2`（Krea-2 官方模型仓库） | ⚠ 仓库根目录**没有 LICENSE / LICENSE.txt**（2026-10-18 实测 404），README 只说明**权重**受 Krea 2 Community License 约束 | 移植其代码（`mmdit.py` / `sampling.py` 等）**前必须先确认代码许可**；确认前不要把这类文件发布到 public 仓库 |
| 训练权重 | Anima 基座 = CircleStone Labs NCL（非商用）；Krea 2 = Krea Community License | 本仓库**不带任何权重**，只在 catalog 里给下载地址（`library/downloads.py` 的行） |

## 2. 上游更新怎么跟（自动化）

两条路，都是一件事：把 `upstream/main` 合进 `main`，然后让守卫测试说话。

### 2.1 自动（推荐，默认开启）

`.github/workflows/upstream-sync.yml` 每天 UTC 03:17（北京时间 11:17）跑一次：

1. `git fetch upstream main`，算出落后的提交数（drift）；
2. 起一个 `sync/upstream` 分支把上游 merge 进来：
   * **干净** → 跑守卫测试（`tests/test_downstream_guards.py` + `tests/test_gui_i18n_parity.py`），
     然后开一个 PR（标题带上游提交区间，正文带提交清单 + 冲突文件清单）；
   * **有冲突** → 照样把带冲突标记的分支推上去开 PR，并在正文里列出冲突文件，
     人工只在冲突文件上动手（我们的补丁面很小，通常就是 §3 那 5 个文件）；
3. 想让绿了的同步自动合入：workflow 的 `workflow_dispatch` 传 `auto_merge=true`
   （默认关，先看几轮再打开）。

### 2.2 本地

```powershell
pwsh -File scripts\downstream\upstream-sync.ps1            # 看 drift（只读）
pwsh -File scripts\downstream\upstream-sync.ps1 -Merge     # 合进当前分支并跑守卫
pwsh -File scripts\downstream\upstream-sync.ps1 -Merge -Push   # 合完推回 origin
```

### 2.3 为什么合并能一直是"小冲突"

因为我们的改动**几乎全是新增文件**（`downstream/**`、`gui/i18n/cn_overlay.json`、
`gui/i18n/overlay.py`、`library/krea2/**`、`anima_lora_webui/**` …），只有 **16 个上游文件被改过**
（§3 的"补丁文件"，机器可读版是 `downstream/owned-paths.txt` 的 `patched` 段）。
上游改到这 16 个文件时才会有冲突，`upstream-sync` 会把它们点名。

> 2026-10-30 跟到上游 `v2.3.2`（`upstream/main` 3c579c2）时的实测：合并 147 个上游提交，
> 冲突只有 6 个（`README.md` / `docs/README.md` / `docs/methods/**` 的目录搬迁 /
> `library/lines/adapters.py` / `library/moss/{loader,model}.py`），其余全部自动合入。

### 2.4 守卫测试在守什么

`tests/test_downstream_guards.py`（**只用标准库，CI 里不需要 torch/PySide6**）：

* 补丁文件清单与 `.downstream/patched-upstream-files.txt` 一致（有人偷偷改上游文件就红）；
* 我们依赖的上游缝合点仍然存在（`library/env.py::resolve_under_home`、
  `library/runtime/blockswap.py`、`library/lines/adapters.py`、`anima_daemon/client.py::DaemonClient`、
  `library/downloads.py::DEFAULT_SET`、`gui/i18n/{en,ko,ja,cn}.py::STRINGS` …）；
* `gui/i18n/cn_overlay.json` 可解析、分组/字段键有值，且 **`current_language() != "cn"` 时不生效**。

## 3. 我们的自有面（`downstream/owned-paths.txt` 是机器可读版本）

**新增（永远不会与上游冲突）**

* `DOWNSTREAM.md`、`NOTICE` 里我们那一节
* `gui/i18n/cn_overlay.json`、`gui/i18n/overlay.py`
* `gui/explanations/guides/cn/_fields.json` 的两个补充键（见下）
* `library/krea2/**`、`scripts/krea2/**`、`configs/krea2/**`、`library/models/krea2_raw/**`
  （Krea-2 line，按上游 line 风格实现）
* `anima_daemon/studio.py`、`anima_daemon/easycontrol.py`、`anima_daemon/toolkits.py`
  （训练工作台 / EasyControl / 合并·TensorBoard·实验·队列 的 REST；挂载点在 `server.py`）
* `anima_lora_webui/**`、`tests/test_webui_server.py`（自有 Web 界面，见「界面策略」）
* `library/moss/single_file.py`、`scripts/moss/probe_single_file.py`
  （用 ComfyUI 单文件 int8 convrot / w4a8 权重训练 MOSS，不必下 27 GB bf16）
* `docs/anima/methods/krea2-style-in-the-loop.md`
* `scripts/downstream/**`、`.github/workflows/upstream-sync.yml`、`downstream/**`
* `tests/test_downstream_guards.py`、`tests/test_krea2_line.py`、`tests/test_krea2_stack_lora.py`、
  `tests/test_studio.py`、`tests/test_toolkits.py`

**补丁（改了上游文件，合并时可能在冲突清单里）**

* `gui/tabs/config_tab.py`（字段标签 / 分组标题 / 硬件预设名走 overlay）
* `gui/tabs/easycontrol_tab.py`（同上的描述符表单）
* `gui/tabs/methods_tab.py`、`gui/tabs/preprocess/tab.py`（`Method` 标签走 `t("method")`）
* `gui/i18n/{en,ko,ja,cn}.py`（补 3 个上游漏掉的键：`refresh` / `method` / `path_scope`）
* `gui/explanations/guides/cn/_fields.json`（补 `path_scope` / `preprocess_path_pattern` 两条帮助）
* `README.md`、`NOTICE`、`anima_daemon/README.md`（下游说明与来源声明）
* `anima_daemon/server.py`（挂 studio / easycontrol / toolkits 三组路由）、`anima_daemon/mcp.py`
  （把 `anima_studio_*` 暴露给 AI 工具）
* `scripts/tasks/preprocess.py`（Krea-2 预处理钩子）
* `library/lines/adapters.py`（SVD 初始化走 `dequantized_weight()`，量化底模才有意义）
* `library/moss/loader.py`（`ANIMA_MOSS_{DIT,TE,VAE}_FILE` 单文件分支）、
  `library/moss/model.py`（量化底模的 compute dtype 取 norm，不取 `proj.weight.dtype`）

**界面策略（2026-10-18 决定，2026-10-19 落地）**：自带 PySide6 GUI **继续保留、但不作为主要界面**。
它的功能面很全，但存在 bug；主要界面是仓库内的 [`anima_lora_webui/`](anima_lora_webui/README.md)
—— 零运行时依赖、同源反代守护进程（`python -m anima_lora_webui.server --root <引擎>`）。
守护进程的 REST 是稳定面（`POST /jobs`、`/jobs/{id}/progress`、`/jobs/{id}/{stop,pause,resume}`）；
反代白名单与这些路由的契约由 `tests/test_webui_server.py` 守着 —— 守护进程改端点而界面没跟上，测试就红。
因此 GUI 上的中文补丁只做「标识符 → 中文」这一层，不投入重构。

## 4. 版本与发布

* 我们的 tag 命名：`v<上游版本>+downstream.<n>`（例 `v2.2.1+downstream.1`）；
* `install.ps1` 是上游那个安装器的薄包装（沿用 `ANIMA_DIR` / `ANIMA_VERSION` 两个环境变量），
  指向**本仓库的 release**，这样 ComfyHub 的"自动下载"装到的就是整合版；
* 基线提交记在 `.downstream/upstream-base`，由 `upstream-sync` 在合并成功后更新。
