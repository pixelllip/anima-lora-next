// Anima 训练台 —— 无框架、无构建步骤。
// 所有请求走同源 /api/*（anima_lora_webui.server 反代到 anima_daemon）。
//
// 页面只消费 daemon 的 /studio/* 与 /jobs/*：项目、图片、caption、阶段状态全在
// 后端，前端不缓存"哪一步做完了"这种事实（否则 AI 与网页会看到两份真相）。

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const STATE_CLASS = {
  running: "ok", done: "done", queued: "wait", paused: "wait",
  error: "bad", stopped: "bad",
};
const STAGE_ORDER = [
  ["fetch", "取训练集"],
  ["review", "人工筛选"],
  ["tag", "打标（可选）"],
  ["preprocess", "预处理"],
  ["train", "正式训练"],
];

let env = {};
let projects = [];
let current = null;          // 项目详情（daemon 的 /studio/projects/{slug}）
let images = [];
let imageState = "inbox";
let picked = new Set();
let jobs = [];
let selectedJob = null;
let logSource = null;
let progressTimer = null;
let activeTab = "dataset";
let captionTarget = null;
let pollTimer = null;

// 参数页（Anima 合并链）与 Krea-2 页各自一份草稿：服务端返回的 `value` 是基线，
// 只有和基线不同的字段才会写进 overrides（Krea-2 则是整表落盘）。
let config = null;
let configDraft = {};
let configOverrides = {};
let configDirty = false;
let krea2 = null;
let krea2Draft = {};
let krea2Dirty = false;
let studioSchema = null;

// --- 基础 -------------------------------------------------------------------
async function api(path, opts) {
  const res = await fetch("/api" + path, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }
  if (!res.ok) {
    const msg = (data && (data.error || data.message)) || `HTTP ${res.status}`;
    throw Object.assign(new Error(msg), { data, status: res.status });
  }
  return data;
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function toast(message, bad = false) {
  const el = $("#toast");
  el.textContent = message;
  el.className = "toast" + (bad ? " bad" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add("hidden"), bad ? 7000 : 3500);
}

function fmtDuration(sec) {
  if (sec == null) return "-";
  const s = Math.max(0, Math.round(sec));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

function fmtTime(ts) {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  if (Number.isNaN(d.getTime())) return "-";
  return `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function fmtSize(bytes) {
  if (!bytes) return "0 B";
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function setNotice(id, message, kind = "") {
  const el = $(id);
  el.textContent = message || "";
  el.className = "notice" + (kind ? " " + kind : "");
}

// --- 环境 -------------------------------------------------------------------
async function refreshEnv() {
  try {
    env = await api("/_env");
    env = env || {};
    const d = env.daemon || {};
    const pill = $("#env");
    if (d.port) {
      pill.textContent = `守护进程 :${d.port} · ${env.engineRoot || "-"}`;
      pill.className = "pill ok";
    } else {
      pill.textContent = "守护进程未运行（点任意作业会尝试自动拉起）";
      pill.className = "pill bad";
    }
  } catch (e) {
    const pill = $("#env");
    pill.textContent = "WebUI 后端异常：" + e.message;
    pill.className = "pill bad";
  }
}

// --- 项目 -------------------------------------------------------------------
async function loadProjects() {
  let list = [];
  try { list = await api("/studio/projects"); } catch (e) { list = []; }
  projects = Array.isArray(list) ? list : [];
  const box = $("#projects");
  box.innerHTML = "";
  if (!projects.length) {
    box.innerHTML = '<p class="dim small">还没有项目。点「＋ 新建」开始。</p>';
    return;
  }
  for (const p of projects) {
    const el = document.createElement("div");
    el.className = "project" + (current && current.slug === p.slug ? " sel" : "");
    const stage = p.stages || {};
    const dots = STAGE_ORDER.map(([key]) =>
      `<span class="stage-dot ${stage[key] ? "on" : ""}" title="${esc(key)}"></span>`).join("");
    el.innerHTML = `
      <b>${esc(p.name)}</b>
      <div class="sub">
        <span>${esc(p.variant)}/${esc(p.preset)}</span>
        <span>${p.counts?.dataset ?? 0} 图</span>
        <span style="margin-left:auto">${dots}</span>
      </div>`;
    el.onclick = () => selectProject(p.slug);
    box.appendChild(el);
  }
}

async function selectProject(slug) {
  try {
    current = await api(`/studio/projects/${encodeURIComponent(slug)}`);
  } catch (e) {
    toast("打开项目失败：" + e.message, true);
    return;
  }
  picked.clear();
  images = [];
  selectedJob = null;
  config = null;
  configDraft = {};
  configOverrides = {};
  configDirty = false;
  krea2 = null;
  krea2Draft = {};
  krea2Dirty = false;
  closeLog();
  $("#empty").classList.add("hidden");
  $("#project-view").classList.remove("hidden");
  renderProject();
  await loadImages();
  await loadOutputs();
  await loadJobs();
  renderTag();
  renderPreprocess();
  await loadProjectConfig();
  await loadKrea2();
  await loadProjects();
}

function renderProject() {
  if (!current) return;
  const c = current.counts || {};
  $("#p-name").textContent = current.name || current.slug;
  $("#p-meta").textContent =
    `${current.variant}/${current.preset} · 触发词 ${current.trigger || "（未设置）"} · ` +
    `output_name ${current.outputName || current.slug}`;
  $("#p-counts").innerHTML = [
    ["待筛选", c.inbox], ["已接受", c.dataset], ["已拒绝", c.rejected],
    ["caption", c.captions], ["resized", c.resized], ["latent", c.latents],
    ["产物", c.outputs],
  ].map(([k, v]) => `<span>${esc(k)} <b>${v ?? 0}</b></span>`).join("");

  const stages = current.stages || {};
  const firstOpen = STAGE_ORDER.find(([key]) => !stages[key]);
  $("#stage-bar").innerHTML = STAGE_ORDER.map(([key, label]) => {
    const done = !!stages[key];
    const now = !done && firstOpen && firstOpen[0] === key;
    return `<div class="stage ${done ? "done" : ""} ${now ? "now" : ""}">
      <b>${esc(label)}</b>${done ? "已完成" : now ? "下一步" : "未开始"}
    </div>`;
  }).join("");

  $("#n-inbox").textContent = c.inbox ?? 0;
  $("#n-dataset").textContent = c.dataset ?? 0;
  $("#n-rejected").textContent = c.rejected ?? 0;

}

// --- 图片 -------------------------------------------------------------------
async function loadImages() {
  if (!current) return;
  let payload;
  try {
    payload = await api(
      `/studio/projects/${encodeURIComponent(current.slug)}/images?state=${encodeURIComponent(imageState)}&limit=1000`
    );
  } catch (e) {
    toast("读取图片失败：" + e.message, true);
    return;
  }
  images = payload.images || [];
  picked.clear();
  $("#select-all").checked = false;
  renderImages();
}

function imageStateLabel(state) {
  return { inbox: "待筛选", dataset: "已接受", rejected: "已拒绝" }[state] || state;
}

function renderImages() {
  const grid = $("#image-grid");
  grid.innerHTML = "";
  $("#image-empty").classList.toggle("hidden", images.length > 0);
  for (const img of images) {
    const card = document.createElement("div");
    card.className = "img-card" + (picked.has(img.path) ? " picked" : "");
    card.dataset.path = img.path;
    card.innerHTML = `
      <input type="checkbox" class="pick-box" ${picked.has(img.path) ? "checked" : ""}
        aria-label="选择 ${esc(img.name)}">
      <span class="state-tag">${esc(imageStateLabel(img.state))}</span>
      <img loading="lazy" src="/api${esc(img.url)}" alt="${esc(img.name)}">
      <div class="img-body">
        <div class="name" title="${esc(img.name)}">${esc(img.name)}</div>
        <div class="cap" title="${esc(img.caption)}">${esc(img.caption || "（无 caption）")}</div>
        <div class="img-actions">
          ${img.state === "inbox" ? '<button data-act="accept" class="mini primary">接受</button>' : ""}
          ${img.state !== "rejected" ? '<button data-act="reject" class="mini warn">拒绝</button>' : ""}
          ${img.state !== "inbox" ? '<button data-act="restore" class="mini ghost">待筛选</button>' : ""}
          <button data-act="delete" class="mini danger">删</button>
        </div>
      </div>`;
    grid.appendChild(card);
  }
}

function togglePick(path, on) {
  if (on) picked.add(path); else picked.delete(path);
  const card = Array.from(document.querySelectorAll(".img-card"))
    .find((el) => el.dataset.path === path);
  if (card) card.classList.toggle("picked", on);
}

async function review(action, files) {
  if (!current) return;
  const list = files || Array.from(picked);
  if (!list.length) { toast("先勾选图片（或直接点卡片上的按钮）", true); return; }
  try {
    const result = await api(
      `/studio/projects/${encodeURIComponent(current.slug)}/review`,
      { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, files: list }) }
    );
    const n = result.moved?.length || 0;
    toast(`${action}：${n} 张`, (result.errors || []).length > 0);
    if (result.errors?.length) console.warn("review errors", result.errors);
    await refreshCurrent();
  } catch (e) {
    toast("操作失败：" + e.message, true);
  }
}

async function refreshCurrent() {
  if (!current) return;
  try {
    current = await api(`/studio/projects/${encodeURIComponent(current.slug)}`);
    renderProject();
    renderTag();
    renderPreprocess();
    await loadImages();
    await loadOutputs();
    if (!configDirty) await loadProjectConfig();
    if (!krea2Dirty) await loadKrea2();
    await loadProjects();
  } catch (e) {
    toast("刷新项目失败：" + e.message, true);
  }
}

// --- caption ----------------------------------------------------------------
function openCaption(img) {
  if (!current) return;
  captionTarget = img;
  $("#caption-title").textContent = `${img.name}（${imageStateLabel(img.state)}）`;
  $("#caption-textarea").value = img.caption || "";
  $("#caption-dialog").showModal();
}

async function saveCaption() {
  if (!current || !captionTarget) return;
  try {
    await api(`/studio/projects/${encodeURIComponent(current.slug)}/caption`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: captionTarget.path, caption: $("#caption-textarea").value }),
    });
    $("#caption-dialog").close();
    toast("caption 已保存（训练前记得重跑预处理）");
    await refreshCurrent();
  } catch (e) {
    toast("保存失败：" + e.message, true);
  }
}

// --- 取训练集 ---------------------------------------------------------------
async function importImages() {
  if (!current) return;
  const paths = $("#import-paths").value.split("\n").map((s) => s.trim()).filter(Boolean);
  const urls = $("#import-urls").value.split("\n").map((s) => s.trim()).filter(Boolean);
  const caption = $("#import-caption").value.trim();
  const recursive = $("#import-recursive").checked;
  if (!paths.length && !urls.length) { toast("填一个文件夹路径或图片 URL", true); return; }
  setNotice("#import-msg", "导入中…");
  try {
    const r = await api(`/studio/projects/${encodeURIComponent(current.slug)}/import`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paths, urls, caption, recursive }),
    });
    setNotice(
      "#import-msg",
      `新增 ${r.added?.length || 0}，跳过 ${r.skipped?.length || 0}，失败 ${r.errors?.length || 0}`,
      (r.errors?.length || 0) ? "bad" : "ok"
    );
    if (r.errors?.length) console.warn("import errors", r.errors);
    await refreshCurrent();
  } catch (e) {
    setNotice("#import-msg", "导入失败：" + e.message, "bad");
  }
}

// --- 打标 -------------------------------------------------------------------
function renderTag() {
  if (!current) return;
  const c = current.counts || {};
  const pct = c.dataset ? Math.min(100, Math.round((c.captions / c.dataset) * 100)) : 0;
  $("#caption-bar").style.width = `${pct}%`;
  $("#caption-text").textContent =
    `已接受 ${c.dataset || 0} 张，其中 ${c.captions || 0} 张有 caption（${pct}%）。` +
    (c.dataset && c.captions < c.dataset
      ? " 可以跑一次自动打标，或者逐张手写（点数据集里的缩略图）。"
      : "");
  const job = current.latestJobs?.tag;
  if (job) {
    setNotice(
      "#tag-status",
      `最近打标作业 ${job.id} · ${job.state || "?"}${job.error ? " · " + job.error : ""}`,
      job.state === "error" ? "bad" : job.state === "done" ? "ok" : ""
    );
  }
}

async function runTag() {
  if (!current) return;
  try {
    const r = await api(`/studio/projects/${encodeURIComponent(current.slug)}/tag`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: $("#tag-mode").value }),
    });
    toast(`打标作业已入队：${r.job_id}`);
    await refreshCurrent();
  } catch (e) {
    toast("打标失败：" + e.message, true);
  }
}

async function applyTrigger() {
  if (!current) return;
  try {
    const r = await api(`/studio/projects/${encodeURIComponent(current.slug)}/trigger`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
    });
    toast(`已给 ${r.changed} 张图加触发词「${r.trigger}」`);
    await refreshCurrent();
  } catch (e) {
    toast("加触发词失败：" + e.message, true);
  }
}

// --- 预处理 -----------------------------------------------------------------
function renderPreprocess() {
  if (!current) return;
  const c = current.counts || {};
  $("#prep-counts").innerHTML = [
    ["已接受", c.dataset], ["已 resize", c.resized], ["latent", c.latents], ["TE 缓存", c.teCache],
  ].map(([k, v]) => `<span>${esc(k)} <b>${v ?? 0}</b></span>`).join("");
  $("#config-preview").textContent = current.configText || "（还没有生成配置；提交预处理/训练时会生成）";
  const job = current.latestJobs?.preprocess;
  if (job) {
    setNotice(
      "#prep-status",
      `最近预处理作业 ${job.id} · ${job.state || "?"}${job.error ? " · " + job.error : ""}`,
      job.state === "error" ? "bad" : job.state === "done" ? "ok" : ""
    );
  }
}

async function runPreprocess() {
  if (!current) return;
  try {
    const r = await api(`/studio/projects/${encodeURIComponent(current.slug)}/preprocess`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chain_train: $("#chain-train").checked }),
    });
    toast(`预处理作业已入队：${r.job_id}${r.chain_train ? "（完成后自动训练）" : ""}`);
    await refreshCurrent();
  } catch (e) {
    toast("预处理失败：" + e.message, true);
  }
}

// --- 参数（Anima 合并链；复现 Qt GUI 的 Config 页） -------------------------
const enc = (value) => encodeURIComponent(value);

function eqValue(a, b) {
  if (a === b) return true;
  try { return JSON.stringify(a) === JSON.stringify(b); } catch { return false; }
}

function fillSelect(el, options, selected) {
  const list = (options && options.length) ? options.slice() : [];
  if (selected != null && !list.some((o) => o.value === selected)) {
    list.push({ value: selected, label: selected });
  }
  el.innerHTML = list.map((o) =>
    `<option value="${esc(o.value)}" ${o.value === selected ? "selected" : ""}>${esc(o.label || o.value)}</option>`
  ).join("");
}

function helpIcon(field) {
  return field.help ? `<span class="help" title="${esc(field.help)}">?</span>` : "";
}

function fieldControl(field, value, prefix) {
  const data = `data-key="${esc(field.key)}" id="${prefix}-${esc(field.key)}"`;
  if (field.type === "bool") {
    return `<label class="field field-check">
      <input type="checkbox" ${data} ${value ? "checked" : ""}>
      <span>${esc(field.label)}${helpIcon(field)}${field.overridden ? '<em class="ovr">已覆盖</em>' : ""}</span>
    </label>`;
  }
  let control;
  if (field.type === "list" || field.type === "dict") {
    const text = value == null ? "" : JSON.stringify(value);
    control = `<textarea rows="2" ${data} spellcheck="false">${esc(text)}</textarea>`;
  } else if (field.choices && field.choices.length) {
    const current = String(value ?? "");
    const values = [...new Set([current, ...field.choices].filter((v) => v !== ""))];
    control = `<select ${data}>${values.map((v) =>
      `<option value="${esc(v)}" ${v === current ? "selected" : ""}>${esc(v)}</option>`
    ).join("")}</select>`;
  } else {
    const type = field.type === "int" || field.type === "float" ? "number" : "text";
    const step = field.type === "float" ? 'step="any"' : field.type === "int" ? 'step="1"' : "";
    control = `<input type="${type}" ${step} ${data} value="${esc(value ?? "")}" spellcheck="false">`;
  }
  return `<label class="field" title="${esc(field.key)}">
    <span class="field-label">${esc(field.label)}${helpIcon(field)}${field.overridden ? '<em class="ovr">已覆盖</em>' : ""}</span>
    ${control}
  </label>`;
}

async function loadProjectConfig(params) {
  if (!current) return;
  const query = params ? "?" + new URLSearchParams(params).toString() : "";
  try {
    config = await api(`/studio/projects/${enc(current.slug)}/config${query}`);
    configDraft = {};
    for (const field of config.fields) configDraft[field.key] = field.value;
    configOverrides = { ...(config.overrides || {}) };
    configDirty = false;
    renderConfigForm();
    if (config.warning) setNotice("#cfg-status", "读取参数：" + config.warning, "bad");
  } catch (e) {
    setNotice("#cfg-status", "读取参数失败：" + e.message, "bad");
  }
}

function renderConfigForm() {
  if (!config) return;
  const schema = config.schema || {};
  fillSelect($("#cfg-variant"), schema.variants || [], config.variant);
  fillSelect($("#cfg-preset"), schema.presets || [], config.preset);
  $("#cfg-output").value = config.outputName || "";
  const basic = config.fields.filter((f) => f.basic);
  const advanced = config.fields.filter((f) => !f.basic);
  $("#cfg-basic").innerHTML = basic.length
    ? basic.map((f) => fieldControl(f, configDraft[f.key], "cfg")).join("")
    : '<p class="dim small">这个方法没有基础参数。</p>';
  const groups = {};
  for (const field of advanced) {
    if (!groups[field.group]) groups[field.group] = { label: field.groupLabel, fields: [] };
    groups[field.group].fields.push(field);
  }
  const order = (schema.groups || []).map((g) => g.name);
  const names = Object.keys(groups).sort((a, b) => {
    const ia = order.indexOf(a), ib = order.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  $("#cfg-advanced").innerHTML = names.map((name) => {
    const group = groups[name];
    return `<details class="field-group">
      <summary>${esc(group.label)} <span class="dim small">${group.fields.length} 项</span></summary>
      <div class="field-grid">${group.fields.map((f) => fieldControl(f, configDraft[f.key], "cfg")).join("")}</div>
    </details>`;
  }).join("") || '<p class="dim small">没有高级参数。</p>';
  $("#cfg-preview").textContent = config.configText || "（保存后生成 train_config.toml）";
  updateConfigDirty();
  applyConfigFilter();
}

function updateConfigDirty() {
  configDirty = false;
  if (config) {
    for (const field of config.fields) {
      if (!eqValue(configDraft[field.key], field.value)) { configDirty = true; break; }
    }
    if (!configDirty && $("#cfg-variant").value !== config.variant) configDirty = true;
    if (!configDirty && $("#cfg-preset").value !== config.preset) configDirty = true;
    if (!configDirty && $("#cfg-output").value.trim() !== (config.outputName || "")) configDirty = true;
  }
  $("#cfg-dirty").classList.toggle("hidden", !configDirty);
}

function onConfigFieldInput(ev) {
  const el = ev.target.closest("[data-key]");
  if (!el || !config) return;
  const key = el.dataset.key;
  const field = config.fields.find((f) => f.key === key);
  if (!field) return;
  let value;
  if (field.type === "bool") {
    value = el.checked;
  } else if (field.type === "int" || field.type === "float") {
    if (el.value === "") value = null;
    else {
      value = Number(el.value);
      if (Number.isNaN(value)) return;
    }
  } else if (field.type === "list" || field.type === "dict") {
    if (!el.value.trim()) value = null;
    else {
      try {
        value = JSON.parse(el.value);
        el.classList.remove("bad-input");
      } catch {
        el.classList.add("bad-input");
        return;
      }
    }
  } else {
    value = el.value;
  }
  configDraft[key] = value;
  if (key === "output_name") $("#cfg-output").value = value ?? "";
  updateConfigDirty();
}

function collectConfigOverrides() {
  const out = {};
  if (!config) return out;
  for (const field of config.fields) {
    const draft = configDraft[field.key];
    if (!eqValue(draft, field.value)) out[field.key] = draft;
    else if (Object.prototype.hasOwnProperty.call(configOverrides, field.key)) out[field.key] = null;
  }
  return out;
}

function applyConfigFilter() {
  const query = ($("#cfg-filter").value || "").trim().toLowerCase();
  for (const el of $$("#tab-config .field")) {
    const input = el.querySelector("[data-key]");
    const key = input ? input.dataset.key.toLowerCase() : "";
    const text = el.textContent.toLowerCase();
    el.classList.toggle("hidden", !!query && !key.includes(query) && !text.includes(query));
  }
  for (const group of $$("#cfg-advanced .field-group")) {
    const any = Array.from(group.querySelectorAll(".field")).some((el) => !el.classList.contains("hidden"));
    group.classList.toggle("hidden", !any);
  }
}

async function saveConfig(thenTrain = false) {
  if (!current || !config) return false;
  const btn = $("#btn-cfg-save");
  btn.disabled = true;
  try {
    const result = await api(`/studio/projects/${enc(current.slug)}/config`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        variant: $("#cfg-variant").value,
        preset: $("#cfg-preset").value,
        output_name: $("#cfg-output").value.trim(),
        overrides: collectConfigOverrides(),
      }),
    });
    config = result;
    configDraft = {};
    for (const field of result.fields) configDraft[field.key] = field.value;
    configOverrides = { ...(result.overrides || {}) };
    configDirty = false;
    renderConfigForm();
    if (result.warning) {
      setNotice("#cfg-status", "参数已保存；但生成配置失败：" + result.warning, "bad");
    } else {
      setNotice("#cfg-status", "参数已保存，train_config.toml 已重新生成。", "ok");
    }
    await refreshCurrent();
    if (thenTrain) return await startTrain();
    return true;
  } catch (e) {
    setNotice("#cfg-status", "保存失败：" + e.message, "bad");
    return false;
  } finally {
    btn.disabled = false;
  }
}

async function resetConfigAll() {
  if (!config) return;
  const keys = Object.keys(configOverrides);
  if (!keys.length) {
    setNotice("#cfg-status", "当前没有覆盖项，全部字段都跟随预设。");
    return;
  }
  if (!confirm(`删除 ${keys.length} 个覆盖项，全部恢复预设默认？`)) return;
  const overrides = {};
  for (const key of keys) overrides[key] = null;
  try {
    const result = await api(`/studio/projects/${enc(current.slug)}/config`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        variant: config.variant,
        preset: config.preset,
        output_name: config.outputName,
        overrides,
      }),
    });
    config = result;
    configDraft = {};
    for (const field of result.fields) configDraft[field.key] = field.value;
    configOverrides = {};
    configDirty = false;
    renderConfigForm();
    setNotice("#cfg-status", "已恢复预设默认。", "ok");
    await refreshCurrent();
  } catch (e) {
    setNotice("#cfg-status", "恢复失败：" + e.message, "bad");
  }
}

// --- 训练 -------------------------------------------------------------------
async function startTrain() {
  if (!current) return false;
  if (configDirty && !(await saveConfig(false))) return false;
  try {
    const result = await api(`/studio/projects/${enc(current.slug)}/train`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    setNotice("#train-status", `训练作业已入队：${result.job_id}（${result.state}）`, "ok");
    toast(`训练作业已入队：${result.job_id}`);
    switchTab("jobs");
    await loadJobs();
    await refreshCurrent();
    return true;
  } catch (e) {
    setNotice("#train-status", "训练未开始：" + e.message, "bad");
    return false;
  }
}

function renderOutputs() {
  const files = current?.outputFiles || [];
  $("#output-list").innerHTML = files.length
    ? files.map((f) => `<div class="job"><div class="r1"><b>${esc(f.name)}</b><span class="dim">${fmtSize(f.size)}</span></div>
        <div class="r2"><span>${fmtTime(f.mtime)}</span></div></div>`).join("")
    : '<p class="dim small">还没有训练产物。</p>';
}

async function loadOutputs() { renderOutputs(); }

// --- Krea-2（独立训练线） ----------------------------------------------------
async function loadKrea2() {
  if (!current) return;
  try {
    krea2 = await api(`/studio/projects/${enc(current.slug)}/krea2`);
    krea2Draft = { ...(krea2.values || {}) };
    krea2Dirty = false;
    renderKrea2Form();
    if (krea2.warning) setNotice("#k2-notice", "读取 Krea-2 参数：" + krea2.warning, "bad");
  } catch (e) {
    setNotice("#k2-notice", "读取 Krea-2 参数失败：" + e.message, "bad");
  }
}

function renderKrea2Form() {
  if (!krea2) return;
  const checks = krea2.checks || {};
  $("#k2-checks").innerHTML = [
    ["缓存", checks.cacheExists ? `已找到 · ${checks.cacheFiles} 个 .npz` : "缺失",
      !!(checks.cacheExists && checks.cacheFiles), checks.cache],
    ["底模", checks.modelExists ? "已找到" : "缺失", !!checks.modelExists, checks.model],
    ["产物", checks.output || "-", true, checks.output],
  ].map(([label, value, ok, title]) =>
    `<span class="${ok ? "ok" : "bad"}" title="${esc(title || "")}">${esc(label)} <b>${esc(value)}</b></span>`
  ).join("");

  const groups = {};
  for (const field of krea2.fields) {
    if (!groups[field.group]) groups[field.group] = { label: field.groupLabel, fields: [] };
    groups[field.group].fields.push(field);
  }
  const order = (krea2.groups || []).map((g) => g.name);
  const names = Object.keys(groups).sort((a, b) => {
    const ia = order.indexOf(a), ib = order.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  $("#k2-fields").innerHTML = names.map((name) => {
    const group = groups[name];
    return `<details class="field-group" open>
      <summary>${esc(group.label)} <span class="dim small">${group.fields.length} 项</span></summary>
      <div class="field-grid">${group.fields.map((f) => fieldControl(f, krea2Draft[f.key], "k2")).join("")}</div>
    </details>`;
  }).join("");
  $("#k2-preview").textContent = krea2.text || "";
  const outputs = krea2.outputs || [];
  $("#k2-outputs").innerHTML = outputs.length
    ? outputs.map((f) => `<div class="job"><div class="r1"><b>${esc(f.name)}</b>
        <span class="dim">${fmtSize(f.size)}</span></div>
        <div class="r2"><span>${fmtTime(f.mtime)}</span></div></div>`).join("")
    : '<p class="dim small">还没有 Krea-2 产物。</p>';
  const latest = (current?.jobs || []).filter((j) => j.stage === "krea2").slice(-1)[0];
  if (latest) {
    setNotice(
      "#k2-notice",
      `最近 Krea-2 作业 ${latest.id} · ${latest.state || "?"}${latest.error ? " · " + latest.error : ""}`,
      latest.state === "error" ? "bad" : latest.state === "done" ? "ok" : ""
    );
  }
  updateKrea2Dirty();
}

function updateKrea2Dirty() {
  krea2Dirty = false;
  if (krea2) {
    for (const field of krea2.fields) {
      if (!eqValue(krea2Draft[field.key], field.value)) { krea2Dirty = true; break; }
    }
  }
  const badge = $("#k2-dirty");
  if (badge) badge.classList.toggle("hidden", !krea2Dirty);
}

function onKrea2FieldInput(ev) {
  const el = ev.target.closest("[data-key]");
  if (!el || !krea2) return;
  const key = el.dataset.key;
  const field = krea2.fields.find((f) => f.key === key);
  if (!field) return;
  let value;
  if (field.type === "bool") {
    value = el.checked;
  } else if (field.type === "int" || field.type === "float") {
    if (el.value === "") value = null;
    else {
      value = Number(el.value);
      if (Number.isNaN(value)) return;
    }
  } else if (field.type === "list" || field.type === "dict") {
    if (!el.value.trim()) value = null;
    else {
      try {
        value = JSON.parse(el.value);
        el.classList.remove("bad-input");
      } catch {
        el.classList.add("bad-input");
        return;
      }
    }
  } else {
    value = el.value;
  }
  krea2Draft[key] = value;
  updateKrea2Dirty();
}

async function saveKrea2(thenTrain = false) {
  if (!current || !krea2) return false;
  const btn = $("#btn-k2-save");
  btn.disabled = true;
  try {
    const result = await api(`/studio/projects/${enc(current.slug)}/krea2`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values: krea2Draft }),
    });
    krea2 = result;
    krea2Draft = { ...(result.values || {}) };
    krea2Dirty = false;
    renderKrea2Form();
    setNotice("#k2-notice", "Krea-2 参数已保存到 krea2.toml。", "ok");
    if (thenTrain) return await startKrea2Train();
    return true;
  } catch (e) {
    setNotice("#k2-notice", "保存失败：" + e.message, "bad");
    return false;
  } finally {
    btn.disabled = false;
  }
}

async function startKrea2Train() {
  if (!current) return false;
  if (krea2Dirty && !(await saveKrea2(false))) return false;
  try {
    const result = await api(`/studio/projects/${enc(current.slug)}/krea2/train`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values: krea2Draft }),
    });
    toast(`Krea-2 训练作业已入队：${result.job_id}`);
    setNotice("#k2-notice", `Krea-2 训练作业已入队：${result.job_id}（${result.state}）`, "ok");
    switchTab("jobs");
    await loadJobs();
    await refreshCurrent();
    return true;
  } catch (e) {
    setNotice("#k2-notice", "Krea-2 训练未开始：" + e.message, "bad");
    return false;
  }
}

function resetKrea2All() {
  if (!krea2) return;
  krea2Draft = {};
  for (const field of krea2.fields) krea2Draft[field.key] = field.default;
  renderKrea2Form();
  setNotice("#k2-notice", "已恢复默认值（尚未保存）。", "");
}

// --- 作业 -------------------------------------------------------------------
async function loadJobs() {
  let list = [];
  try { list = await api("/jobs"); } catch { list = []; }
  jobs = Array.isArray(list) ? list : [];
  $("#jobs-count").textContent = `${jobs.length} 条`;
  const box = $("#jobs");
  box.innerHTML = "";
  for (const j of jobs.slice().reverse()) {
    const el = document.createElement("div");
    const state = j.state || "?";
    el.className = "job" + (j.id === selectedJob ? " sel" : "");
    const label = j.label || j.overrides?.output_name || j.method || j.kind || j.id;
    const running = j.started_at && !j.ended_at ? Date.now() / 1000 - j.started_at : null;
    el.innerHTML = `
      <div class="r1"><b>${esc(label)}</b><span class="tag ${STATE_CLASS[state] || ""}">${esc(state)}</span></div>
      <div class="r2"><span>${esc(j.id)}</span><span>${j.rc != null ? "rc=" + j.rc : running != null ? fmtDuration(running) : ""}</span></div>`;
    el.onclick = () => selectJob(j.id);
    box.appendChild(el);
  }
  if (selectedJob && !jobs.some((j) => j.id === selectedJob)) closeLog();
}

function selectJob(id) {
  selectedJob = id;
  loadJobs();
  loadJobDetail();
  $("#detail-title").textContent = "作业 " + id;
  closeLog();
  $("#log").textContent = "";
  logSource = new EventSource(`/api/jobs/${encodeURIComponent(id)}/logs`);
  logSource.onmessage = (ev) => {
    const pre = $("#log");
    pre.textContent += ev.data + "\n";
    if (pre.textContent.length > 200000) pre.textContent = pre.textContent.slice(-100000);
    pre.scrollTop = pre.scrollHeight;
  };
  logSource.onerror = () => {};
  clearInterval(progressTimer);
  progressTimer = setInterval(() => { if (!document.hidden) loadJobDetail(); }, 3000);
}

function closeLog() {
  if (logSource) { logSource.close(); logSource = null; }
  clearInterval(progressTimer);
  progressTimer = null;
}

async function loadJobDetail() {
  if (!selectedJob) return;
  try {
    const rec = await api(`/jobs/${encodeURIComponent(selectedJob)}`);
    const j = rec.job || rec;
    const started = j.started_at, ended = j.ended_at;
    $("#job-meta").innerHTML = `
      state=<b>${esc(j.state)}</b> · kind=${esc(j.kind)} · method=${esc(j.method || "-")} · preset=${esc(j.preset || "-")}
      · ${started ? fmtDuration((ended || Date.now() / 1000) - started) : "-"}
      ${j.error ? `<div class="bad">error: ${esc(j.error)}</div>` : ""}
      <div class="row" style="margin-top:6px">
        <button data-job="stop" class="mini warn">停止</button>
        <button data-job="pause" class="mini ghost">暂停</button>
        <button data-job="resume" class="mini ghost">恢复</button>
      </div>`;
    const p = await api(`/jobs/${encodeURIComponent(selectedJob)}/progress?last_n=2000`);
    renderProgress(p.events || [], p.count);
  } catch (e) {
    $("#job-meta").textContent = "拿不到作业记录：" + e.message;
  }
}

async function controlJob(action) {
  if (!selectedJob) return;
  try {
    await api(`/jobs/${encodeURIComponent(selectedJob)}/${action}`, { method: "POST" });
    toast(`${action} 已提交`);
    loadJobs();
    loadJobDetail();
  } catch (e) {
    toast(`${action} 失败：` + e.message, true);
  }
}

function renderProgress(events, count) {
  const steps = events.filter((e) => e.ev === "step" && e["loss/current"] != null);
  $("#progress-summary").textContent = `${events.length}/${count || events.length} 条 · step 事件 ${steps.length}`;
  drawLoss(steps);
  const rows = events.slice(-120).reverse().map((e) => {
    const when = e.ts != null ? `${Math.round(e.ts)}s` : "";
    const step = e.global_step != null ? `step ${e.global_step}` : "";
    const loss = e["loss/current"] != null ? `loss ${e["loss/current"].toFixed(4)}` : "";
    const avg = e["loss/average"] != null ? `avg ${e["loss/average"].toFixed(4)}` : "";
    const lr = e["lr/unet"] != null ? `lr ${Number(e["lr/unet"]).toExponential(2)}` : "";
    const extra = e.msg || e.message || e.run || e.error || "";
    return `<div class="tr"><span class="dim">${esc(when)}</span><span>${esc(e.ev)}</span><span>${esc(step)}</span><span>${esc(loss)}</span><span>${esc(avg)}</span><span>${esc(lr)}</span><span class="dim">${esc(extra)}</span></div>`;
  });
  $("#progress").innerHTML =
    '<div class="tr head"><span>t</span><span>ev</span><span>step</span><span>loss</span><span>avg</span><span>lr</span><span></span></div>' +
    rows.join("");
}

function drawLoss(steps) {
  const cv = $("#loss");
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 600, h = 160;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!steps.length) { ctx.fillStyle = "#666"; ctx.fillText("还没有 step 事件", 8, 20); return; }
  const xs = steps.map((s) => s.global_step ?? 0);
  const ys = steps.map((s) => s["loss/current"]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = 24;
  const px = (x) => pad + ((x - x0) / Math.max(1, x1 - x0)) * (w - pad * 2);
  const py = (y) => h - pad - ((y - y0) / Math.max(1e-9, y1 - y0)) * (h - pad * 2);
  ctx.strokeStyle = "#3a3f4b"; ctx.beginPath();
  ctx.moveTo(pad, h - pad); ctx.lineTo(w - pad, h - pad);
  ctx.moveTo(pad, pad); ctx.lineTo(pad, h - pad); ctx.stroke();
  ctx.strokeStyle = "#6cf"; ctx.lineWidth = 1.5; ctx.beginPath();
  steps.forEach((s, i) => {
    const X = px(s.global_step ?? i), Y = py(s["loss/current"]);
    i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
  });
  ctx.stroke();
  ctx.fillStyle = "#9aa"; ctx.font = "11px ui-monospace,monospace";
  ctx.fillText(`step ${x0}→${x1}`, pad, h - 6);
  ctx.fillText(`loss ${y0.toFixed(3)}~${y1.toFixed(3)}`, pad + 84, h - 6);
}

// --- 标签页 / 事件 ----------------------------------------------------------
function switchTab(name) {
  activeTab = name;
  for (const b of $$("#tabs .tab")) b.classList.toggle("active", b.dataset.tab === name);
  for (const pane of $$("#project-view > .tabpane")) {
    pane.classList.toggle("hidden", pane.id !== "tab-" + name);
  }
  if (name === "jobs") { loadJobs(); if (selectedJob) loadJobDetail(); }
  if (name === "preprocess") renderPreprocess();
  if (name === "config") { if (!config) loadProjectConfig(); else renderConfigForm(); }
  if (name === "krea2") { if (!krea2) loadKrea2(); else renderKrea2Form(); }
  if (name === "train") renderOutputs();
}

function bindEvents() {
  $("#btn-new").onclick = () => $("#form-new").classList.toggle("hidden");
  $("#btn-new-cancel").onclick = () => $("#form-new").classList.add("hidden");
  $("#form-new").onsubmit = async (ev) => {
    ev.preventDefault();
    const f = ev.target;
    try {
      const p = await api("/studio/projects", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: f.name.value.trim(), trigger: f.trigger.value.trim(),
          variant: f.variant.value, preset: f.preset.value,
        }),
      });
      f.reset();
      $("#form-new").classList.add("hidden");
      await loadProjects();
      await selectProject(p.slug);
    } catch (e) { toast("新建失败：" + e.message, true); }
  };

  $("#btn-refresh").onclick = async () => {
    await refreshEnv(); await loadProjects();
    if (current) await refreshCurrent();
    if (activeTab === "jobs") await loadJobs();
  };
  $("#btn-queue").onclick = async () => {
    const btn = $("#btn-queue");
    const paused = btn.dataset.paused === "1";
    try {
      await api(paused ? "/queue/start" : "/queue/pause", { method: "POST" });
      btn.dataset.paused = paused ? "0" : "1";
      btn.textContent = paused ? "暂停队列" : "恢复队列";
      toast(paused ? "队列已恢复" : "队列已暂停（当前作业会跑完）");
    } catch (e) { toast("队列操作失败：" + e.message, true); }
  };

  for (const b of $$("#tabs .tab")) b.onclick = () => switchTab(b.dataset.tab);
  for (const b of $$("#image-filters .chip")) {
    b.onclick = () => {
      imageState = b.dataset.state;
      for (const x of $$("#image-filters .chip")) x.classList.toggle("active", x === b);
      loadImages();
    };
  }

  $("#select-all").onchange = (ev) => {
    picked.clear();
    if (ev.target.checked) for (const img of images) picked.add(img.path);
    renderImages();
  };
  $("#btn-accept").onclick = () => review("accept");
  $("#btn-reject").onclick = () => review("reject");
  $("#btn-restore").onclick = () => review("restore");
  $("#btn-delete").onclick = () => {
    if (confirm("永久删除选中的图片（连同 resized / latent 缓存）？")) review("delete");
  };
  $("#btn-import").onclick = importImages;
  $("#btn-tag").onclick = runTag;
  $("#btn-trigger").onclick = applyTrigger;
  $("#btn-preprocess").onclick = runPreprocess;
  $("#btn-cfg-save").onclick = () => saveConfig(false);
  $("#btn-cfg-train").onclick = () => saveConfig(true);
  $("#btn-cfg-reset").onclick = resetConfigAll;
  $("#btn-train").onclick = startTrain;
  $("#btn-goto-config").onclick = () => switchTab("config");
  $("#btn-k2-save").onclick = () => saveKrea2(false);
  $("#btn-k2-train").onclick = startKrea2Train;
  $("#btn-k2-reset").onclick = resetKrea2All;
  $("#cfg-filter").oninput = applyConfigFilter;
  $("#cfg-output").oninput = () => {
    if (!config) return;
    const value = $("#cfg-output").value;
    configDraft["output_name"] = value;
    const mirror = document.querySelector('#cfg-basic [data-key="output_name"]');
    if (mirror) mirror.value = value;
    updateConfigDirty();
  };
  $("#cfg-variant").onchange = async () => {
    if (configDirty && !confirm("切换方法会丢掉未保存的参数改动，继续？")) {
      $("#cfg-variant").value = config?.variant || $("#cfg-variant").value;
      return;
    }
    await loadProjectConfig({ variant: $("#cfg-variant").value, preset: $("#cfg-preset").value });
  };
  $("#cfg-preset").onchange = async () => {
    if (configDirty && !confirm("切换硬件档会丢掉未保存的参数改动，继续？")) {
      $("#cfg-preset").value = config?.preset || $("#cfg-preset").value;
      return;
    }
    await loadProjectConfig({ variant: $("#cfg-variant").value, preset: $("#cfg-preset").value });
  };
  $("#tab-config").addEventListener("input", onConfigFieldInput);
  $("#tab-config").addEventListener("change", onConfigFieldInput);
  $("#tab-krea2").addEventListener("input", onKrea2FieldInput);
  $("#tab-krea2").addEventListener("change", onKrea2FieldInput);
  $("#caption-save").onclick = saveCaption;

  // 事件委托：图片卡片 / 作业按钮
  $("#image-grid").addEventListener("click", (ev) => {
    const act = ev.target.closest("[data-act]")?.dataset.act;
    const card = ev.target.closest(".img-card");
    if (!card) return;
    const img = images.find((x) => x.path === card.dataset.path);
    if (!img) return;
    if (act) { review(act, [img.path]); return; }
    if (ev.target.tagName === "IMG") openCaption(img);
  });
  $("#image-grid").addEventListener("change", (ev) => {
    if (!ev.target.classList.contains("pick-box")) return;
    const card = ev.target.closest(".img-card");
    if (card) togglePick(card.dataset.path, ev.target.checked);
  });
  $("#job-meta").addEventListener("click", (ev) => {
    const act = ev.target.closest("[data-job]")?.dataset.job;
    if (act) controlJob(act);
  });
  for (const b of $$(".mini-tabs .tab")) {
    b.onclick = () => {
      for (const x of $$(".mini-tabs .tab")) x.classList.toggle("active", x === b);
      $("#view-progress").classList.toggle("hidden", b.dataset.view !== "progress");
      $("#view-log").classList.toggle("hidden", b.dataset.view !== "log");
    };
  }
}

// --- 启动 -------------------------------------------------------------------
async function loadStudioSchema() {
  try {
    studioSchema = await api("/studio/schema");
    const form = $("#form-new");
    const variant = form.querySelector('select[name="variant"]');
    const preset = form.querySelector('select[name="preset"]');
    if (variant) fillSelect(variant, studioSchema.variants || [], "lora");
    if (preset) fillSelect(preset, studioSchema.presets || [], "low_vram");
  } catch {
    /* 静态选项仍然可用；schema 只影响下拉里有没有 easycontrol / 自定义档 */
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  await refreshEnv();
  await loadStudioSchema();
  await loadProjects();
  if (projects.length) {
    await selectProject(projects[0].slug);
    // Deep link: `?tab=config` / `?tab=krea2` opens that pane directly.
    const wanted = new URLSearchParams(location.search).get("tab");
    if (wanted && document.querySelector(`#tabs .tab[data-tab="${wanted}"]`)) switchTab(wanted);
  }

  pollTimer = setInterval(async () => {
    if (document.hidden) return;
    if (current) {
      try {
        current = await api(`/studio/projects/${encodeURIComponent(current.slug)}`);
        renderProject();
        renderTag();
        renderPreprocess();
        renderOutputs();
        await loadProjects();
      } catch { /* 页面下一次手动刷新会如实报错 */ }
    }
    if (activeTab === "jobs") await loadJobs();
  }, 5000);
});
