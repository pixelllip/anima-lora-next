// Anima LoRA 训练台 —— 无框架、无构建步骤的前端。
// 所有请求都打到同源的 /api/*（由 anima_lora_webui.server 反代到 anima_daemon），
// 因为守护进程不带 CORS 头，浏览器不能跨源直连它。

const $ = (sel) => document.querySelector(sel);
const STATE_CLASS = { running: "ok", done: "done", queued: "wait", paused: "wait", error: "bad", stopped: "bad" };

let jobs = [];
let selected = null;
let logSource = null;
let progressTimer = null;

async function api(path, opts) {
  const res = await fetch("/api" + path, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }
  if (!res.ok) throw Object.assign(new Error(`HTTP ${res.status}`), { data });
  return data;
}

function fmtDuration(sec) {
  if (sec == null) return "-";
  const s = Math.max(0, Math.round(sec));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
}

function jobLabel(j) {
  const target = j.result_summary?.label || j.overrides?.output_name || j.method || j.label || j.kind;
  return target || j.id;
}

// --- 环境 / 守护进程 ---------------------------------------------------------
async function refreshEnv() {
  try {
    const env = await api("/_env");
    const d = env.daemon || {};
    $("#env").textContent = d.port ? `守护进程 :${d.port} ｜ 引擎 ${env.engineRoot || "-"}` : "守护进程未运行";
    $("#env").className = "pill " + (d.port ? "ok" : "bad");
  } catch (e) {
    $("#env").textContent = "WebUI 后端异常：" + e.message;
    $("#env").className = "pill bad";
  }
}

// --- 作业列表 ---------------------------------------------------------------
async function loadJobs() {
  let list = [];
  try { list = await api("/jobs"); } catch { list = []; }
  jobs = Array.isArray(list) ? list : [];
  const box = $("#jobs");
  box.innerHTML = "";
  $("#jobs-count").textContent = `（${jobs.length}）`;
  for (const j of jobs.slice().reverse()) {
    const el = document.createElement("div");
    el.className = "job " + (j.id === selected ? "sel" : "");
    const state = j.state || "?";
    const running = j.started_at && !j.ended_at ? (Date.now() / 1000 - j.started_at) : null;
    el.innerHTML = `
      <div class="row"><b>${jobLabel(j)}</b><span class="tag ${STATE_CLASS[state] || ""}">${state}</span></div>
      <div class="row dim small"><span>${j.id}</span><span>${j.rc != null ? "rc=" + j.rc : running != null ? fmtDuration(running) : ""}</span></div>`;
    el.onclick = () => selectJob(j.id);
    box.appendChild(el);
  }
}

async function control(id, action) {
  try { await api(`/jobs/${id}/${action}`, { method: "POST" }); } catch (e) { alert(`${action} 失败：${e.message}`); }
  loadJobs(); if (selected === id) loadProgress();
}

// --- 选中作业 ---------------------------------------------------------------
function selectJob(id) {
  selected = id;
  $("#detail-title").textContent = "作业 " + id;
  loadJobs();
  loadProgress();
  if (logSource) { logSource.close(); logSource = null; }
  $("#log").textContent = "";
  logSource = new EventSource(`/api/jobs/${id}/logs`);
  logSource.onmessage = (ev) => {
    const pre = $("#log");
    pre.textContent += ev.data + "\n";
    if (pre.textContent.length > 200000) pre.textContent = pre.textContent.slice(-100000);
    pre.scrollTop = pre.scrollHeight;
  };
  logSource.onerror = () => { /* 作业结束会 EOF，浏览器自己重连；不用报错 */ };
}

async function loadProgress() {
  if (!selected) return;
  try {
    const rec = await api(`/jobs/${selected}`);
    const j = rec.job || rec;
    const started = j.started_at, ended = j.ended_at;
    $("#job-meta").innerHTML = `
      state=<b>${j.state}</b> · kind=${j.kind} · method=${j.method || "-"} · preset=${j.preset || "-"}
      · ${started ? fmtDuration((ended || Date.now() / 1000) - started) : "-"}
      ${j.error ? `<div class="bad">error: ${j.error}</div>` : ""}
      <div class="row"><button onclick="control('${j.id}','stop')">停止</button>
      <button onclick="control('${j.id}','pause')">暂停</button>
      <button onclick="control('${j.id}','resume')">恢复</button></div>`;
    const p = await api(`/jobs/${selected}/progress?last_n=2000`);
    renderProgress(p.events || [], p.count);
  } catch (e) {
    $("#job-meta").textContent = "拿不到作业记录：" + e.message;
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
    return `<div class="tr"><span class="dim">${when}</span><span>${e.ev}</span><span>${step}</span><span>${loss}</span><span>${avg}</span><span>${lr}</span><span class="dim">${extra}</span></div>`;
  });
  $("#progress").innerHTML = `<div class="tr head"><span>t</span><span>ev</span><span>step</span><span>loss</span><span>avg</span><span>lr</span><span></span></div>` + rows.join("");
}

function drawLoss(steps) {
  const cv = $("#loss");
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 600, h = 180;
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
  ctx.moveTo(pad, h - pad); ctx.lineTo(w - pad, h - pad); ctx.moveTo(pad, pad); ctx.lineTo(pad, h - pad); ctx.stroke();
  ctx.strokeStyle = "#6cf"; ctx.lineWidth = 1.5; ctx.beginPath();
  steps.forEach((s, i) => { const X = px(s.global_step ?? i), Y = py(s["loss/current"]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
  ctx.stroke();
  ctx.fillStyle = "#9aa"; ctx.font = "11px ui-monospace,monospace";
  ctx.fillText(`step ${x0}→${x1}`, pad, h - 6);
  ctx.fillText(`loss ${y0.toFixed(3)}~${y1.toFixed(3)}`, pad + 84, h - 6);
}

// --- 提交 -------------------------------------------------------------------
function overridesOf(form) {
  const out = {};
  for (const el of form.querySelectorAll("input[name]")) {
    const v = el.value.trim();
    if (!v) continue;
    out[el.name] = /^-?\d+$/.test(v) ? Number(v) : /^-?\d*\.\d+(e-?\d+)?$/i.test(v) ? Number(v) : v.replace(/^["']|["']$/g, "");
  }
  return out;
}

async function submitTrain(ev) {
  ev.preventDefault();
  const form = ev.target;
  const extra = form.extra.value.split("\n").map((s) => s.trim()).filter(Boolean);
  const body = {
    method: form.method.value.trim(),
    preset: form.preset.value.trim() || "default",
    methods_subdir: form.methods_subdir.value,
    overrides: overridesOf(form),
    extra,
    start: true,
  };
  const msg = $("#submit-msg");
  msg.textContent = "提交中…";
  try {
    const r = await api("/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    msg.textContent = `已入队 ${r.job_id}（${r.state}）`;
    loadJobs();
  } catch (e) { msg.textContent = "失败：" + e.message + " " + JSON.stringify(e.data || {}); }
}

async function submitCommand(ev) {
  ev.preventDefault();
  const form = ev.target;
  const argv = form.argv.value.split("\n").map((s) => s.trim()).filter(Boolean);
  const msg = $("#cmd-msg");
  if (!argv.length) { msg.textContent = "argv 不能为空"; return; }
  msg.textContent = "提交中…";
  try {
    const r = await api("/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind: "command", label: form.label.value.trim(), argv, start: true }) });
    msg.textContent = `已入队 ${r.job_id}（${r.state}）`;
    loadJobs();
  } catch (e) { msg.textContent = "失败：" + e.message + " " + JSON.stringify(e.data || {}); }
}

async function toggleQueue() {
  const btn = $("#btn-pause");
  const paused = btn.dataset.paused === "1";
  try {
    await api(paused ? "/queue/start" : "/queue/pause", { method: "POST" });
    btn.dataset.paused = paused ? "0" : "1";
    btn.textContent = paused ? "暂停队列" : "恢复队列";
  } catch (e) { alert("队列操作失败：" + e.message); }
}

// --- 启动 -------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  $("#btn-refresh").onclick = () => { refreshEnv(); loadJobs(); loadProgress(); };
  $("#btn-pause").onclick = toggleQueue;
  $("#form-train").onsubmit = submitTrain;
  $("#form-cmd").onsubmit = submitCommand;
  for (const b of document.querySelectorAll(".tab")) {
    b.onclick = () => {
      for (const x of document.querySelectorAll(".tab")) x.classList.toggle("active", x === b);
      for (const pane of document.querySelectorAll(".tabpane")) pane.classList.add("hidden");
      $("#tab-" + b.dataset.tab).classList.remove("hidden");
    };
  }
  refreshEnv();
  loadJobs();
  setInterval(() => { if (!document.hidden) { loadJobs(); if (selected) loadProgress(); } }, 3000);
});
