"""Stdlib HTTP surface for the daemon — zero new deps, localhost only.

A hand-written ``(method, path)`` dispatch on a ``BaseHTTPRequestHandler``;
request bodies are unvalidated ``json.loads``'d dicts. Served by
``ThreadingHTTPServer``, so a parked SSE stream holds one thread.

Full endpoint reference: ``anima_daemon/README.md`` (also served live at
``GET /``); machine-readable manifest at ``GET /tools`` (``TOOLS`` below).
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, easycontrol, studio, tail, toolkits
from .manager import JobManager

logger = logging.getLogger("anima.daemon")

_JOB_RE = re.compile(r"^/jobs/(?P<id>[^/]+)$")
_JOB_STOP_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/stop$")
_JOB_PAUSE_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/pause$")
_JOB_RESUME_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/resume$")
_JOB_LOGS_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/logs$")
_JOB_PROGRESS_RE = re.compile(r"^/jobs/(?P<id>[^/]+)/progress$")

# Training studio (projects / dataset / pipeline). The literal `/studio/`
# prefix is what the WebUI proxy whitelist covers and what the route-contract
# test looks for in `do_GET` / `do_POST`.
_STUDIO_PROJECT_RE = re.compile(r"^/studio/projects/(?P<slug>[^/]+)$")
_STUDIO_SUB_RE = re.compile(
    r"^/studio/projects/(?P<slug>[^/]+)/"
    r"(?P<sub>update|images|file|caption|import|review|trigger|tag|preprocess|train|status|config|krea2)$"
)
_STUDIO_KREA2_TRAIN_RE = re.compile(r"^/studio/projects/(?P<slug>[^/]+)/krea2/train$")

# Toolkit pages (merge / TensorBoard / anime_tools) and the EasyControl tab.
_EASYCONTROL_RE = re.compile(r"^/studio/easycontrol/(?P<variant>[^/]+)$")
_EASYCONTROL_RUN_RE = re.compile(
    r"^/studio/easycontrol/(?P<variant>[^/]+)/(?P<action>preprocess|train)$"
)
_DISTILL_RE = re.compile(r"^/studio/toolkits/distill/(?P<kind>turbo|soup)$")
_DISTILL_RUN_RE = re.compile(r"^/studio/toolkits/distill/(?P<kind>turbo|soup)/run$")

_README = Path(__file__).resolve().parent / "README.md"

# Machine-readable manifest served at GET /tools and registered by mcp.py — one
# entry per operation with a JSON-Schema `input_schema`. Kept in sync with the
# handlers by hand.
TOOLS = [
    {
        "name": "submit_training",
        "description": (
            "Enqueue a train.py run built from method + preset + overrides. "
            "Runs when it reaches the front of the serial queue. Returns {job_id, state}."
        ),
        "method": "POST",
        "path": "/jobs",
        "input_schema": {
            "type": "object",
            "required": ["method"],
            "properties": {
                "method": {
                    "type": "string",
                    "description": "Method/adapter config name (e.g. 'lora', 'soft_tokens', 'easycontrol').",
                },
                "preset": {
                    "type": "string",
                    "default": "default",
                    "description": "Hardware preset: default | low_vram | half | quarter | tenth | graft | debug (configs/presets.toml).",
                },
                "methods_subdir": {
                    "type": "string",
                    "description": "Config subdir, e.g. 'gui-methods' for the clean per-variant tree. Omit for 'methods'.",
                },
                "overrides": {
                    "type": "object",
                    "description": '{key: value} → --key value CLI overrides, e.g. {"network_dim": 32, "max_train_epochs": 64}.',
                },
                "extra": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra CLI args appended verbatim.",
                },
                "config_snapshot": {
                    "type": "object",
                    "description": "A fully-merged config dict to pin instead of re-resolving the base→preset→method→overrides chain at launch.",
                },
                "config_file": {
                    "type": "string",
                    "description": "Path to a config snapshot file (alternative to config_snapshot).",
                },
                "start": {
                    "type": "boolean",
                    "description": "true → run now (resume queue); false → add to queue (held for Start Queue only if the queue is idle; auto-advances behind a job that's already running); omit → leave the gate as-is.",
                },
            },
        },
    },
    {
        "name": "submit_command",
        "description": (
            "Enqueue a plain `python <argv>` task (preprocess, mask, a distill loop). "
            "Optionally carries a chain_train spec the daemon auto-enqueues on success "
            "(how 'preprocess → train' survives the caller closing). Returns {job_id, state}."
        ),
        "method": "POST",
        "path": "/jobs",
        "input_schema": {
            "type": "object",
            "required": ["label", "argv"],
            "properties": {
                "kind": {
                    "type": "string",
                    "const": "command",
                    "description": "Must be 'command'.",
                },
                "label": {
                    "type": "string",
                    "description": "Display label (doubles as the job's 'method' field).",
                },
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argv run under the anima venv, e.g. ['tasks.py', 'preprocess-config', …].",
                },
                "extra_env": {
                    "type": "object",
                    "description": "Extra environment variables for the subprocess.",
                },
                "chain_train": {
                    "type": "object",
                    "description": "Training spec {method, preset, methods_subdir, overrides} auto-enqueued when this command finishes successfully.",
                },
                "stall_timeout": {
                    "type": "number",
                    "description": (
                        "Per-job stall-watchdog budget in seconds, overriding the "
                        "120s command-job default; 0 disables it. Raise (or disable) "
                        "for a legitimately quiet embed/eval loop."
                    ),
                },
                "config_snapshot": {
                    "type": "object",
                    "description": "Config dict to pin (forwarded to the chained train job).",
                },
                "config_file": {
                    "type": "string",
                    "description": "Config snapshot file path (alternative to config_snapshot).",
                },
                "start": {
                    "type": "boolean",
                    "description": "true → run now; false → add to queue (held only if the queue is idle; auto-advances behind a running job); omit → leave gate as-is.",
                },
            },
        },
    },
    {
        "name": "list_jobs",
        "description": "List all jobs (full records, submission order). Each has state ∈ queued|running|paused|done|error|stopped.",
        "method": "GET",
        "path": "/jobs",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_job",
        "description": (
            "Fetch one job record plus live fields: 'latest' (last progress.jsonl event, "
            "train jobs only) and 'stale_for' (seconds since last progress tick). Poll this for completion."
        ),
        "method": "GET",
        "path": "/jobs/{id}",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Job id from submit_*."}
            },
        },
    },
    {
        "name": "get_progress",
        "description": (
            "Query a train job's structured progress.jsonl — events are "
            "run_start | step (loss/lr/metrics) | val (cmmd) | ckpt | log "
            "(mirrored WARNING+ records) | run_end. Filters keep the payload "
            "small on long runs; returns {job_id, state, count, events}. "
            "This is the surface to debug/analyze a run from (loss curve, "
            "warnings, checkpoints) — stdout is only for raw crash output."
        ),
        "method": "GET",
        "path": "/jobs/{id}/progress",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Job id from submit_*."},
                "events": {
                    "type": "string",
                    "description": "Comma-separated ev kinds to keep, e.g. 'step,val' or 'log,run_end'. Omit for all.",
                },
                "since_step": {
                    "type": "integer",
                    "description": "Keep events at/after this global_step (step-less events inherit the preceding step).",
                },
                "every_nth": {
                    "type": "integer",
                    "description": "Thin step events to every n-th (latest step always kept).",
                },
                "last_n": {
                    "type": "integer",
                    "default": 200,
                    "description": "Trailing cap on returned events.",
                },
            },
        },
    },
    {
        "name": "stop_job",
        "description": "Abort a running or queued job (tree-kills the process). Returns {job_id, state}.",
        "method": "POST",
        "path": "/jobs/{id}/stop",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string", "description": "Job id to stop."}},
        },
    },
    {
        "name": "pause_job",
        "description": (
            "Freeze a running job's process tree (SIGSTOP) — VRAM/CUDA context "
            "stay put, SM utilisation drops to zero, resume is instant. The queue "
            "does NOT advance past it (it still owns its slot). Refuses anything "
            "not running, and refuses a multi-GPU accelerate-launch run. Returns "
            "{job_id, state, error?}. With release_model=true (train jobs only) "
            "the trainer instead saves a resumable state at its next optimizer "
            "step and exits: the GPU is freed, the queue advances, and the job "
            "parks as paused with released=true until resume_job relaunches it "
            "with --resume (model reload + recompile, not instant)."
        ),
        "method": "POST",
        "path": "/jobs/{id}/pause",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Job id to pause."},
                "release_model": {
                    "type": "boolean",
                    "description": "Cooperative release: save resumable state, exit, free the GPU (train.py jobs only). Default false = SIGSTOP freeze.",
                },
            },
        },
    },
    {
        "name": "resume_job",
        "description": (
            "Thaw a paused job's process tree (SIGCONT) back to running. A "
            "release-paused job (released=true) is re-enqueued at the front of "
            "the queue with --resume <state_dir> instead. Returns "
            "{job_id, state, error?} (error if the job isn't paused)."
        ),
        "method": "POST",
        "path": "/jobs/{id}/resume",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "string", "description": "Job id to resume."}
            },
        },
    },
    {
        "name": "tail_logs",
        "description": (
            "SSE stream of a job's combined stdout+stderr from the start of the file; "
            "ends with a {ev:'eof', state} event once the job is terminal and drained, "
            "then closes the connection. There is no blocking server-side 'wait' "
            "endpoint — tail this, poll get_job, or use the client-side "
            "DaemonClient.wait / `make daemon-wait JOB=<id>`."
        ),
        "method": "GET",
        "path": "/jobs/{id}/logs",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string", "description": "Job id to tail."}},
        },
    },
    {
        "name": "pause_queue",
        "description": "Hold the queue gate — queued jobs wait until start_queue. Submissions still accepted.",
        "method": "POST",
        "path": "/queue/pause",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "start_queue",
        "description": "Resume a paused queue — the worker launches queued jobs in order.",
        "method": "POST",
        "path": "/queue/start",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "health",
        "description": "Daemon liveness: {ok, pid, port, root, fingerprint, active_job, paused, worker_alive, worker_idle_for}. 'root' is the checkout it belongs to; 'fingerprint' is the daemon-source hash it booted with (stale if it differs from current on-disk source); worker_idle_for is seconds since the job worker last advanced.",
        "method": "GET",
        "path": "/health",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "shutdown",
        "description": "Stop the daemon. Destructive: kill_jobs=true (default) tree-kills the running job too.",
        "method": "POST",
        "path": "/shutdown",
        "input_schema": {
            "type": "object",
            "properties": {
                "kill_jobs": {
                    "type": "boolean",
                    "default": True,
                    "description": "Kill the running job on the way down.",
                }
            },
        },
    },
    # ------------------------------------------------------------------ #
    #  Training studio — the full pipeline around the queue:
    #  fetch → (human) review → optional tag → preprocess → train.
    # ------------------------------------------------------------------ #
    {
        "name": "studio_projects",
        "description": (
            "List training projects (name, trigger, variant, image counts and the five "
            "stage flags). Read-only. Start here before touching a dataset."
        ),
        "method": "GET",
        "path": "/studio/projects",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "studio_create_project",
        "description": (
            "Create a training project. A project owns its inbox (fetched), dataset "
            "(accepted) and rejected trees plus the generated train_config.toml. Slug is "
            "derived from name when omitted; only lowercase letters/digits/_/- are allowed."
        ),
        "method": "POST",
        "path": "/studio/projects",
        "input_schema": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "description": "Human name, e.g. 'shinku style'."},
                "slug": {"type": "string", "description": "Optional ASCII slug; auto from name."},
                "trigger": {
                    "type": "string",
                    "description": "Trigger word prepended to accepted captions (e.g. '@artist').",
                },
                "variant": {
                    "type": "string",
                    "default": "lora",
                    "description": "gui-methods variant: lora | tlora | hydralora | ...",
                },
                "preset": {
                    "type": "string",
                    "default": "low_vram",
                    "description": "Hardware preset from configs/presets.toml.",
                },
                "notes": {"type": "string"},
                "target_res": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [width, height] preprocessing tier.",
                },
                "overrides": {"type": "object", "description": "Train CLI overrides."},
            },
        },
    },
    {
        "name": "studio_project",
        "description": (
            "One project with counts, the five stage flags, its generated config and the "
            "stage→job ledger (with live job states). Read-only."
        ),
        "method": "GET",
        "path": "/studio/projects/{slug}",
        "input_schema": {
            "type": "object",
            "required": ["slug"],
            "properties": {"slug": {"type": "string"}},
        },
    },
    {
        "name": "studio_update_project",
        "description": (
            "Change project settings (name/trigger/variant/preset/output_name/notes/"
            "target_res/overrides). Regenerates the config on the next stage."
        ),
        "method": "POST",
        "path": "/studio/projects/{slug}/update",
        "input_schema": {
            "type": "object",
            "required": ["slug"],
            "properties": {
                "slug": {"type": "string"},
                "name": {"type": "string"},
                "trigger": {"type": "string"},
                "variant": {"type": "string"},
                "preset": {"type": "string"},
                "output_name": {"type": "string"},
                "notes": {"type": "string"},
                "target_res": {"type": "array", "items": {"type": "integer"}},
                "overrides": {"type": "object"},
            },
        },
    },
    {
        "name": "studio_import",
        "description": (
            "Fetch the training set into the project inbox. Accepts local file/dir paths "
            "and/or http(s) image URLs; copies images (max 64MB) plus optional .txt "
            "sidecars. Nothing lands in the training dataset until a human (or "
            "studio_review) accepts it."
        ),
        "method": "POST",
        "path": "/studio/projects/{slug}/import",
        "input_schema": {
            "type": "object",
            "required": ["slug"],
            "properties": {
                "slug": {"type": "string"},
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Absolute local files or directories.",
                },
                "urls": {"type": "array", "items": {"type": "string"}},
                "recursive": {"type": "boolean", "default": False},
                "caption": {
                    "type": "string",
                    "description": "Optional caption written to every imported image.",
                },
            },
        },
    },
    {
        "name": "studio_images",
        "description": (
            "List images with state inbox|dataset|rejected, their caption sidecar and a "
            "thumbnail URL. Use it to show the human what is waiting for review."
        ),
        "method": "GET",
        "path": "/studio/projects/{slug}/images",
        "input_schema": {
            "type": "object",
            "required": ["slug"],
            "properties": {
                "slug": {"type": "string"},
                "state": {
                    "type": "string",
                    "enum": ["all", "inbox", "dataset", "rejected"],
                    "default": "all",
                },
                "limit": {"type": "integer", "default": 500},
            },
        },
    },
    {
        "name": "studio_review",
        "description": (
            "The human-filtering step: accept / reject / restore / delete a batch of "
            "images. Accept moves inbox→dataset; reject moves to rejected and purges any "
            "derived resize/latent/mask so a rejected image can never train."
        ),
        "method": "POST",
        "path": "/studio/projects/{slug}/review",
        "input_schema": {
            "type": "object",
            "required": ["slug", "action", "files"],
            "properties": {
                "slug": {"type": "string"},
                "action": {"type": "string", "enum": ["accept", "reject", "restore", "delete"]},
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths as returned by studio_images ('inbox/a.jpg').",
                },
            },
        },
    },
    {
        "name": "studio_caption",
        "description": "Read one image's caption sidecar. Read-only.",
        "method": "GET",
        "path": "/studio/projects/{slug}/caption",
        "input_schema": {
            "type": "object",
            "required": ["slug", "path"],
            "properties": {
                "slug": {"type": "string"},
                "path": {"type": "string", "description": "e.g. 'dataset/a.jpg'."},
            },
        },
    },
    {
        "name": "studio_set_caption",
        "description": (
            "Write one caption sidecar (max 20k chars). Changing a caption invalidates "
            "that image's TE cache — run studio_preprocess again before training."
        ),
        "method": "POST",
        "path": "/studio/projects/{slug}/caption",
        "input_schema": {
            "type": "object",
            "required": ["slug", "path", "caption"],
            "properties": {
                "slug": {"type": "string"},
                "path": {"type": "string"},
                "caption": {"type": "string"},
            },
        },
    },
    {
        "name": "studio_apply_trigger",
        "description": (
            "Prepend the project trigger word to every accepted caption that lacks it "
            "(idempotent). Use after tagging; tags are not re-encoded until preprocess."
        ),
        "method": "POST",
        "path": "/studio/projects/{slug}/trigger",
        "input_schema": {
            "type": "object",
            "required": ["slug"],
            "properties": {
                "slug": {"type": "string"},
                "trigger": {"type": "string", "description": "Override the project trigger."},
            },
        },
    },
    {
        "name": "studio_tag",
        "description": (
            "Optional tagging stage: enqueue `tasks.py caption-autotag --apply` over the "
            "accepted dataset (mode missing|merge|overwrite). Returns a daemon job id; "
            "follow it with studio_project / get_job / tail_log."
        ),
        "method": "POST",
        "path": "/studio/projects/{slug}/tag",
        "input_schema": {
            "type": "object",
            "required": ["slug"],
            "properties": {
                "slug": {"type": "string"},
                "mode": {"type": "string", "enum": ["missing", "merge", "overwrite"], "default": "missing"},
            },
        },
    },
    {
        "name": "studio_preprocess",
        "description": (
            "Preprocess stage: resize → VAE latents → TE cache into the project's scoped "
            "dirs (preprocess-config with the generated dataset config). Set "
            "chain_train=true to auto-enqueue training when it succeeds."
        ),
        "method": "POST",
        "path": "/studio/projects/{slug}/preprocess",
        "input_schema": {
            "type": "object",
            "required": ["slug"],
            "properties": {
                "slug": {"type": "string"},
                "chain_train": {
                    "type": "boolean",
                    "default": False,
                    "description": "Auto-enqueue training after a successful preprocess.",
                },
            },
        },
    },
    {
        "name": "studio_train",
        "description": (
            "Formal training: enqueue a train.py job from the project's generated config "
            "(paths are scoped to this project). Requires the preprocessing caches; "
            "overrides here are persisted on the project first."
        ),
        "method": "POST",
        "path": "/studio/projects/{slug}/train",
        "input_schema": {
            "type": "object",
            "required": ["slug"],
            "properties": {
                "slug": {"type": "string"},
                "overrides": {
                    "type": "object",
                    "description": 'e.g. {"max_train_epochs": 8, "network_dim": 32}.',
                },
            },
        },
    },
    {
        "name": "studio_status",
        "description": (
            "Project + live stage→job ledger. The one call to answer 'where is this "
            "training run and what should happen next?'. Read-only."
        ),
        "method": "GET",
        "path": "/studio/projects/{slug}/status",
        "input_schema": {
            "type": "object",
            "required": ["slug"],
            "properties": {"slug": {"type": "string"}},
        },
    },
]


class _Handler(BaseHTTPRequestHandler):
    server_version = "AnimaDaemon/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def manager(self) -> JobManager:
        return self.server.manager  # type: ignore[attr-defined]

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, *, content_type: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def _open_sse(self) -> None:
        """Open an SSE response — one response per connection, never keep-alive.

        An SSE body has no ``Content-Length``/chunked framing, so the socket
        closing is the client's only EOF signal. With keep-alive every consumer
        hangs after the ``eof`` event — keep ``Connection: close`` +
        ``close_connection``.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse(self, obj) -> bool:
        """Write one SSE event and flush. Returns False on a dropped client —
        ``wfile`` buffers, so without the flush the client sees nothing until
        the buffer fills."""
        try:
            payload = obj if isinstance(obj, str) else json.dumps(obj)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def log_message(self, fmt, *args) -> None:  # quieter than default stderr spam
        logger.debug("http: " + fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        if path in ("/", "/readme"):
            self._handle_readme()
        elif path == "/tools":
            self._send_json(TOOLS)
        elif path == "/health":
            self._handle_health()
        elif path == "/jobs":
            self._handle_list()
        elif path == "/events":
            self._handle_events()
        elif m := _JOB_LOGS_RE.match(path):
            self._handle_logs(m.group("id"))
        elif m := _JOB_PROGRESS_RE.match(path):
            self._handle_progress(m.group("id"), query)
        elif path.startswith("/studio/toolkits/") or path.startswith("/studio/easycontrol"):
            self._handle_toolkit_get(path, query)
        elif path.startswith("/studio/"):
            self._handle_studio_get(path, query)
        elif m := _JOB_RE.match(path):
            self._handle_get(m.group("id"))
        else:
            self._send_json({"error": "not found", "path": path}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/jobs":
            self._handle_submit()
        elif path == "/queue/start":
            self.manager.resume()
            self._send_json({"ok": True, "paused": False})
        elif path == "/queue/pause":
            self.manager.pause()
            self._send_json({"ok": True, "paused": True})
        elif path == "/shutdown":
            self._handle_shutdown()
        elif m := _JOB_STOP_RE.match(path):
            self._handle_stop(m.group("id"))
        elif m := _JOB_PAUSE_RE.match(path):
            self._handle_pause(m.group("id"))
        elif m := _JOB_RESUME_RE.match(path):
            self._handle_resume(m.group("id"))
        elif path.startswith("/studio/toolkits/") or path.startswith("/studio/easycontrol"):
            self._handle_toolkit_post(path)
        elif path.startswith("/studio/"):
            self._handle_studio_post(path)
        else:
            self._send_json({"error": "not found", "path": path}, 404)

    def _handle_readme(self) -> None:
        try:
            text = _README.read_text(encoding="utf-8")
        except OSError:
            # README not shipped (e.g. a trimmed vendor tree): point at /tools.
            self._send_json({"error": "README.md not found", "tools": "/tools"}, 404)
            return
        self._send_text(text, content_type="text/markdown; charset=utf-8")

    def _handle_health(self) -> None:
        active = self.manager.active_job()
        self._send_json(
            {
                "ok": True,
                "pid": os.getpid(),
                "port": self.server.server_address[1],
                "root": str(config.ROOT),
                # Fingerprint we BOOTED with, not re-hashed live — see
                # config.source_fingerprint / client.daemon_is_stale.
                "fingerprint": getattr(self.server, "boot_fingerprint", None),
                "active_job": active.id if active else None,
                "paused": self.manager.is_paused(),
                "worker_alive": self.manager.worker_alive(),
                "worker_idle_for": self.manager.worker_idle_for(),
            }
        )

    def _handle_submit(self) -> None:
        body = self._read_json()
        # start: True → run now (resume queue), False → enqueue paused, None → as-is.
        start = body.get("start")
        if (body.get("kind") or "train") == "command":
            argv = body.get("argv")
            if not isinstance(argv, list) or not argv:
                self._send_json({"error": "missing 'argv' for command job"}, 400)
                return
            stall_timeout = body.get("stall_timeout")
            try:
                stall_timeout = (
                    float(stall_timeout) if stall_timeout is not None else None
                )
            except (TypeError, ValueError):
                self._send_json({"error": "'stall_timeout' must be a number"}, 400)
                return
            job = self.manager.submit_command(
                label=body.get("label") or "command",
                argv=[str(a) for a in argv],
                extra_env=body.get("extra_env") or {},
                chain_train=body.get("chain_train") or None,
                stall_timeout=stall_timeout,
                config_snapshot=body.get("config_snapshot") or None,
                config_file=body.get("config_file") or None,
                start=start,
                captured_env=body.get("captured_env") or {},
            )
            self._send_json({"job_id": job.id, "state": job.state}, 201)
            return
        method = body.get("method")
        if not method:
            self._send_json({"error": "missing 'method'"}, 400)
            return
        job = self.manager.submit(
            method=method,
            preset=body.get("preset") or "default",
            methods_subdir=body.get("methods_subdir"),
            config_snapshot=body.get("config_snapshot") or None,
            config_file=body.get("config_file") or None,
            overrides=body.get("overrides") or {},
            extra=body.get("extra") or [],
            start=start,
            captured_env=body.get("captured_env") or {},
        )
        self._send_json({"job_id": job.id, "state": job.state}, 201)

    def _handle_list(self) -> None:
        self._send_json([j.public() for j in self.manager.list_jobs()])

    def _handle_get(self, job_id: str) -> None:
        job = self.manager.get(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        out = job.public()
        out["latest"] = tail.last_event(job.progress_path)
        out["stale_for"] = self.manager.stale_for(job)
        self._send_json(out)

    def _handle_progress(self, job_id: str, query: str) -> None:
        job = self.manager.get(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        params = urllib.parse.parse_qs(query)

        def _int(name: str, default=None):
            raw = (params.get(name) or [None])[0]
            try:
                return int(raw) if raw is not None else default
            except ValueError:
                return default

        raw_events = (params.get("events") or [None])[0]
        kinds = (
            [s.strip() for s in raw_events.split(",") if s.strip()]
            if raw_events
            else None
        )
        evs = tail.read_events(
            job.progress_path,
            events=kinds,
            since_step=_int("since_step"),
            every_nth=_int("every_nth"),
            last_n=_int("last_n", 200),
        )
        self._send_json(
            {
                "job_id": job.id,
                "state": job.state,
                "progress_path": job.progress_path,
                "count": len(evs),
                "events": evs,
            }
        )

    def _handle_stop(self, job_id: str) -> None:
        job = self.manager.stop(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        self._send_json({"job_id": job.id, "state": job.state})

    def _handle_pause(self, job_id: str) -> None:
        body = self._read_json()
        result = self.manager.pause_job(
            job_id, release_model=bool(body.get("release_model", False))
        )
        if result is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        # A refusal rides an `error` field in a 200 body; only a missing job 404s.
        self._send_json(result)

    def _handle_resume(self, job_id: str) -> None:
        result = self.manager.resume_job(job_id)
        if result is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        self._send_json(result)

    def _handle_shutdown(self) -> None:
        body = self._read_json()
        kill = bool(body.get("kill_jobs", True))
        self._send_json({"ok": True, "kill_jobs": kill})
        # Trigger shutdown after the response is flushed, off the handler thread
        # (server.shutdown() must not run in a request thread).
        threading.Thread(
            target=self.server.request_shutdown,  # type: ignore[attr-defined]
            args=(kill,),
            daemon=True,
        ).start()

    # ---- training studio ------------------------------------------------ #
    def _studio_guard(self, fn):
        """Run one studio operation, mapping caller errors to HTTP 400."""
        try:
            return self._send_json(fn())
        except (studio.StudioError, toolkits.ToolkitError, easycontrol.EasyControlError) as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001 — keep the daemon alive, report honestly
            logger.exception("studio operation failed")
            self._send_json({"error": f"studio 操作失败：{e}"}, 500)

    def _query_one(self, query: str, name: str, default: str = "") -> str:
        values = urllib.parse.parse_qs(query or "").get(name)
        return values[0] if values else default

    # ---- toolkit pages (merge / TensorBoard / anime_tools / EasyControl) -- #
    def _handle_toolkit_get(self, path: str, query: str) -> None:
        p = path.rstrip("/")
        if p == "/studio/toolkits/merge":
            dir_name = self._query_one(query, "dir")
            file_name = self._query_one(query, "file")
            self._studio_guard(lambda: toolkits.merge_state(dir_name, file_name))
            return
        if p == "/studio/toolkits/tensorboard":
            self._studio_guard(toolkits.tensorboard_state)
            return
        if p == "/studio/toolkits/anime":
            self._studio_guard(toolkits.anime_state)
            return
        m = _DISTILL_RE.match(p)
        if m:
            self._studio_guard(lambda: toolkits.distill_state(m.group("kind")))
            return
        if p == "/studio/easycontrol":
            self._studio_guard(easycontrol.list_variants)
            return
        m = _EASYCONTROL_RE.match(p)
        if m:
            preset = self._query_one(query, "preset", "default") or "default"
            self._studio_guard(lambda: easycontrol.detail(m.group("variant"), preset))
            return
        self._send_json({"error": "not found", "path": path}, 404)

    def _handle_toolkit_post(self, path: str) -> None:
        body = self._read_json()
        p = path.rstrip("/")
        if p == "/studio/toolkits/merge":
            self._studio_guard(lambda: toolkits.merge_run(body, self.manager))
            return
        if p == "/studio/toolkits/tensorboard/start":
            self._studio_guard(lambda: toolkits.tensorboard_start(body))
            return
        if p == "/studio/toolkits/tensorboard/stop":
            self._studio_guard(toolkits.tensorboard_stop)
            return
        if p == "/studio/toolkits/tensorboard/remove":
            self._studio_guard(lambda: toolkits.tensorboard_remove(body))
            return
        if p == "/studio/toolkits/anime/start":
            self._studio_guard(lambda: toolkits.anime_start(body))
            return
        m = _DISTILL_RUN_RE.match(p)
        if m:
            kind = m.group("kind")
            self._studio_guard(lambda: toolkits.distill_run(kind, body, self.manager))
            return
        m = _DISTILL_RE.match(p)
        if m:
            kind = m.group("kind")
            values = body.get("values")
            if not isinstance(values, dict):
                self._send_json({"error": "缺少 values"}, 400)
                return
            self._studio_guard(lambda: toolkits.distill_save(kind, values))
            return
        m = _EASYCONTROL_RUN_RE.match(p)
        if m:
            variant, action = m.group("variant"), m.group("action")
            if action == "preprocess":
                self._studio_guard(lambda: easycontrol.preprocess(variant, self.manager))
            else:
                preset = str(body.get("preset") or "default")
                self._studio_guard(lambda: easycontrol.train(variant, preset, self.manager))
            return
        m = _EASYCONTROL_RE.match(p)
        if m:
            self._studio_guard(lambda: easycontrol.save(m.group("variant"), body))
            return
        self._send_json({"error": "not found", "path": path}, 404)

    def _handle_studio_get(self, path: str, query: str) -> None:
        if path.rstrip("/") == "/studio/projects":
            self._studio_guard(studio.list_projects)
            return
        if path.rstrip("/") == "/studio/schema":
            self._studio_guard(studio.config_schema)
            return
        m = _STUDIO_SUB_RE.match(path)
        if not m:
            m2 = _STUDIO_PROJECT_RE.match(path)
            if m2:
                self._studio_guard(lambda: studio.status(m2.group("slug"), self.manager))
                return
            self._send_json({"error": "not found", "path": path}, 404)
            return
        slug, sub = m.group("slug"), m.group("sub")
        if sub == "images":
            try:
                limit = int(self._query_one(query, "limit", "500") or 500)
            except ValueError:
                limit = 500
            state = self._query_one(query, "state", "all")
            self._studio_guard(lambda: studio.list_images(slug, state, limit))
        elif sub == "caption":
            rel = self._query_one(query, "path")
            self._studio_guard(lambda: studio.get_caption(slug, rel))
        elif sub == "file":
            rel = self._query_one(query, "path")
            self._send_studio_file(slug, rel)
        elif sub == "status":
            self._studio_guard(lambda: studio.status(slug, self.manager))
        elif sub == "config":
            variant = self._query_one(query, "variant") or None
            preset = self._query_one(query, "preset") or None
            self._studio_guard(lambda: studio.project_config(slug, variant, preset))
        elif sub == "krea2":
            self._studio_guard(lambda: studio.project_krea2(slug))
        else:
            self._send_json({"error": "not found", "path": path}, 404)

    def _send_studio_file(self, slug: str, rel: str) -> None:
        try:
            path = studio.resolve_file(slug, rel)
        except studio.StudioError as e:
            self._send_json({"error": str(e)}, 400)
            return
        try:
            data = path.read_bytes()
        except OSError as e:
            self._send_json({"error": f"读不到文件：{e}"}, 404)
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_studio_post(self, path: str) -> None:
        body = self._read_json()
        if path.rstrip("/") == "/studio/projects":
            self._studio_guard(lambda: studio.create_project(body))
            return
        krea2_train = _STUDIO_KREA2_TRAIN_RE.match(path)
        if krea2_train:
            self._studio_guard(
                lambda: studio.enqueue_krea2_train(krea2_train.group("slug"), self.manager, body)
            )
            return
        m = _STUDIO_SUB_RE.match(path)
        if not m:
            self._send_json({"error": "not found", "path": path}, 404)
            return
        slug, sub = m.group("slug"), m.group("sub")
        if sub == "update":
            self._studio_guard(lambda: studio.update_project(slug, body))
        elif sub == "import":
            self._studio_guard(lambda: studio.import_images(slug, body))
        elif sub == "review":
            self._studio_guard(lambda: studio.review_images(slug, body))
        elif sub == "caption":
            self._studio_guard(lambda: studio.set_caption(slug, body))
        elif sub == "trigger":
            self._studio_guard(lambda: studio.apply_trigger(slug, body))
        elif sub == "tag":
            self._studio_guard(lambda: studio.enqueue_tag(slug, self.manager, body))
        elif sub == "preprocess":
            self._studio_guard(lambda: studio.enqueue_preprocess(slug, self.manager, body))
        elif sub == "train":
            self._studio_guard(lambda: studio.enqueue_train(slug, self.manager, body))
        elif sub == "config":
            self._studio_guard(lambda: studio.save_project_config(slug, body))
        elif sub == "krea2":
            self._studio_guard(lambda: studio.save_project_krea2(slug, body))
        else:
            self._send_json({"error": "not found", "path": path}, 404)

    def _handle_events(self) -> None:
        q = self.manager.subscribe()
        self._open_sse()
        try:
            if not self._sse({"ev": "hello", "ts": time.time()}):
                return
            while True:
                try:
                    event = q.get(timeout=15)
                except Exception:
                    # idle keepalive comment so proxies/clients don't time out
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                    continue
                if not self._sse(event):
                    return
        finally:
            self.manager.unsubscribe(q)

    def _handle_logs(self, job_id: str) -> None:
        job = self.manager.get(job_id)
        if job is None:
            self._send_json({"error": "no such job", "job_id": job_id}, 404)
            return
        self._open_sse()
        path = Path(job.stdout_path) if job.stdout_path else None
        if path is None:
            self._sse({"error": "no stdout for job"})
            return
        for line in tail.follow(path, from_start=True):
            if line:
                if not self._sse(line.rstrip("\n")):
                    return
            else:
                # Empty tick = log drained; stop once the job has left
                # queued/running.
                cur = self.manager.get(job_id)
                if cur is not None and cur.state not in ("queued", "running"):
                    self._sse({"ev": "eof", "state": cur.state})
                    return


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # SO_REUSEADDR means "rebind a TIME_WAIT socket" on POSIX (safe) but
    # "double-bind a live in-use port" on Windows, defeating
    # serve_with_fallback's collision detection — so Windows must fail loudly.
    allow_reuse_address = os.name != "nt"

    def __init__(self, addr, manager: JobManager, *, fingerprint=None):
        super().__init__(addr, _Handler)
        self.manager = manager
        self.boot_fingerprint = fingerprint

    def request_shutdown(self, kill_jobs: bool) -> None:
        self.manager.shutdown(kill_jobs=kill_jobs)
        self.shutdown()  # unblocks serve_forever()


def serve(manager: JobManager, *, port: int, fingerprint=None) -> _Server:
    """Bind 127.0.0.1:port and return the server (call ``serve_forever``)."""
    return _Server((config.HOST, port), manager, fingerprint=fingerprint)


def serve_with_fallback(manager: JobManager, *, port: int, fingerprint=None) -> _Server:
    """Bind ``port``; if it's already taken, fall back to an OS-chosen free one.

    On ``EADDRINUSE`` we first probe the port: if an anima daemon already
    answers ``/health`` there (a startup-race sibling that won), we re-raise
    so the caller stands down instead of spinning up a second daemon that
    would overwrite the pidfile. Only a *stranger* holding the port sends us
    to an ephemeral one (``ensure_daemon`` re-resolves it from the pidfile).
    """
    try:
        return _Server((config.HOST, port), manager, fingerprint=fingerprint)
    except OSError:
        from .client import DaemonClient

        # A sibling may have bound the socket but not yet reached
        # serve_forever; probe a few times.
        for _ in range(3):
            if DaemonClient(port).health(timeout=0.5) is not None:
                raise  # an anima daemon owns it → let the caller stand down
            time.sleep(0.3)
        logger.warning(
            "127.0.0.1:%s held by a non-anima process; using an ephemeral port",
            port,
        )
        return _Server((config.HOST, 0), manager, fingerprint=fingerprint)
