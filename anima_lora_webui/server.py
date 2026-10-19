"""anima_lora_webui —— anima_lora 训练守护进程（anima_daemon）的自有 Web 界面。

设计约束（来自 2026-10-18 的决定）：

* **不碰 Qt**：上游那个 PySide6 GUI 功能全但有 bug，且启动后常驻 160MB 上下。
  这里只要一个浏览器页面 + 一个极薄的本地服务器。
* **零运行时依赖**：只用标准库 —— 装到哪台机器上都能跑，不用 pip install。
* **同源代理**：`anima_daemon` 是 localhost HTTP/JSON，但**不带 CORS 头**，
  所以浏览器不能跨源直连它。这个服务器就把 `/api/*` 反代到守护进程（同源），
  顺带把可访问的路径**白名单化**（不是开放代理）。
* **内存省**：一个线程化的 stdlib HTTP server，实测常驻 20~30MB（见 README）。

启动：

    <引擎的 venv>\\Scripts\\python.exe -m anima_lora_webui.server --root E:\\anima_lora

`--root` 指向 anima_lora 引擎目录（用来找 `output/daemon/daemon.json`）；
不传则退回 `~/.anima/daemon.json`（守护进程的全局镜像）。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

__version__ = "0.1.0"

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: 允许反代到守护进程的路径（前缀匹配）。**不是**开放代理 —— 没在白名单里的
#: 一律 404，避免这个本地服务被当成任意 HTTP 转发器。
GET_ALLOW = ("/health", "/tools", "/jobs", "/events")
POST_ALLOW = ("/jobs", "/queue/", "/shutdown")


# --------------------------------------------------------------------------- #
# 守护进程发现
# --------------------------------------------------------------------------- #
def daemon_pidfile(root: Path | None) -> Path:
    if os.environ.get("ANIMA_DAEMON_PIDFILE"):
        return Path(os.environ["ANIMA_DAEMON_PIDFILE"])
    if root is not None:
        return root / "output" / "daemon" / "daemon.json"
    return Path.home() / ".anima" / "daemon.json"


def discover_daemon(root: Path | None) -> dict:
    """`{port, root, pid}`；找不到就返回空 dict（界面会显示"守护进程没起"）。"""
    path = daemon_pidfile(root)
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(info, dict) or not info.get("port"):
        return {}
    info["pidfile"] = str(path)
    return info


# --------------------------------------------------------------------------- #
# 代理
# --------------------------------------------------------------------------- #
def _allowed(method: str, path: str) -> bool:
    if method == "GET":
        return any(path == p or path.startswith(p + "/") or path.startswith(p + "?") for p in GET_ALLOW)
    if method == "POST":
        return any(path.startswith(p) for p in POST_ALLOW)
    return False


class Handler(BaseHTTPRequestHandler):
    server_version = f"anima-lora-webui/{__version__}"
    protocol_version = "HTTP/1.1"

    # -- 静态 -------------------------------------------------------------- #
    def _send_file(self, rel: str) -> None:
        target = (STATIC_DIR / rel).resolve()
        if STATIC_DIR not in target.parents and target != STATIC_DIR:
            self._send_json(403, {"error": "forbidden"})
            return
        if not target.is_file():
            self._send_json(404, {"error": "not found", "path": rel})
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8" if ctype.startswith("text/") or ctype.endswith("javascript") or ctype.endswith("json") else ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, obj) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- 本地自有端点 ------------------------------------------------------ #
    def _env(self) -> None:
        info = discover_daemon(self.server.engine_root)  # type: ignore[attr-defined]
        self._send_json(
            200,
            {
                "webuiVersion": __version__,
                "engineRoot": str(self.server.engine_root) if self.server.engine_root else None,  # type: ignore[attr-defined]
                "daemon": info,
                "daemonUrl": f"http://127.0.0.1:{info['port']}" if info else None,
                "python": sys.executable,
            },
        )

    # -- 反代 -------------------------------------------------------------- #
    def _proxy(self, method: str, body: bytes | None, target_path: str) -> None:
        info = discover_daemon(self.server.engine_root)  # type: ignore[attr-defined]
        if not info:
            self._send_json(
                503,
                {
                    "error": "daemon not running",
                    "hint": "在引擎目录跑 `python tasks.py daemon`（或界面里点“启动守护进程”）",
                },
            )
            return
        # target_path 已经去掉 /api 前缀（守护进程的路径没有这一层）
        url = f"http://127.0.0.1:{info['port']}{target_path}"
        req = urllib.request.Request(url, data=body, method=method)
        if body is not None:
            req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                ctype = resp.headers.get("Content-Type", "application/json")
                if "text/event-stream" in ctype:
                    self._stream(resp)
                    return
                payload = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as e:  # 守护进程的 4xx/5xx 原样透传
            payload = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            self._send_json(502, {"error": f"daemon unreachable: {e}"})

    def _stream(self, resp) -> None:
        """SSE 直通：SSE 是长连接，不能等 Content-Length。"""
        self.send_response(resp.status)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = resp.readline()
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            resp.close()

    # -- HTTP 入口 --------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/_env":
            self._env()
            return
        if path.startswith("/api/"):
            if not _allowed("GET", path[len("/api") :]):
                self._send_json(404, {"error": "path not allowed", "path": path})
                return
            self._proxy("GET", None, self.path[len("/api") :])
            return
        self._send_file("index.html" if path in ("/", "") else posixpath.normpath(path).lstrip("/"))

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            if not _allowed("POST", path[len("/api") :]):
                self._send_json(404, {"error": "path not allowed", "path": path})
                return
            self._proxy("POST", body, self.path[len("/api") :])
            return
        self._send_json(404, {"error": "not found", "path": path})

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        if os.environ.get("ANIMA_WEBUI_VERBOSE"):
            sys.stderr.write("[webui] %s\n" % (fmt % args))


# --------------------------------------------------------------------------- #
def serve(host: str, port: int, root: Path | None) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.engine_root = root  # type: ignore[attr-defined]
    httpd.daemon_threads = True
    return httpd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="anima_lora_webui.server", description=__doc__)
    ap.add_argument("--root", default=os.environ.get("ANIMA_LORA_ROOT"), help="anima_lora 引擎目录（找 output/daemon/daemon.json）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8810)
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser() if args.root else None
    httpd = serve(args.host, args.port, root)
    info = discover_daemon(root)
    print(f"[webui] http://{args.host}:{args.port}  (engine={root or '-'})")
    print(f"[webui] daemon: {info.get('port') if info else '未运行'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[webui] bye")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
