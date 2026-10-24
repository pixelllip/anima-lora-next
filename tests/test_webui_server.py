"""服务器自测：纯标准库（起一个假守护进程 + 真起 WebUI），不依赖 PySide6/torch。

    python -m pytest tests/test_webui_server.py -q
"""

from __future__ import annotations

import json
import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anima_lora_webui import server as webui  # noqa: E402


# --------------------------------------------------------------------------- #
# 假守护进程：记录收到的请求，按路径回固定 JSON
# --------------------------------------------------------------------------- #
class FakeDaemon(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: list[tuple[str, str, bytes]] = []

    def _respond(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        FakeDaemon.seen.append(("GET", self.path, b""))
        if self.path == "/health":
            return self._respond({"ok": True, "port": self.server.server_port})
        if self.path.startswith("/jobs"):
            return self._respond([{"id": "j1", "state": "done"}])
        if self.path == "/events":
            body = b"data: hello\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return self._respond({"error": "nope"}, 404)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        FakeDaemon.seen.append(("POST", self.path, body))
        if self.path == "/jobs":
            return self._respond({"job_id": "new-1", "state": "queued"}, 201)
        return self._respond({"error": "nope"}, 404)

    def log_message(self, *a):  # noqa: A003
        pass


@pytest.fixture()
def fake_daemon():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeDaemon)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    FakeDaemon.seen = []
    yield httpd
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def webui_server(tmp_path, fake_daemon):
    """真起 WebUI：pidfile 指向假守护进程的端口（engine root 用 tmp_path）。"""
    (tmp_path / "output" / "daemon").mkdir(parents=True)
    (tmp_path / "output" / "daemon" / "daemon.json").write_text(
        json.dumps({"pid": 1, "port": fake_daemon.server_port, "root": str(tmp_path)}), encoding="utf-8"
    )
    httpd = webui.serve("127.0.0.1", 0, tmp_path)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_env_endpoint_reports_discovered_daemon(webui_server):
    status, body = _get(webui_server + "/api/_env")
    assert status == 200
    data = json.loads(body)
    assert data["daemon"]["port"] > 0
    assert data["daemonUrl"].startswith("http://127.0.0.1:")


def test_static_index_and_assets_are_served(webui_server):
    for path, needle in (("/", b"Anima LoRA"), ("/app.js", b"/api/jobs"), ("/style.css", b"--bg")):
        status, body = _get(webui_server + path)
        assert status == 200, path
        assert needle in body, path


def test_get_proxies_allowlisted_paths_only(webui_server):
    status, body = _get(webui_server + "/api/jobs")
    assert status == 200 and json.loads(body)[0]["id"] == "j1"
    status, _ = _get(webui_server + "/api/../../etc/passwd")
    assert status in (404, 400)
    status, body = _get(webui_server + "/api/shutdown")
    assert status == 404, "GET /shutdown 不该被放行"


def test_post_jobs_is_proxied_with_body(webui_server):
    body = json.dumps({"method": "lora", "start": True}).encode()
    req = urllib.request.Request(webui_server + "/api/jobs", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 201
        assert json.loads(r.read())["job_id"] == "new-1"
    assert ("POST", "/jobs", body) in FakeDaemon.seen


def test_post_outside_allowlist_is_refused(webui_server):
    req = urllib.request.Request(webui_server + "/api/whatever", data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
        raise AssertionError("应该被拒绝")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_allowlist_helper():
    assert webui._allowed("GET", "/jobs/abc/progress")
    assert webui._allowed("POST", "/jobs")
    assert webui._allowed("POST", "/queue/pause")
    assert not webui._allowed("POST", "/repo/delete")
    assert not webui._allowed("GET", "/secrets")


def test_optional_token_gates_the_whole_page(tmp_path):
    """绑 0.0.0.0 给手机用时：首跳 token → Cookie，之后 JS/图片/SSE 自动带。"""
    import http.client

    httpd = webui.serve("127.0.0.1", 0, tmp_path, token="s3cret")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    port = httpd.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", "/")
        r = conn.getresponse()
        assert r.status == 401
        r.read()

        conn.request("GET", "/?token=s3cret")
        r = conn.getresponse()
        assert r.status == 302
        assert r.getheader("Location") == "/"
        cookie = r.getheader("Set-Cookie") or ""
        assert webui.TOKEN_COOKIE in cookie
        r.read()

        conn.request("GET", "/", headers={"Cookie": f"{webui.TOKEN_COOKIE}=s3cret"})
        r = conn.getresponse()
        assert r.status == 200
        assert b"Anima" in r.read()

        conn.request("GET", "/?token=wrong")
        r = conn.getresponse()
        assert r.status == 401
        r.read()
    finally:
        conn.close()
        httpd.shutdown()
        httpd.server_close()


def test_daemon_pidfile_override(tmp_path, monkeypatch):
    override = tmp_path / "custom.json"
    monkeypatch.setenv("ANIMA_DAEMON_PIDFILE", str(override))
    assert webui.daemon_pidfile(Path("E:/whatever")) == override


def test_discover_daemon_returns_empty_when_missing(tmp_path):
    assert webui.discover_daemon(tmp_path) == {}


# --------------------------------------------------------------------------- #
# 契约：反代白名单必须与 anima_daemon 的路由对得上
# --------------------------------------------------------------------------- #
def _daemon_routes(source: str, method: str) -> set[str]:
    """`do_GET` / `do_POST` 里 `path == "…"` 与 `path.startswith("…")` 的字面量。"""
    block = re.search(
        rf"def do_{method}\b(.*?)(?=\n    def |\Z)", source, re.S
    ).group(1)
    return set(re.findall(r'path\s*(?:==|\.startswith\()\s*"([^"]+)"', block))


def test_proxy_whitelist_matches_daemon_routes():
    """白名单里的每条前缀，在守护进程的对应方法里都必须有路由。

    拆仓库时丢掉的正是这条防线：守护进程改了 /events，界面照旧全绿，
    但浏览器里那块面板会静默 404。
    """
    source = (ROOT / "anima_daemon" / "server.py").read_text(encoding="utf-8")
    for method, allow in (("GET", webui.GET_ALLOW), ("POST", webui.POST_ALLOW)):
        routes = _daemon_routes(source, method)
        missing = [
            p for p in allow if not any(r == p or r.startswith(p) for r in routes)
        ]
        assert not missing, (
            f"{method} 白名单里有 anima_daemon 没有的路径：{missing}；"
            f"守护进程该方法的字面路由：{sorted(routes)}"
        )
