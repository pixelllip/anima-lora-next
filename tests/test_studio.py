"""Training-studio logic tests — stdlib only, no GPU / torch / Qt.

    python -m pytest tests/test_studio.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anima_daemon import studio  # noqa: E402


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Point every studio tree at a temp dir (the real engine is untouched)."""
    monkeypatch.setattr(studio, "PROJECTS_DIR", tmp_path / "training_projects")
    monkeypatch.setattr(studio, "IMAGE_DATASET_DIR", tmp_path / "image_dataset")
    monkeypatch.setattr(studio, "RESIZED_DIR", tmp_path / "post_image_dataset" / "resized")
    monkeypatch.setattr(studio, "CACHE_DIR", tmp_path / "post_image_dataset" / "lora")
    monkeypatch.setattr(studio, "MASK_DIR", tmp_path / "post_image_dataset" / "masks")
    monkeypatch.setattr(studio, "OUTPUT_DIR", tmp_path / "output" / "ckpt")
    return tmp_path


def _image(path: Path, payload: bytes = b"\x89PNG\r\n\x1a\nfake") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class FakeJob:
    def __init__(self, job_id: str) -> None:
        self.id = job_id
        self.state = "queued"


class FakeManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def submit_command(self, **kwargs):
        self.calls.append(("submit_command", kwargs))
        return FakeJob(f"cmd-{len(self.calls)}")

    def submit(self, **kwargs):
        self.calls.append(("submit", kwargs))
        return FakeJob(f"train-{len(self.calls)}")

    def get(self, job_id: str):
        class Record:
            def public(self_inner):
                return {"id": job_id, "state": "queued"}

        return Record()


def test_project_import_review_caption_and_trigger(sandbox):
    project = studio.create_project({"name": "Shinku Style", "trigger": "@shinku"})
    assert project["slug"] == "shinku_style"

    src = _image(sandbox / "source" / "a.png")
    src.with_suffix(".txt").write_text("1girl, solo\n", encoding="utf-8")
    result = studio.import_images("shinku_style", {"paths": [str(src)]})
    assert result["added"] == ["a.png"]
    assert result["counts"]["inbox"] == 1

    inbox = studio.list_images("shinku_style", "inbox")
    assert inbox["images"][0]["caption"].strip() == "1girl, solo"

    moved = studio.review_images(
        "shinku_style", {"action": "accept", "files": ["inbox/a.png"]}
    )
    assert moved["counts"]["dataset"] == 1
    assert studio.list_images("shinku_style", "inbox")["count"] == 0

    studio.set_caption(
        "shinku_style", {"path": "dataset/a.png", "caption": "1girl, solo, smile"}
    )
    assert studio.get_caption("shinku_style", "dataset/a.png")["caption"].strip().endswith("smile")

    applied = studio.apply_trigger("shinku_style", {})
    assert applied["changed"] == 1
    assert studio.get_caption("shinku_style", "dataset/a.png")["caption"].startswith("@shinku, ")
    # Idempotent: a second run changes nothing.
    assert studio.apply_trigger("shinku_style", {})["changed"] == 0


def test_reject_purges_derived_files_so_it_cannot_train(sandbox):
    studio.create_project({"name": "p1"})
    studio.import_images("p1", {"paths": [_image(sandbox / "src" / "b.jpg")]})
    studio.review_images("p1", {"action": "accept", "files": ["inbox/b.jpg"]})

    resized = studio.RESIZED_DIR / "p1" / "b.png"
    latent = studio.CACHE_DIR / "p1" / "b_512x512_anima.npz"
    neighbour = studio.CACHE_DIR / "p1" / "b2_512x512_anima.npz"
    for path in (resized, latent, neighbour):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    studio.review_images("p1", {"action": "reject", "files": ["dataset/b.jpg"]})
    assert not resized.exists(), "rejected image must not stay in the resized tree"
    assert not latent.exists(), "rejected image must not keep a latent cache"
    assert neighbour.exists(), "`b` 的清理不许误删 `b2` 的缓存"
    assert (studio.rejected_dir("p1") / "b.jpg").is_file()
    assert studio.list_images("p1", "dataset")["count"] == 0


def test_build_config_scopes_every_path_to_the_project(sandbox):
    studio.create_project({"name": "scoped one", "variant": "lora", "preset": "low_vram"})
    studio.import_images("scoped_one", {"paths": [_image(sandbox / "src" / "c.png")]})
    studio.review_images("scoped_one", {"action": "accept", "files": ["inbox/c.png"]})

    built = studio.build_config("scoped_one", force=True)
    snapshot = built["snapshot"]
    assert snapshot["source_image_dir"] == "image_dataset/scoped_one"
    assert snapshot["resized_image_dir"].endswith("/scoped_one")
    assert snapshot["lora_cache_dir"].endswith("/scoped_one")
    assert snapshot["output_dir"].endswith("/scoped_one")
    subset = snapshot["datasets"][0]["subsets"][0]
    assert subset["image_dir"].endswith("/scoped_one")
    assert subset["cache_dir"].endswith("/scoped_one")
    # The file is real TOML on disk and matches the returned snapshot.
    assert Path(built["path"]).is_file()


def test_stage_jobs_reuse_one_generated_config_and_can_chain(sandbox):
    studio.create_project({"name": "pipeline"})
    studio.import_images("pipeline", {"paths": [_image(sandbox / "src" / "d.png")]})
    studio.review_images("pipeline", {"action": "accept", "files": ["inbox/d.png"]})
    manager = FakeManager()

    tagged = studio.enqueue_tag("pipeline", manager, {"mode": "missing"})
    assert tagged["stage"] == "tag"
    tag_call = manager.calls[-1][1]
    assert tag_call["label"] == "studio:pipeline:tag"
    assert tag_call["argv"][1:4] == ["caption-autotag", "--apply", "--mode"]
    assert tag_call["config_file"].endswith("train_config.toml")

    pre = studio.enqueue_preprocess("pipeline", manager, {"chain_train": True})
    assert pre["chain_train"] is True
    pre_call = manager.calls[-1][1]
    assert pre_call["chain_train"]["config_file"].endswith("train_config.toml")
    assert "preprocess-config" in pre_call["argv"]
    assert "pipeline" in " ".join(pre_call["argv"])

    # Pretend preprocessing finished: one latent per image.
    cache = studio.CACHE_DIR / "pipeline"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "d_512x512_anima.npz").write_bytes(b"x")
    (cache / "d_anima_te.safetensors").write_bytes(b"x")
    (cache / "d_anima_pe_spatial.safetensors").write_bytes(b"x")
    resized = studio.RESIZED_DIR / "pipeline"
    resized.mkdir(parents=True, exist_ok=True)
    (resized / "d.png").write_bytes(b"x")
    trained = studio.enqueue_train("pipeline", manager, {"overrides": {"max_train_epochs": 8}})
    assert trained["stage"] == "train"
    train_call = manager.calls[-1][1]
    assert train_call["config_file"].endswith("train_config.toml")
    assert train_call["method"] == "lora"

    state = studio.status("pipeline", manager)
    assert state["latestJobs"]["tag"]["state"] == "queued"
    assert state["stages"]["preprocess"] is True


def test_import_rejects_non_images_and_reports_errors(sandbox):
    studio.create_project({"name": "guarded"})
    bogus = sandbox / "src" / "note.txt"
    bogus.parent.mkdir(parents=True)
    bogus.write_text("hello", encoding="utf-8")
    result = studio.import_images("guarded", {"paths": [str(bogus), str(sandbox / "nope")]})
    assert result["added"] == []
    assert result["skipped"]
    assert result["errors"]


def test_studio_tools_are_in_the_mcp_manifest():
    from anima_daemon.server import TOOLS

    names = {t["name"] for t in TOOLS}
    expected = {
        "studio_projects",
        "studio_create_project",
        "studio_import",
        "studio_images",
        "studio_review",
        "studio_caption",
        "studio_set_caption",
        "studio_apply_trigger",
        "studio_tag",
        "studio_preprocess",
        "studio_train",
        "studio_status",
    }
    assert expected <= names
    # Every entry needs a dispatchable HTTP shape.
    for tool in TOOLS:
        assert tool["method"] in {"GET", "POST"}
        assert tool["path"].startswith("/")
        assert isinstance(tool["input_schema"], dict)


def test_mcp_bridge_fills_path_and_query_arguments():
    from anima_daemon import mcp

    class FakeClient:
        def __init__(self):
            self.calls = []

        def health(self):
            return {"ok": True}

        def _request(self, method, path, body=None):
            self.calls.append((method, path, body))
            return {"ok": True}

    client = FakeClient()
    bridge = mcp.MCPServer(client_factory=lambda: client)
    tools = {t["name"]: t for t in bridge._tools}

    bridge._invoke("studio_images", tools["studio_images"], {"slug": "demo", "state": "inbox", "limit": 10})
    assert client.calls[-1][1] == "/studio/projects/demo/images?state=inbox&limit=10"

    bridge._invoke("studio_caption", tools["studio_caption"], {"slug": "demo", "path": "dataset/a.jpg"})
    assert client.calls[-1][1].startswith("/studio/projects/demo/caption?path=dataset%2Fa.jpg")

    bridge._invoke("studio_project", tools["studio_project"], {"slug": "demo"})
    assert client.calls[-1][1] == "/studio/projects/demo"


def test_daemon_studio_routes_serve_the_whole_flow(sandbox):
    """HTTP-level smoke: create → import → review → list → caption → status."""
    import threading
    import urllib.request

    from anima_daemon import server as daemon_server

    httpd = daemon_server.serve(FakeManager(), port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())

    try:
        status, created = call("POST", "/studio/projects", {"name": "http demo"})
        assert status == 200 and created["slug"] == "http_demo"

        src = _image(sandbox / "src" / "flow.png")
        status, imported = call("POST", "/studio/projects/http_demo/import", {"paths": [str(src)]})
        assert status == 200 and imported["added"] == ["flow.png"]

        status, listed = call("GET", "/studio/projects/http_demo/images?state=inbox")
        assert status == 200 and listed["count"] == 1
        rel = listed["images"][0]["path"]

        # The image URL the WebUI puts in <img src> must stream the real bytes.
        with urllib.request.urlopen(base + listed["images"][0]["url"], timeout=10) as resp:
            assert resp.status == 200
            assert resp.read() == src.read_bytes()

        status, reviewed = call(
            "POST", "/studio/projects/http_demo/review", {"action": "accept", "files": [rel]}
        )
        assert status == 200 and reviewed["counts"]["dataset"] == 1

        status, captioned = call(
            "POST", "/studio/projects/http_demo/caption",
            {"path": "dataset/flow.png", "caption": "1girl, solo"},
        )
        assert status == 200 and "1girl" in captioned["caption"]

        status, detail = call("GET", "/studio/projects/http_demo/status")
        assert status == 200 and detail["counts"]["captions"] == 1
        assert detail["stages"]["review"] is True

        # Parameter tab: schema + effective config + override save.
        status, schema = call("GET", "/studio/schema")
        assert status == 200 and any(v["value"] == "lora" for v in schema["variants"])
        status, cfg = call("GET", "/studio/projects/http_demo/config")
        assert status == 200 and any(f["key"] == "network_dim" for f in cfg["fields"])
        status, saved = call(
            "POST", "/studio/projects/http_demo/config", {"overrides": {"network_dim": 8}}
        )
        assert status == 200 and saved["overrides"]["network_dim"] == 8
        status, preview = call("GET", "/studio/projects/http_demo/config?variant=tlora")
        assert status == 200 and preview["variant"] == "tlora" and preview["preview"] is True

        # Krea-2 tab: its own TrainRequest table.
        status, krea = call("GET", "/studio/projects/http_demo/krea2")
        assert status == 200 and krea["values"]["rank"] == 16
        status, krea_saved = call(
            "POST", "/studio/projects/http_demo/krea2", {"values": {"rank": 8}}
        )
        assert status == 200 and krea_saved["values"]["rank"] == 8
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_webui_proxy_whitelist_covers_studio():
    from anima_lora_webui import server as webui

    assert webui._allowed("GET", "/studio/projects")
    assert webui._allowed("GET", "/studio/projects/x/images")
    assert webui._allowed("POST", "/studio/projects/x/review")
    assert webui._allowed("GET", "/studio/projects/x/config")
    assert webui._allowed("POST", "/studio/projects/x/krea2/train")
    assert not webui._allowed("DELETE", "/studio/projects/x")


def test_project_config_exposes_the_gui_form_and_roundtrips_overrides(sandbox):
    studio.create_project({"name": "cfg demo"})
    cfg = studio.project_config("cfg_demo")
    assert cfg["variant"] == "lora"
    assert cfg["preset"] == "low_vram"
    by_key = {f["key"]: f for f in cfg["fields"]}
    assert by_key["network_dim"]["type"] == "int"
    assert by_key["network_dim"]["group"] == "Architecture"
    assert by_key["network_dim"]["basic"] is True
    assert by_key["network_dim"]["help"], "Qt GUI help text must come along"
    assert by_key["learning_rate"]["type"] == "float"
    assert "path_scope" not in by_key, "GUI-only path_scope must stay hidden"

    saved = studio.save_project_config(
        "cfg_demo",
        {
            "overrides": {
                "network_dim": "8",
                "learning_rate": "1e-4",
                "blocks_to_swap": "0",
            }
        },
    )
    assert saved["overrides"] == {
        "network_dim": 8,
        "learning_rate": 1e-4,
        "blocks_to_swap": 0,
    }
    saved_by_key = {f["key"]: f for f in saved["fields"]}
    assert saved_by_key["network_dim"]["value"] == 8
    assert saved_by_key["network_dim"]["overridden"] is True

    # null = drop the override and fall back to the preset value.
    reset = studio.save_project_config("cfg_demo", {"overrides": {"network_dim": None}})
    assert "network_dim" not in reset["overrides"]
    assert {f["key"]: f for f in reset["fields"]}["network_dim"]["overridden"] is False

    with pytest.raises(studio.StudioError):
        studio.save_project_config("cfg_demo", {"overrides": {"typo_key": 1}})
    with pytest.raises(studio.StudioError):
        studio.save_project_config("cfg_demo", {"overrides": {"network_dim": "not-a-number"}})


def test_virtual_validation_keys_are_written_into_the_dataset_block(sandbox):
    studio.create_project({"name": "valid demo"})
    studio.import_images("valid_demo", {"paths": [_image(sandbox / "src" / "v.png")]})
    studio.review_images("valid_demo", {"action": "accept", "files": ["inbox/v.png"]})
    studio.save_project_config(
        "valid_demo", {"overrides": {"use_valid": True, "validation_split_num": 8}}
    )
    built = studio.build_config("valid_demo", force=True)
    first = built["snapshot"]["datasets"][0]
    assert first["validation_split_num"] == 8


def test_krea2_config_is_its_own_table_and_train_submits_the_cli(sandbox):
    studio.create_project({"name": "krea demo"})
    info = studio.project_krea2("krea_demo")
    by_key = {f["key"]: f for f in info["fields"]}
    assert by_key["rank"]["type"] == "int"
    assert by_key["target_preset"]["choices"] == ["official_gate", "official", "attention"]
    assert info["values"]["output"].endswith("output/ckpt_krea2/krea_demo.safetensors")

    saved = studio.save_project_krea2(
        "krea_demo",
        {
            "values": {
                "rank": "8",
                "alpha": "8.0",
                "stack_lora_mode": "pinned",
                "stack_lora_paths": "a.safetensors;b.safetensors",
                "stack_lora_strengths": "1.0;1.25",
                "stack_lora_warmup_steps": "0",
            }
        },
    )
    assert saved["exists"] is True
    assert saved["values"]["rank"] == 8
    assert "rank = 8" in saved["text"]

    with pytest.raises(studio.StudioError, match="strengths"):
        studio.save_project_krea2(
            "krea_demo",
            {
                "values": {
                    "stack_lora_mode": "pinned",
                    "stack_lora_paths": "a;b",
                    "stack_lora_strengths": "1.0",
                }
            },
        )

    manager = FakeManager()
    with pytest.raises(studio.StudioError, match="缓存目录"):
        studio.enqueue_krea2_train("krea_demo", manager, {})
    assert not manager.calls, "缺缓存时不许入队"

    cache = sandbox / "cache" / "krea"
    cache.mkdir(parents=True)
    (cache / "a_anima.npz").write_bytes(b"x")
    model = sandbox / "model.safetensors"
    model.write_bytes(b"x")
    studio.save_project_krea2(
        "krea_demo",
        {"values": {"cache": str(cache), "model": str(model), "output": str(sandbox / "out.safetensors")}},
    )
    job = studio.enqueue_krea2_train("krea_demo", manager, {})
    assert job["stage"] == "krea2"
    call = manager.calls[-1][1]
    assert call["argv"][0] == "scripts/krea2/train.py"
    assert call["argv"][1] == "--config"
    assert call["extra_env"]["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync"
