"""Krea-2 line 训练 CLI。

    python scripts/krea2/train.py --config configs/krea2/lora.toml
    python scripts/krea2/train.py --config configs/krea2/lora.toml --max_steps 20 --rank 16

优先级：`TrainRequest` 字段默认值 → `--config` 的 TOML → 命令行。
（与 Anima 的 base→preset→method 链无关；那条链是 Anima 专属的。）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from library.krea2.requests import TrainRequest  # noqa: E402
from library.krea2.train import run_train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config = None
    if "--config" in argv:
        i = argv.index("--config")
        config = argv[i + 1]
        del argv[i : i + 2]
    elif any(a.startswith("--config=") for a in argv):
        a = next(a for a in argv if a.startswith("--config="))
        config = a.split("=", 1)[1]
        argv.remove(a)

    req = TrainRequest.load_config(config) if config else TrainRequest()
    req.apply_cli(argv)
    print(f"[krea2] config={config or '(默认值)'} → {req.to_dict()}")
    return run_train(req)


if __name__ == "__main__":
    raise SystemExit(main())
