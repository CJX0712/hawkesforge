"""HawkesForge CLI 入口（argparse）。"""

from __future__ import annotations

import argparse
import json

from core.config import Config
from pipeline.hawkes_pipeline import MODEL_NAMES, HawkesPipeline
from pipeline.report import format_report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="hawkesforge", description="HawkesForge — 时间点过程 / Hawkes 自激发事件建模"
    )
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("bench", help="运行 benchmark 并打印 + 落盘 JSON")
    b.add_argument("--seeds", type=int, default=None, help="覆盖 n_seeds")
    b.add_argument("--regime", default="coupled", choices=["coupled", "poisson"])
    b.add_argument("--out", default="benchmark.json")

    sub.add_parser("models", help="列出参与评测的模型")

    args = p.parse_args(argv)
    cfg = Config.from_env()
    if getattr(args, "seeds", None):
        cfg.n_seeds = args.seeds

    if args.cmd == "models":
        for m in MODEL_NAMES:
            print(m)
        return 0

    pipe = HawkesPipeline(cfg)
    res = pipe.benchmark(regime=getattr(args, "regime", "coupled"))
    print(format_report(res))
    out = args.out if hasattr(args, "out") else "benchmark.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_jsonable(res), f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {out}")
    return 0


def _jsonable(res: dict) -> dict:
    return res


if __name__ == "__main__":
    raise SystemExit(main())
