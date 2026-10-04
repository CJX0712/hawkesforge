"""benchmark 结果文本化（电报式表格）。"""

from __future__ import annotations

from typing import Any

from pipeline.hawkes_pipeline import MODEL_NAMES


def _f(x: Any) -> str:
    if x is None:
        return f"{'NA':>12}"
    return f"{float(x):12.4f}"


def format_report(res: dict[str, Any], title: str = "HawkesForge Benchmark") -> str:
    rows = res["rows"]
    agg = res["aggregate"]
    col_w = 12
    header = f"{'seed':>4} {'n_tr':>5} {'n_te':>5} " + "".join(f"{m:>{col_w}}" for m in MODEL_NAMES)
    lines = [f"# {title}  (regime={res['regime']})", ""]
    lines.append(header)
    for r in rows:
        line = f"{r['seed']:>4} {r['n_train']:>5} {r['n_test']:>5} "
        line += "".join(_f(r.get(m)) for m in MODEL_NAMES)
        lines.append(line)
    lines.append("-" * (len(header) + 4))
    lines.append("mean±std  test NLL/event（越低越好）:")
    for m in MODEL_NAMES:
        pm = agg["per_model"][m]
        tag = "   <- best" if agg["best"] == m else ""
        lines.append(f"  {m:>{col_w}}: {pm['mean']:.4f} ± {pm['std']:.4f}{tag}")
    lines.append("")
    lines.append("增益（flagship − baseline，>0 表示旗舰更优）:")
    for k, v in agg["deltas"].items():
        lines.append(f"  {k:>28}: {v:+.4f}")
    return "\n".join(lines)
