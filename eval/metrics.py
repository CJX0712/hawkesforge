"""评测指标聚合。所有 NLL 越低越好。"""

from __future__ import annotations

import numpy as np


def mean_std(vals) -> tuple[float, float]:
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def summarize(rows: list[dict], model_names: list[str]) -> dict:
    out = {}
    for m in model_names:
        col = [r.get(m) for r in rows]
        mean, std = mean_std(col)
        out[m] = {"mean": mean, "std": std}
    return out


def rank_by_nll(rows: list[dict], model_names: list[str]) -> list[str]:
    avg = {m: float(np.nanmean([r.get(m) for r in rows])) for m in model_names}
    return sorted(model_names, key=lambda m: avg[m])
