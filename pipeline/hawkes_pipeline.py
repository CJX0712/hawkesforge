"""流水线：数据生成 → 多模型拟合 → 测试集 NLL 评测。

单向无环：pipeline → {data, hawkes, eval} → core。
每个 seed 独立生成 train/test（同参数独立采样，零泄漏）。
"""

from __future__ import annotations

from typing import Any

from core.config import Config
from core.seed import set_all
from data.synth import generate_test, generate_train
from eval.metrics import rank_by_nll, summarize
from hawkes import all_models

MODEL_NAMES = ["poisson", "unihawkes", "hawkesfuse"]


class HawkesPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, regime: str = "coupled") -> list[dict]:
        set_all(self.cfg.seed)
        rows: list[dict] = []
        for s in range(self.cfg.n_seeds):
            seed = self.cfg.seed * 1000 + s
            params, train = generate_train(self.cfg, seed, regime=regime)
            test = generate_test(params, self.cfg, seed + 500003)
            row: dict[str, Any] = {
                "seed": s,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "true_spectral": float(params.spectral_radius()),
            }
            for m in all_models(self.cfg):
                try:
                    fr = m.fit(train)
                    nll = m.nll(fr, test)
                    row[m.name] = float(nll)
                    row[m.name + "_sr"] = float(fr.spectral_radius)
                    row[m.name + "_conv"] = bool(fr.converged)
                    row[m.name + "_beta"] = float(fr.beta)
                except Exception as e:  # noqa: BLE001
                    row[m.name] = None
                    row[m.name + "_err"] = repr(e)
            rows.append(row)
        return rows

    def aggregate(self, rows: list[dict]) -> dict:
        return {
            "per_model": summarize(rows, MODEL_NAMES),
            "best": rank_by_nll(rows, MODEL_NAMES)[0],
            "deltas": self._deltas(rows),
        }

    @staticmethod
    def _deltas(rows: list[dict]) -> dict:
        def mean(m: str) -> float:
            import numpy as np

            col = [r[m] for r in rows if r.get(m) is not None]
            return float(np.mean(col))

        base = mean("poisson")
        uni = mean("unihawkes")
        fuse = mean("hawkesfuse")
        return {
            "hawkesfuse_vs_poisson": base - fuse,
            "hawkesfuse_vs_unihawkes": uni - fuse,
        }

    def benchmark(self, regime: str = "coupled") -> dict:
        rows = self.run(regime=regime)
        agg = self.aggregate(rows)
        return {"regime": regime, "rows": rows, "aggregate": agg}


def core_metrics(rows: list[dict]) -> list[float]:
    """用于确定性逐位比对的核心指标序列。"""
    out: list[float] = []
    for r in rows:
        for m in MODEL_NAMES:
            v = r.get(m)
            out.append(v if v is None else float(v))
    return out
