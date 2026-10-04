"""基线 A：齐次泊松过程（每维恒定强度）。解析 MLE，闭式解。"""

from __future__ import annotations

import numpy as np

from core.config import Config
from core.types import EventSeq, FitResult


class PoissonModel:
    name = "poisson"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def available(self) -> bool:
        return True

    def fit(self, train: EventSeq) -> FitResult:
        d = train.d
        T = train.T
        counts = train.counts()
        mu = (counts / T).clip(min=self.cfg.min_intensity)
        ll = float(np.sum(counts * np.log(mu) - mu * T))
        n = max(1, len(train))
        return FitResult(
            method="poisson",
            mu=mu,
            alpha=np.zeros((d, d)),
            beta=0.0,
            ll=ll,
            nll_per_event=-ll / n,
            converged=True,
            spectral_radius=0.0,
            branching_ratio=0.0,
            notes="齐次泊松基线（无激发结构）",
        )

    def nll(self, params: FitResult, seq: EventSeq) -> float:
        counts = seq.counts()
        T = seq.T
        mu = params.mu.clip(min=self.cfg.min_intensity)
        ll = float(np.sum(counts * np.log(mu) - mu * T))
        return -ll / max(1, len(seq))

    def intensity_at(self, params: FitResult, t: float, seq: EventSeq) -> np.ndarray:
        return params.mu.copy()
