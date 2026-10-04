"""Hawkes MLE 基类：跨 beta 网格搜索 + 闭式梯度拟合。

uni / multi / fuse 三个模型共享同一拟合内核，仅在「是否允许跨维系数」
与「HawkesFuse 的 L2 正则 + 谱投影」上不同，确保评测口径一致。
"""

from __future__ import annotations

import numpy as np

from core.config import Config
from core.errors import FitError
from core.types import EventSeq, FitResult
from hawkes._math import branching_ratio, fit_mle, log_likelihood_and_grad, spectral_radius


class HawkesMLE:
    name = "hawkes"
    allow_offdiag: bool = True
    l2_offdiag: float = 0.0

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def available(self) -> bool:
        return True

    def _init(self, train: EventSeq) -> tuple[np.ndarray, np.ndarray]:
        d = train.d
        counts = train.counts()
        mu0 = (counts / train.T).clip(min=self.cfg.min_intensity)
        a0 = np.zeros((d, d))
        np.fill_diagonal(a0, 0.1)
        if self.allow_offdiag:
            mask = ~np.eye(d, dtype=bool)
            a0[mask] = 0.02
        return mu0, a0

    def fit(self, train: EventSeq) -> FitResult:
        best = None
        for beta in self.cfg.beta_grid:
            mu0, a0 = self._init(train)
            try:
                mu, alpha, ll = fit_mle(
                    mu0,
                    a0,
                    beta,
                    train,
                    self.cfg,
                    allow_offdiag=self.allow_offdiag,
                    l2_offdiag=self.l2_offdiag,
                )
            except FitError:
                continue
            if best is None or ll > best[0]:
                best = (ll, beta, mu, alpha)
        if best is None:
            raise FitError(f"{self.name} 在所有 beta 网格上拟合失败")
        ll, beta, mu, alpha = best
        n = max(1, len(train))
        sr = spectral_radius(alpha, beta)
        br = branching_ratio(alpha, beta)
        converged = sr < self.cfg.spectral_cap
        notes = "" if converged else f"谱半径 {sr:.3f} >= cap {self.cfg.spectral_cap:.2f}（不稳定）"
        return FitResult(
            method=self.name,
            mu=mu,
            alpha=alpha,
            beta=beta,
            ll=ll,
            nll_per_event=-ll / n,
            converged=converged,
            spectral_radius=sr,
            branching_ratio=br,
            notes=notes,
        )

    def nll(self, params: FitResult, seq: EventSeq) -> float:
        ll, _, _ = log_likelihood_and_grad(params.mu, params.alpha, params.beta, seq)
        return -ll / max(1, len(seq))

    def intensity_at(self, params: FitResult, t: float, seq: EventSeq) -> np.ndarray:
        mask = seq.times < t
        tt = seq.times[mask]
        dd = seq.dims[mask]
        G = np.zeros(params.mu.size)
        for tm, dm in zip(tt, dd, strict=False):
            G = G * np.exp(-params.beta * (t - tm))
            G[dm] += 1.0
        return params.mu + params.beta * (params.alpha @ G)
