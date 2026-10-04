"""合成数据生成：Ogata 稀疏采样法精确生成多元 Hawkes 自激发序列。

生成器严格遵循与拟合内核一致的强度公式，保证「无泄漏」——评测用的
测试序列由同一组参数独立采样得到，训练/测试完全隔离。
"""

from __future__ import annotations

import numpy as np

from core.seed import rng
from core.types import EventSeq, HawkesParams


def make_params(cfg, rand: np.random.Generator) -> HawkesParams:
    """构造 ground-truth 参数：对角自激发 + 适度跨维激发（谱半径 < 1）。"""
    d = cfg.n_dims
    mu = cfg.mu_scale * (0.7 + 0.6 * rand.random(d))
    alpha = np.zeros((d, d))
    np.fill_diagonal(alpha, cfg.alpha_diag)
    if d > 1:
        mask = ~np.eye(d, dtype=bool)
        alpha[mask] = cfg.alpha_off
    return HawkesParams(mu=mu, alpha=alpha, beta=cfg.beta_true)


def _generate_hawkes(
    params: HawkesParams, T: float, rand: np.random.Generator, max_events: int = 20_000
) -> EventSeq:
    mu = params.mu
    alpha = params.alpha
    beta = params.beta
    d = int(len(mu))
    t = 0.0
    S = np.zeros((d, d))
    times: list[float] = []
    dims: list[int] = []
    while True:
        # 当前各维强度
        lam = mu + np.array([np.sum(alpha[i, :] * beta * S[i, :]) for i in range(d)])
        total = float(lam.sum())
        if total <= 1e-12:
            break
        tau = t + rand.exponential(1.0 / total)
        if tau > T:
            break
        S = S * np.exp(-beta * (tau - t))
        lam_tau = mu + np.array([np.sum(alpha[i, :] * beta * S[i, :]) for i in range(d)])
        total_tau = float(lam_tau.sum())
        if total_tau <= 1e-12:
            t = tau
            continue
        c = int(rand.choice(d, p=(lam_tau / total_tau)))
        times.append(tau)
        dims.append(c)
        S[:, c] += 1.0
        t = tau
        if len(times) > max_events:
            break
    times_arr = np.asarray(times, dtype=float)
    dims_arr = np.asarray(dims, dtype=int)
    # 保序（tau 单调递增，正常已有序；防御性排序）
    order = np.argsort(times_arr, kind="stable")
    return EventSeq(times=times_arr[order], dims=dims_arr[order], d=d, T=T)


def _generate_poisson(mu: np.ndarray, T: float, rand: np.random.Generator) -> EventSeq:
    d = int(len(mu))
    times: list[float] = []
    dims: list[int] = []
    for i in range(d):
        if mu[i] <= 0:
            continue
        t = 0.0
        while True:
            t += rand.exponential(1.0 / mu[i])
            if t > T:
                break
            times.append(t)
            dims.append(i)
    times_arr = np.asarray(times, dtype=float)
    dims_arr = np.asarray(dims, dtype=int)
    order = np.argsort(times_arr, kind="stable")
    return EventSeq(times=times_arr[order], dims=dims_arr[order], d=d, T=T)


def generate_train(cfg, seed: int, regime: str = "coupled") -> tuple[HawkesParams, EventSeq]:
    rand = rng(seed)
    params = make_params(cfg, rand)
    if regime == "poisson":
        params = HawkesParams(mu=params.mu, alpha=np.zeros_like(params.alpha), beta=params.beta)
        seq = _generate_poisson(params.mu, cfg.T, rand)
    else:
        seq = _generate_hawkes(params, cfg.T, rand, max_events=cfg.max_events)
    return params, seq


def generate_test(params: HawkesParams, cfg, seed: int) -> EventSeq:
    rand = rng(seed)
    if np.allclose(params.alpha, 0.0):
        return _generate_poisson(params.mu, cfg.T, rand)
    return _generate_hawkes(params, cfg.T, rand, max_events=cfg.max_events)
