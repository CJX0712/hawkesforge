"""Hawkes 似然与解析梯度（共享全局衰减率 beta）。

所有算法都建在同一数学内核上，保证跨模型公平评测：
    λ_i(t) = μ_i + β · Σ_j α_{i,j} · G_j(t)
    G_j(t) = Σ_{t_j^m < t} exp(−β (t − t_j^m))

log-likelihood（Ogata 形式）与梯度均为闭式，单次前向 O(N·d)。
"""

from __future__ import annotations

import numpy as np

from core.config import Config
from core.errors import FitError
from core.types import EventSeq


def log_likelihood_and_grad(
    mu: np.ndarray, alpha: np.ndarray, beta: float, seq: EventSeq
) -> tuple[float, np.ndarray, np.ndarray]:
    """返回 (ll, grad_mu (d,), grad_alpha (d,d))。

    对事件序列做一次 O(N·d) 扫描，同时累加对数似然与解析梯度。
    """
    times = seq.times
    dims = seq.dims
    T = seq.T
    N = int(times.size)
    d = int(mu.size)
    G = np.zeros(d)  # 各源维度的尾部和
    integ = np.zeros(d)  # 积分项（每源维度）
    prev_t = 0.0
    ll = 0.0
    gmu = np.zeros(d)
    galpha = np.zeros((d, d))

    for k in range(N):
        t = float(times[k])
        c = int(dims[k])
        dt = t - prev_t
        if dt > 0.0:
            G = G * np.exp(-beta * dt)
        lam = mu[c] + beta * float(np.dot(alpha[c], G))
        if not np.isfinite(lam) or lam <= 0.0:
            lam = 1e12  # 防御：超临界数据下强度溢出时夹到安全大值，避免 log 出 NaN/inf
        ll += np.log(lam)
        gmu[c] += 1.0 / lam
        galpha[c, :] += (beta * G) / lam
        integ[c] += 1.0 - np.exp(-beta * (T - t))
        G[c] += 1.0
        prev_t = t

    integral = float(np.sum(mu)) * T + float(np.sum(alpha * integ[None, :]))
    ll -= integral
    gmu -= T
    galpha -= integ[None, :]
    return float(ll), gmu, galpha


def spectral_radius(alpha: np.ndarray, beta: float) -> float:
    from numpy.linalg import eigvals

    w = eigvals(alpha / beta)
    return float(np.max(np.abs(w)))


def branching_ratio(alpha: np.ndarray, beta: float) -> float:
    return float(np.max(np.sum(alpha, axis=1) / beta))


def _project_spectral(alpha: np.ndarray, beta: float, cap: float) -> np.ndarray:
    """谱半径超 cap 时整体等比缩放至 cap*0.99，保证过程稳定（HawkesFuse 守卫）。"""
    rho = spectral_radius(alpha, beta)
    if rho > cap and rho > 0.0:
        alpha = alpha * ((cap * 0.99) / rho)
    return alpha


def fit_mle(
    mu0: np.ndarray,
    alpha0: np.ndarray,
    beta: float,
    seq: EventSeq,
    cfg: Config,
    allow_offdiag: bool = True,
    l2_offdiag: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """解析梯度 + Adam 极大似然估计。

    返回 (mu, alpha, ll)。非负投影；HawkesFuse 额外做跨维 L2 正则 + 谱投影。
    """
    d = int(mu0.size)
    mu = mu0.astype(float).copy()
    alpha = alpha0.astype(float).copy()
    if not allow_offdiag:
        alpha = np.diag(np.diag(alpha))

    b1, b2, eps = 0.9, 0.999, 1e-8
    m_mu = np.zeros(d)
    v_mu = np.zeros(d)
    m_a = np.zeros((d, d))
    v_a = np.zeros((d, d))
    mask_off = ~np.eye(d, dtype=bool)
    prev_ll = -np.inf

    for it in range(1, cfg.max_iter + 1):
        ll, gmu, galpha = log_likelihood_and_grad(mu, alpha, beta, seq)
        if not np.isfinite(ll):
            raise FitError(f"非有限似然 @iter {it}")
        if l2_offdiag > 0.0:
            galpha[mask_off] -= 2.0 * l2_offdiag * alpha[mask_off]

        m_mu = b1 * m_mu + (1 - b1) * gmu
        v_mu = b2 * v_mu + (1 - b2) * (gmu**2)
        m_a = b1 * m_a + (1 - b1) * galpha
        v_a = b2 * v_a + (1 - b2) * (galpha**2)
        mhat = m_mu / (1 - b1**it)
        vhat = v_mu / (1 - b2**it)
        mu = mu + cfg.lr * mhat / (np.sqrt(vhat) + eps)
        mhat_a = m_a / (1 - b1**it)
        vhat_a = v_a / (1 - b2**it)
        alpha = alpha + cfg.lr * mhat_a / (np.sqrt(vhat_a) + eps)

        mu = np.clip(mu, cfg.min_intensity, None)
        if not allow_offdiag:
            alpha = np.diag(np.diag(alpha))
        alpha = np.clip(alpha, 0.0, None)
        if allow_offdiag and l2_offdiag >= 0.0:
            alpha = _project_spectral(alpha, beta, cfg.spectral_cap)

        if it > 30 and abs(ll - prev_ll) < cfg.tol * abs(ll):
            break
        prev_ll = ll

    ll, _, _ = log_likelihood_and_grad(mu, alpha, beta, seq)
    if not np.isfinite(ll):
        raise FitError("拟合发散")
    return mu, alpha, float(ll)
