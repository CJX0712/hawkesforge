"""数学内核：对数似然与解析梯度 vs 朴素双循环 / 数值差分交叉验证。"""

from __future__ import annotations

import numpy as np

from core.types import EventSeq
from hawkes._math import (
    branching_ratio,
    log_likelihood_and_grad,
    spectral_radius,
)


def _naive_ll(mu, alpha, beta, seq):
    """独立实现的 O(N^2) 参考对数似然（与 _math 互不共享代码）。"""
    times = seq.times
    dims = seq.dims
    T = seq.T
    d = len(mu)
    ll = 0.0
    for k in range(len(times)):
        c = int(dims[k])
        t = times[k]
        lam = mu[c]
        for j in range(d):
            s = 0.0
            for m in range(k):
                if int(dims[m]) == j:
                    s += np.exp(-beta * (t - times[m]))
            lam += alpha[c, j] * beta * s
        ll += np.log(lam)
    for i in range(d):
        integral = mu[i] * T
        for j in range(d):
            for m in range(len(times)):
                if int(dims[m]) == j:
                    integral += alpha[i, j] * (1.0 - np.exp(-beta * (T - times[m])))
        ll -= integral
    return ll


def test_ll_matches_naive_reference():
    rng = np.random.default_rng(3)
    d = 3
    mu = rng.uniform(0.1, 0.5, d)
    alpha = np.abs(rng.standard_normal((d, d))) * 0.3
    np.fill_diagonal(alpha, 0.4)
    beta = 1.0
    t = np.sort(rng.uniform(0, 10, 40))
    dims = rng.integers(0, d, 40)
    seq = EventSeq(times=t, dims=dims, d=d, T=10.0)
    ll, _, _ = log_likelihood_and_grad(mu, alpha, beta, seq)
    assert abs(ll - _naive_ll(mu, alpha, beta, seq)) < 1e-9


def test_grad_matches_finite_diff():
    rng = np.random.default_rng(11)
    d = 3
    mu = rng.uniform(0.1, 0.5, d)
    alpha = np.abs(rng.standard_normal((d, d))) * 0.3
    np.fill_diagonal(alpha, 0.4)
    beta = 1.0
    t = np.sort(rng.uniform(0, 8, 35))
    dims = rng.integers(0, d, 35)
    seq = EventSeq(times=t, dims=dims, d=d, T=8.0)
    ll, gmu, galpha = log_likelihood_and_grad(mu, alpha, beta, seq)
    eps = 1e-6
    for i in range(d):
        mu2 = mu.copy()
        mu2[i] += eps
        mu1 = mu.copy()
        mu1[i] -= eps
        num = (_naive_ll(mu2, alpha, beta, seq) - _naive_ll(mu1, alpha, beta, seq)) / (2 * eps)
        assert abs(num - gmu[i]) < 1e-4, f"mu[{i}] grad {gmu[i]} vs num {num}"
    for i in range(d):
        for j in range(d):
            a2 = alpha.copy()
            a2[i, j] += eps
            a1 = alpha.copy()
            a1[i, j] -= eps
            num = (_naive_ll(mu, a2, beta, seq) - _naive_ll(mu, a1, beta, seq)) / (2 * eps)
            assert abs(num - galpha[i, j]) < 1e-4, (
                f"alpha[{i},{j}] grad {galpha[i, j]} vs num {num}"
            )


def test_spectral_and_branching():
    alpha = np.array([[0.5, 0.15, 0.15], [0.15, 0.5, 0.15], [0.15, 0.15, 0.5]])
    beta = 1.0
    assert abs(spectral_radius(alpha, beta) - 0.8) < 1e-9
    assert abs(branching_ratio(alpha, beta) - 0.8) < 1e-9
