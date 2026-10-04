"""核心数据类型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EventSeq:
    """一段事件序列：时间戳（升序）+ 对应维度。

    Attributes
    ----------
    times : (N,) float，严格升序
    dims  : (N,) int，取值 [0, d)
    d     : 维度数
    T     : 观测窗右端（>= max(times)）
    """

    times: np.ndarray
    dims: np.ndarray
    d: int
    T: float = 0.0

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.dims = np.asarray(self.dims, dtype=int)
        if self.times.size:
            self.dims = self.dims.astype(int)
            mx = float(self.times.max())
            if mx > self.T or self.T <= 0:
                self.T = mx
        else:
            self.T = float(self.T) if self.T > 0 else 0.0

    def counts(self) -> np.ndarray:
        return np.bincount(self.dims, minlength=self.d).astype(float)

    def __len__(self) -> int:
        return int(self.times.size)


@dataclass
class FitResult:
    """一个模型在训练集上拟合出的参数与诊断信息。"""

    method: str
    mu: np.ndarray  # (d,) 基线强度
    alpha: np.ndarray  # (d,d) alpha[i,j] = 维度 j 对 i 的激发系数
    beta: float  # 全局指数衰减率（共享）
    ll: float  # 训练集对数似然
    nll_per_event: float  # -ll / N_train（越低越好）
    converged: bool
    spectral_radius: float  # alpha/beta 的谱半径（<1 稳定）
    branching_ratio: float  # 最大分支比（<1 稳定）
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "ll": self.ll,
            "nll_per_event": self.nll_per_event,
            "converged": self.converged,
            "spectral_radius": self.spectral_radius,
            "branching_ratio": self.branching_ratio,
            "notes": self.notes,
        }


@dataclass
class HawkesParams:
    """生成数据用的真实/假定参数（ground truth）。"""

    mu: np.ndarray  # (d,)
    alpha: np.ndarray  # (d,d)
    beta: float

    def spectral_radius(self) -> float:
        from numpy.linalg import eigvals

        w = eigvals(self.alpha / self.beta)
        return float(np.max(np.abs(w)))
