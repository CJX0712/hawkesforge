"""模型接口契约（Protocol）。

任何强度模型都必须实现 fit / intensity / nll / available，
保证 pipeline 与评测层可以统一对待基线、旗舰与离线兜底。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from core.types import EventSeq, FitResult


@runtime_checkable
class IntensityModel(Protocol):
    """强度模型契约。"""

    name: str

    def available(self) -> bool:
        """后端是否可用（纯 numpy 永远 True）。"""
        ...

    def fit(self, train: EventSeq) -> FitResult:
        """在训练序列上拟合参数。"""
        ...

    def nll(self, params: FitResult, seq: EventSeq) -> float:
        """用已拟合参数计算 seq 的每事件负对数似然（越低越好）。"""
        ...

    def intensity_at(self, params: FitResult, t: float, seq: EventSeq) -> np.ndarray:
        """给定历史 seq（截至 < t），返回各维强度向量。"""
        ...
