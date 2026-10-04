"""全局确定性：唯一 seed 入口。

所有随机源（numpy legacy / random / Generator）一次性设齐，
保证 demo 两次运行 benchmark 核心指标逐位一致。
"""

from __future__ import annotations

import random

import numpy as np

_CURRENT_SEED: int | None = None


def set_all(seed: int) -> np.random.Generator:
    """设置全局确定性种子，返回可复现的 numpy Generator。"""
    global _CURRENT_SEED
    _CURRENT_SEED = int(seed)
    np.random.seed(_CURRENT_SEED)
    random.seed(_CURRENT_SEED)
    return np.random.default_rng(_CURRENT_SEED)


def get_seed() -> int:
    return _CURRENT_SEED if _CURRENT_SEED is not None else 42


def rng(seed: int | None = None) -> np.random.Generator:
    """取得（可选覆盖的）确定性 Generator。"""
    return np.random.default_rng(seed if seed is not None else get_seed())
