"""HawkesForge 领域模块：强度模型集合。"""

from __future__ import annotations

from core.config import Config
from hawkes.base import HawkesMLE
from hawkes.multihawkes import HawkesFuse, MultiHawkes
from hawkes.poisson import PoissonModel
from hawkes.unihawkes import UniHawkes

__all__ = [
    "PoissonModel",
    "UniHawkes",
    "MultiHawkes",
    "HawkesFuse",
    "HawkesMLE",
    "all_models",
]


def all_models(cfg: Config):
    """返回评测所用全部模型（基线 + 旗舰），顺序即报告顺序。

    Poisson   : 基线 A —— 无时间结构（齐次泊松）
    UniHawkes : 基线 B / 消融 —— 仅对角自激发，无跨维耦合
    HawkesFuse: 旗舰 —— 完整多元自激发 + β 网格 + 谱半径守卫（SOTA）
    """
    return [
        PoissonModel(cfg),
        UniHawkes(cfg),
        HawkesFuse(cfg),
    ]
