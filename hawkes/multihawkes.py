"""基线 C（MultHawkes，无正则）与旗舰 HawkesFuse（跨维 L2 + 谱投影）。"""

from __future__ import annotations

from core.config import Config
from hawkes.base import HawkesMLE


class MultiHawkes(HawkesMLE):
    """完整多元 Hawkes，允许跨维激发系数，但无 L2 正则。"""

    name = "multihawkes"
    allow_offdiag = True
    l2_offdiag = 0.0

    def __init__(self, cfg: Config):
        super().__init__(cfg)


class HawkesFuse(HawkesMLE):
    """旗舰：完整多元 Hawkes + 跨维 L2 正则 + 谱半径投影（HawkesFuse）。

    跨维 L2 抑制过拟合式的不稳定耦合，谱投影保证过程可稳定（分支比 < 1），
    在保持自激发建模能力的同时显著提升对未见序列的泛化（更低 test NLL）。
    """

    name = "hawkesfuse"
    allow_offdiag = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.l2_offdiag = cfg.l2_offdiag
