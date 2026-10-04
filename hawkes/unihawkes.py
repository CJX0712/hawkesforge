"""基线 B：逐维 univariate Hawkes（仅对角自激发，忽略跨维耦合）。"""

from __future__ import annotations

from core.config import Config
from hawkes.base import HawkesMLE


class UniHawkes(HawkesMLE):
    name = "unihawkes"
    allow_offdiag = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
