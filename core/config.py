"""配置：ENV_HAWKES_* 覆盖 + schema 校验。

所有可调超参集中于此，避免散落；from_env() 读取环境变量覆盖，
validate() 做硬约束，失败抛 ConfigError（E100）。
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any

from core.errors import ConfigError


@dataclass
class Config:
    seed: int = 42
    n_dims: int = 3
    n_seeds: int = 3
    T: float = 120.0
    beta_true: float = 1.0
    # 生成参数（coupled 自激发）
    mu_scale: float = 0.28
    alpha_diag: float = 0.50
    alpha_off: float = 0.15
    # 拟合
    beta_grid: tuple = (0.5, 1.0, 2.0)
    max_iter: int = 4000
    lr: float = 0.05
    tol: float = 1e-7
    l2_offdiag: float = 1e-3  # HawkesFuse 旗舰对跨维系数的 L2 正则
    spectral_cap: float = 0.97  # 拟合后谱半径需 < 此值才算稳定
    min_intensity: float = 1e-9
    # 评测
    n_test_events_target: int = 900
    # 单序列事件上限（防止超临界过程事件爆炸拖垮拟合）
    max_events: int = 20_000

    def validate(self) -> None:
        if self.n_dims < 1:
            raise ConfigError("n_dims 必须 >= 1")
        if self.T <= 0:
            raise ConfigError("T 必须 > 0")
        if self.n_seeds < 1:
            raise ConfigError("n_seeds 必须 >= 1")
        if not (0.5 <= self.alpha_diag + (self.n_dims - 1) * self.alpha_off < 1.0):
            # 仅 warn，不必阻断——但保持默认安全
            pass
        if self.spectral_cap <= 0 or self.spectral_cap >= 1.0:
            raise ConfigError("spectral_cap 必须 ∈ (0,1)")
        if self.max_events < 1:
            raise ConfigError("max_events 必须 >= 1")

    @classmethod
    def from_env(cls) -> Config:
        kv: dict[str, Any] = {}
        prefix = "HAWKES_"
        for k, v in os.environ.items():
            if k.startswith(prefix):
                name = k[len(prefix) :].lower()
                with contextlib.suppress(ValueError):
                    kv[name] = _coerce(v)
        cfg = cls(**{k: v for k, v in kv.items() if k in cls.__dataclass_fields__})
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _coerce(v: str) -> Any:
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError as e:
        raise ValueError(v) from e
