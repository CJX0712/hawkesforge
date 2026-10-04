"""数据模块：合成 Hawkes / Poisson 事件序列生成。"""

from __future__ import annotations

from data.synth import (
    generate_test,
    generate_train,
    make_params,
)

__all__ = ["generate_train", "generate_test", "make_params"]
