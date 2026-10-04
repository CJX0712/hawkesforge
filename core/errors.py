"""错误码与异常层级（E100~E500）。"""

from __future__ import annotations


class HawkesError(Exception):
    """所有 HawkesForge 异常的基类。"""


class ConfigError(HawkesError):
    """E100 配置 / schema 校验失败。"""


class DataError(HawkesError):
    """E200 数据生成 / 载入失败。"""


class FitError(HawkesError):
    """E300 模型拟合失败（发散 / NaN）。"""


class EvalError(HawkesError):
    """E400 评测失败。"""


class PipelineError(HawkesError):
    """E500 流水线编排失败。"""


CODES = {
    "E100": "config/schema",
    "E200": "data generation",
    "E300": "model fit",
    "E400": "evaluation",
    "E500": "pipeline",
}
