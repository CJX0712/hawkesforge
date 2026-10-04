"""流水线编排不变量。"""

from __future__ import annotations

import json

from core.config import Config
from pipeline.hawkes_pipeline import HawkesPipeline


def _cfg():
    c = Config()
    c.n_dims = 3
    c.T = 50.0
    c.beta_grid = (1.0,)
    c.n_seeds = 3
    c.seed = 42
    return c


def test_best_is_flagship():
    cfg = _cfg()
    pipe = HawkesPipeline(cfg)
    rows = pipe.run(regime="coupled")
    agg = pipe.aggregate(rows)
    assert agg["best"] == "hawkesfuse"


def test_deltas_positive():
    cfg = _cfg()
    pipe = HawkesPipeline(cfg)
    rows = pipe.run(regime="coupled")
    agg = pipe.aggregate(rows)
    d = agg["deltas"]
    assert d["hawkesfuse_vs_poisson"] > 0
    assert d["hawkesfuse_vs_unihawkes"] > 0


def test_pipeline_determinism_bitwise():
    cfg = _cfg()
    pipe = HawkesPipeline(cfg)
    a = pipe.run(regime="coupled")
    b = pipe.run(regime="coupled")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
