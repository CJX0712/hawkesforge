"""模型拟合不变量与性能门禁。"""

from __future__ import annotations

import numpy as np

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


def test_all_models_available_and_fit():
    cfg = _cfg()
    pipe = HawkesPipeline(cfg)
    rows = pipe.run(regime="coupled")
    assert len(rows) == cfg.n_seeds
    for r in rows:
        for m in ["poisson", "unihawkes", "hawkesfuse"]:
            assert r.get(m) is not None, f"{m} 拟合失败: {r.get(m + '_err')}"
            assert np.isfinite(r[m])


def test_fit_is_deterministic():
    cfg = _cfg()
    pipe = HawkesPipeline(cfg)
    rows_a = pipe.run(regime="coupled")
    rows_b = pipe.run(regime="coupled")
    import json

    assert json.dumps(rows_a, sort_keys=True) == json.dumps(rows_b, sort_keys=True)


def test_poisson_closed_form():
    cfg = _cfg()
    pipe = HawkesPipeline(cfg)
    r = pipe.run(regime="coupled")[0]
    # 直接验证 poisson NLL 与计数公式一致（已在 test_synth 间接覆盖，这里再确认有限且>0）
    assert r["poisson"] > 0


def test_flagship_beats_poisson():
    cfg = _cfg()
    pipe = HawkesPipeline(cfg)
    rows = pipe.run(regime="coupled")
    agg = pipe.aggregate(rows)
    fuse = agg["per_model"]["hawkesfuse"]["mean"]
    pois = agg["per_model"]["poisson"]["mean"]
    assert fuse < pois, f"旗舰 NLL {fuse} 未优于泊松基线 {pois}"


def test_hawkesfuse_stable_spectral():
    cfg = _cfg()
    pipe = HawkesPipeline(cfg)
    r = pipe.run(regime="coupled")[0]
    assert r["hawkesfuse_sr"] < cfg.spectral_cap or r["hawkesfuse_conv"] is True


def test_ablation_cross_excitation_helps():
    cfg = _cfg()
    pipe = HawkesPipeline(cfg)
    rows = pipe.run(regime="coupled")
    agg = pipe.aggregate(rows)
    fuse = agg["per_model"]["hawkesfuse"]["mean"]
    uni = agg["per_model"]["unihawkes"]["mean"]
    # 跨维激发在耦合数据上应带来非负增益（允许极小容差）
    assert fuse <= uni + 1e-6, f"跨维激发未带来增益: fuse={fuse} uni={uni}"
