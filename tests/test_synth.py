"""合成数据生成不变量。"""

from __future__ import annotations

import numpy as np

from core.config import Config
from core.seed import set_all
from data.synth import generate_test, generate_train
from hawkes.poisson import PoissonModel


def _cfg():
    c = Config()
    c.n_dims = 3
    c.T = 60.0
    c.beta_grid = (1.0,)
    c.seed = 5
    return c


def test_generate_sorted_and_in_range():
    cfg = _cfg()
    set_all(cfg.seed)
    params, seq = generate_train(cfg, cfg.seed, regime="coupled")
    assert np.all(np.diff(seq.times) >= 0)
    assert seq.times.min() >= 0
    assert seq.times.max() <= seq.T + 1e-9
    assert seq.dims.min() >= 0 and seq.dims.max() < seq.d
    assert seq.d == cfg.n_dims
    assert params.spectral_radius() < 1.0


def test_train_test_independent():
    cfg = _cfg()
    params, train = generate_train(cfg, cfg.seed, regime="coupled")
    test = generate_test(params, cfg, cfg.seed + 999)
    # 独立采样，不应逐点相同
    assert not np.array_equal(train.times, test.times)


def test_poisson_regime_has_no_excitation():
    cfg = _cfg()
    params, seq = generate_train(cfg, cfg.seed, regime="poisson")
    assert np.allclose(params.alpha, 0.0)
    # 泊松数据下 poisson 模型 NLL 应接近理论值
    m = PoissonModel(cfg)
    fr = m.fit(seq)
    ll = np.sum(seq.counts() * np.log(fr.mu) - fr.mu * seq.T)
    assert abs(-ll / len(seq) - m.nll(fr, seq)) < 1e-12
