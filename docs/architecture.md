# Architecture — HawkesForge

> 作者 晨星 · 纯 `numpy` 多元 Hawkes 自激发过程建模系统

## 1. 设计原则

1. **零重型依赖**：仅 `numpy>=2.0`，`win_amd64 + py3.13` 有预编译 wheel，零编译、零下载、离线可跑。
2. **单一数学内核**：所有模型共用 `hawkes/_math.py` 的 Ogata 形式对数似然与解析梯度，保证跨模型公平评测。
3. **全局确定性**：唯一 seed 入口 `core.seed.set_all(seed)`，numpy / random / default_rng 一次设齐；demo 两次运行核心指标逐位一致。
4. **无泄漏**：合成数据的训练/测试由同一组参数**独立采样**得到，预处理只在 train 拟合（本系统无 scaler，DGP 即唯一真值来源）。
5. **单向无环**：`cli → pipeline → {data, hawkes, eval} → core`。

## 2. 目录骨架

```
hawkesforge/
  core/
    types.py      EventSeq / FitResult / HawkesParams（含谱半径计算）
    errors.py     HawkesError + ConfigError(E100)/DataError(E200)/FitError(E300)/EvalError(E400)/PipelineError(E500)
    config.py     Config 数据类（ENV_HAWKES_* 覆盖 + validate 硬约束）
    interfaces.py  IntensityModel Protocol（available/fit/nll/intensity_at）
    seed.py       set_all/get_seed/rng（全局确定性）
  data/
    synth.py      make_params / _generate_hawkes(Ogata 稀疏采样) / _generate_poisson / generate_train / generate_test
  hawkes/
    _math.py      log_likelihood_and_grad（闭式梯度）/ spectral_radius / branching_ratio / _project_spectral / fit_mle
    base.py       HawkesMLE（β-grid 搜索 + 早停）
    poisson.py    PoissonModel（无时间结构基线）
    unihawkes.py  UniHawkes（仅对角自激发，强基线）
    multihawkes.py MultiHawkes + HawkesFuse（旗舰：跨维 + L2 正则 + 谱守卫）
    __init__.py   all_models(cfg) -> [PoissonModel, UniHawkes, HawkesFuse]
  eval/
    metrics.py    mean_std / summarize / rank_by_nll
  pipeline/
    hawkes_pipeline.py  HawkesPipeline.run/benchmark/aggregate/core_metrics
    report.py           format_report（电报式表）
  cli.py          argparse: bench / models
  examples/run_demo.py  端到端演示（benchmark.json）
tests/            20 个 pytest（core/math/synth/models/pipeline/cli + 确定性 + 消融 + 失败案例）
docs/             architecture.md / model_card.md
```

## 3. 核心数学

强度模型（共享全局衰减率 β）：

```
λ_i(t) = μ_i + β · Σ_j α_{i,j} · G_j(t)
G_j(t) = Σ_{t_j^m < t} exp(−β (t − t_j^m))
```

- **对数似然（Ogata 形式）**：O(N·d) 单次前向扫描，同时累加 `ll`、`grad_mu`、`grad_alpha`。
- **解析梯度**：由链式法则直接给出，无需自动微分；与朴素 O(N²) 参考实现交叉验证误差 < 1e-9，与有限差分验证误差 < 1e-4。
- **MLE（fit_mle）**：Adam 优化，非负投影（`clip(μ, min_intensity, ·)`、α ≥ 0），HawkesFuse 额外做跨维 L2 正则 + **谱投影守卫**（`ρ(α/β) > cap` 时整体等比缩放至 `cap·0.99`）。`|Δll| < tol·|ll|` 早停。

## 4. 数据生成（Ogata 稀疏采样）

`_generate_hawkes` 实现 Ogata (1981)  thinning：以当前总强度为上界抽样候选时刻，按 `λ(t)/λ*(t)` 接受。强度公式与拟合内核**逐字一致**——这是「无泄漏」的根本保证。单序列事件上限 `Config.max_events`（默认 20000）防止超临界过程事件爆炸。

## 5. 评测链路

`HawkesPipeline`
- `run(regime)`：每 seed 独立生成 train/test，逐个模型 `fit` → 记录 test NLL/event、谱半径、收敛标志、β。
- `benchmark()`：n_seeds 次 `run` 聚合为 `rows`。
- `aggregate()`：per_model mean±std、best、deltas（`hawkesfuse_vs_poisson` / `hawkesfuse_vs_unihawkes`）。
- `core_metrics(rows)`：提取核心数值用于**确定性逐位复验**。

## 6. 模块接口契约

`IntensityModel.available() → bool`：本系统全部纯 numpy，恒为 True（离线兜底即自身）。
`fit(seq) → FitResult`；`nll(fr, test) → float`（越小越好）；`intensity_at(fr, t, dim) → float`。

## 7. 失败案例与消融（详见 model_card）

- FC1 不稳定过程（真实分支比>1）→ 谱守卫压回 cap，旗舰可用但不声称还原真实结构。
- FC2 稀疏数据 → 跨维激发不可辨识，旗舰退化为 uni（增益≈0）。
- FC3 衰减率误设（β=0.5 未进网格）→ NLL 显著变高，匹配后回落。
- 消融：HawkesFuse vs UniHawkes（无跨维）+0.0068；vs Poisson（无时间结构）+0.1438。
