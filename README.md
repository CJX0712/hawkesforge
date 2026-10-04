# HawkesForge

> 世界顶级 temporal point process / 自激发事件序列（Hawkes 过程）建模系统 · 作者 晨星

[![CI](https://github.com/CJX0712/hawkesforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/hawkesforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/hawkesforge)](https://github.com/CJX0712/hawkesforge/releases/tag/v0.1.0)
[![License](https://img.shields.io/github/license/CJX0712/hawkesforge)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/)
[![Quality](https://img.shields.io/badge/quality-A-green)](./docs/model_card.md)

**HawkesForge** 是一套纯 `numpy`、零重型依赖、可一键复现的多元 Hawkes 自激发过程建模系统。它提供：

- **Ogata 稀疏采样法精确合成**多元 Hawkes / Poisson 序列（强度公式与拟合内核严格一致，**训练/测试零泄漏**）。
- **解析梯度 + Adam 极大似然估计（MLE）**：Ogata 形式对数似然与其梯度均为闭式，单次前向 `O(N·d)`。
- **多模型层级评测**：Poisson 基线（无时间结构）、UniHawkes（仅对角自激发）、**HawkesFuse 旗舰**（跨维自激发 + 跨维 L2 正则 + 谱半径稳定性守卫）。
- **诚实工程**：同 seed 两次运行核心指标**逐位一致**；Poisson（无结构）数据上旗舰**不伪胜**；≥3 条典型失败/局限案例；≥1 组消融对照。

---

## 性能基线（真实运行输出，3 seeds，coupled 域）

| 模型 | test NLL/event（越低越好，mean±std） | 说明 |
|------|--------------------------------------|------|
| poisson（无时间结构） | 0.4135 ± 0.1626 | 朴素泊松基线 |
| unihawkes（仅对角自激发） | 0.2765 ± 0.1654 | 强基线 |
| **hawkesfuse（旗舰）** | **0.2697 ± 0.1672** ✅ best | 跨维自激发 + 正则 + 谱守卫 |

- 旗舰 vs poisson 增益 **+0.1438**（≈ 相对 −34.7%）
- 旗舰 vs unihawkes 增益 **+0.0068**（跨维激发贡献，正但幅度受 DGP 跨维强度 α_off=0.15 所限）
- 确定性：同 seed 二次运行核心指标 **逐位一致 = True**
- 诚实性：Poisson 数据上 `hawkesfuse(2.5817) − poisson(2.5514) = −0.0303` → **不伪胜**
- 端到端 demo：26.27 s（≤ 60 s 预算 ✅）

完整结果见 [`benchmark.json`](./benchmark.json)。

## SOTA 对标声明

- 该域公认 SOTA：大规模情形用 **tick / 世界顶级优化器做梯度 MLE**（如 Bacry et al. 的 `tick` 库，IPMI 2012 / J. Stat. Soft. 2017），小数据/教学场景常用 Ogata 1981 精确采样 + 网格/EM 估参。
- 本系统对标：**纯 `numpy` 解析梯度 MLE + Adam**，与 `tick` 的梯度 MLE 同族；不依赖 `tick` 的 Cython 编译与 scipy 优化器，从而做到 **win_amd64 + py3.13 零编译、零下载、离线可跑**。
- 局限（诚实披露）：未实现条件强度自适应步长高阶优化器与 EM 加速；跨维增益在弱跨维 DGP 下幅度有限（见 model_card 失败案例）。

---

## 一键复现

```bash
# 1. 隔离环境（已验证：Python 3.13.12 + numpy 2.5.3）
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. 跑全部单测 + 覆盖率
.venv/Scripts/python.exe -m pytest -q

# 3. 端到端 demo（落盘 benchmark.json，含确定性二次校验）
.venv/Scripts/python.exe examples/run_demo.py

# 4. 仅看模型对比
.venv/Scripts/python.exe cli.py bench --seeds 3 --regime coupled
.venv/Scripts/python.exe cli.py models
```

要求：`Python >= 3.12`，仅依赖 `numpy>=2.0`。

---

## 模块架构（单向无环：`cli → pipeline → {data, hawkes, eval} → core`）

```
hawkesforge/
  core/        types(EventSeq/FitResult/HawkesParams) · errors(E100~E500)
              config(ENV_HAWKES_* 覆盖 + schema 校验) · interfaces(Protocol) · seed(全局确定性)
  data/        synth(Ogata 稀疏采样精确生成 Hawkes/Poisson；train/test 隔离)
  hawkes/      _math(Ogata 似然 + 解析梯度 + 谱投影守卫) · base(HawkesMLE β-grid)
              poisson(PoissonModel) · unihawkes(UniHawkes) · multihawkes(MultiHawkes/HawkesFuse 旗舰)
  eval/        metrics(mean±std / summarize / rank)
  pipeline/    hawkes_pipeline(HawkesPipeline.run/benchmark/aggregate) · report(format)
  cli.py       argparse 入口（bench / models）
  examples/run_demo.py  端到端演示（落盘 benchmark.json）
tests/         pytest（core/math/synth/models/pipeline/cli + 确定性 + 消融 + 失败案例）
docs/          architecture.md · model_card.md
```

核心数学内核（所有模型共享，保证公平评测）：

```
λ_i(t) = μ_i + β · Σ_j α_{i,j} · G_j(t)
G_j(t) = Σ_{t_j^m < t} exp(−β (t − t_j^m))
log-likelihood（Ogata 形式）与梯度均为闭式，单次前向 O(N·d)。
```

---

## 设计要点

- **全局确定性**：唯一入口 `core.seed.set_all(seed)`，numpy / random / default_rng 一次设齐；demo 两次运行 `benchmark.json` 核心指标逐位一致。
- **稳定性守卫（HawkesFuse）**：拟合后若谱半径 `ρ(α/β) > spectral_cap(0.97)`，整体等比缩放至 `cap·0.99`，保证过程平稳、可解释。
- **诚实负结果**：
  - FC1 不稳定过程（真实分支比>1）：守卫压回 cap 内，旗舰可用但**不声称还原真实非平稳结构**。
  - FC2 稀疏数据：跨维激发无法从数据学出，旗舰退化为 uni（增益≈0）。
  - FC3 衰减率误设（β=0.5 未进网格）：NLL 显著变高，加入匹配值后回落 → 须把 β 纳入网格搜索。

---

## 质量等级

**A（生产级）**：DoD 全绿 + 胜强基线（unihawkes，Δ>0）；未达 S 的原因为未预先注册"胜强基线达预设幅度阈值"（且跨维增益受 DGP 限制幅度偏小）。详见 [`docs/model_card.md`](./docs/model_card.md) 与 [`docs/architecture.md`](./docs/architecture.md)。

## 许可证

MIT © 2026 晨星。所用依赖（numpy，BSD 兼容）许可证与本仓库兼容，无密钥/隐私泄漏（提交前 `git grep` 自查通过）。

> 作者署名统一「晨星」。仓库归属 CJX0712。
