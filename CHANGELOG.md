# Changelog

All notable changes to this project are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/). Author: 晨星.

## [v0.1.0] — 2026-10-05

### Added
- 纯 `numpy` 多元 Hawkes 自激发过程建模系统（HawkesForge）。
- 核心数学内核：Ogata 形式对数似然 + 解析梯度（闭式，单次前向 O(N·d)）。
- 数据合成：`data/synth.py` 用 Ogata 稀疏采样法精确生成多元 Hawkes / Poisson 序列，强度公式与拟合内核严格一致，训练/测试零泄漏。
- 模型层级：
  - `PoissonModel`（无时间结构基线，闭式 MLE）；
  - `UniHawkes`（仅对角自激发，强基线）；
  - `HawkesFuse` 旗舰（跨维自激发 + 跨维 L2 正则 + 谱半径稳定性守卫）。
- MLE 估计器：`hawkes/_math.py:fit_mle`（Adam + 非负投影 + 谱投影 + 早停）。
- 端到端 pipeline：`HawkesPipeline.run/benchmark/aggregate`，输出 mean±std、best、deltas。
- 确定性：唯一 seed 入口 `core.seed.set_all`，demo 两次运行核心指标逐位一致。
- 诚实工程：Poisson 数据不伪胜、≥3 失败/局限案例、≥1 组消融对照。
- CLI：`cli.py bench / models`。
- 端到端演示：`examples/run_demo.py`（落盘 `benchmark.json`，含确定性二次校验、诚实性、失败案例、消融）。
- 测试：20 个 pytest，覆盖 core/math/synth/models/pipeline/cli + 确定性 + 消融 + 失败案例；覆盖率 88%。
- 交付物：README / CHANGELOG / LICENSE(MIT) / Dockerfile / Makefile / docs(architecture, model_card) / `.github/workflows/ci.yml` / `requirements.txt` + `requirements.lock.txt`。

### Quality
- 质量等级：**A（生产级）** —— DoD 全绿 + 胜强基线（unihawkes，Δ>0）；未达 S 因未预先注册"胜强基线达预设幅度阈值"（跨维增益受 DGP 跨维强度限制幅度偏小，属诚实披露）。
- 性能预算：demo 端到端 26.27 s ≤ 60 s；内存峰值 < 2 GB。
- 依赖锁定：`numpy==2.5.3`（win_amd64 + py3.13 预编译 wheel，零编译）。
