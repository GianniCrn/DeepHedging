# Deep Hedging — Neural Network-Based Option Hedging

> **Master 2 Finance** — Numerical Methods (Machine Learning) — 2025  
> Deep Hedging of European options under Black-Scholes and Merton Jump-Diffusion using PyTorch, with CVaR loss optimisation and multi-scenario benchmarking against Delta Hedging.

---
s
## Table of Contents

1. [Overview](#-overview)
2. [Quick Start](#-quick-start)
3. [Project Structure](#-project-structure)
4. [Methodology](#-methodology)
5. [Configuration](#%EF%B8%8F-configuration)
6. [Experiments & Results](#-experiments--results)
7. [Technical Stack](#-technical-stack)
8. [Reproducibility](#-reproducibility)
9. [References](#-references)

---

## 🔎 Overview

This project implements an **end-to-end Deep Hedging pipeline** for European vanilla options. A neural network learns an optimal hedging policy by minimising tail risk (CVaR), outperforming the classical Delta Hedging strategy — especially in the presence of **market frictions** (transaction costs) and **jump risk** (Merton model).

### Core Idea

Instead of relying on the model-dependent Black-Scholes delta, a **policy network** (MLP) observes the current spot price and time-to-maturity and outputs the optimal hedge ratio at each rebalancing date. The network is trained by minimising the **Conditional Value-at-Risk (CVaR)** of the hedging P&L, directly targeting worst-case losses.

### Problem Setup

| Parameter | Value |
|---|---|
| **Underlying** | Synthetic stock, S₀ = 100 |
| **Option** | European Call, K = 100, T = 1 year |
| **Rebalancing** | Weekly (52 steps/year) |
| **Transaction Costs** | 2 bps proportional (0.02%) |
| **Portfolio Size** | 200,000 training paths / 50,000 validation paths |

### Market Models

| Model | Dynamics | Features |
|---|---|---|
| **Black-Scholes** | dS = S·(r dt + σ dW) | Continuous diffusion, no jumps |
| **Merton Jump-Diffusion** | dS/S = (r − λκ) dt + σ dW + dJ | Poisson jumps with log-normal sizes |

### Key Results

| Metric | Delta Hedging | Deep Hedging | Improvement |
|---|---|---|---|
| **CVaR (2.5%)** | Baseline | Lower tail risk | ✅ Better worst-case |
| **Std of Gains** | Higher | Lower | ✅ Tighter P&L |
| **Transaction Costs** | Model-agnostic | Learned cost-awareness | ✅ Smarter rebalancing |
| **Jump Robustness** | Degrades with jumps | Adapts to jumps | ✅ More robust |

---

## 🚀 Quick Start

### Prerequisites

| Tool | Details |
|---|---|
| **Python** | 3.9+ |
| **PyTorch** | 1.13+ (CPU, CUDA or MPS) |
| **Hardware** | CPU sufficient; MPS/CUDA recommended for faster training |

### Installation

```bash
# Navigate to project
cd "Deep Hedging/il marche"

# Install dependencies
pip install torch numpy scipy matplotlib pandas
```

### Run the Complete Pipeline

Open the orchestration notebook in Jupyter:

```bash
jupyter notebook DeepHedging.ipynb
```

Then run all cells sequentially. The pipeline executes:

1. **Configuration** → Market parameters, training hyperparameters
2. **Visualisation** → Simulated price paths (BS & Merton)
3. **Training BS** → Deep Hedging policy under Black-Scholes
4. **Training Merton** → Deep Hedging policy under Merton Jump-Diffusion
5. **Evaluation** → 3 cross-scenarios (BS→BS, BS→Merton, Merton→Merton)
6. **Risk Analysis** → VaR, CVaR, KDE, QQ-plot, tail zoom
7. **Synthesis** → Colour-coded comparison table (traffic light)
8. **Model Saving** → `deep_hedging_models.pt`

### Quick Test (command line)

```bash
python3 -c "
from deep_hedging import *
cfg = DeepHedgingConfig()
cfg.market.use_jumps = False
world = SimpleWorldBS(cfg.market)
data = world.simulate_paths(100)
print('Simulation OK:', data['S'].shape)
print('All imports OK')
"
```

---

## 📁 Project Structure

```
il marche/
├── DeepHedging.ipynb              # Orchestration notebook (run this)
├── README.md                      # This file
├── deep_hedging_models.pt         # Saved trained models (generated)
│
├── deep_hedging/                  # Python package (10 modules)
│   ├── __init__.py                # Public API, re-exports all symbols
│   ├── config.py                  # Device detection + dataclasses
│   ├── worlds.py                  # Price simulators (BS, Merton)
│   ├── env.py                     # Hedging environment (rollout, P&L)
│   ├── policies.py                # Neural + analytical policies
│   ├── losses.py                  # Monetary utilities (CVaR, entropic)
│   ├── training.py                # Training loop (DataLoader, early stop)
│   ├── evaluation.py              # Strategy comparison + synthesis tables
│   ├── risk_metrics.py            # VaR/CVaR, KDE, QQ-plot
│   ├── plotting.py                # Visualisation functions
│   └── utils.py                   # Weighted stats, RMSE, max drawdown
│
├── DeepHedgingFonctionnel.ipynb   # Original full notebook (reference)
└── DeepHedging_Optimized.ipynb    # Simplified version (reference)
```

---

## 🎯 Methodology

### 1. Simulation — Price Path Generation

#### Black-Scholes (GBM)

Log-returns are simulated as:

$$\ln\frac{S_{t+1}}{S_t} = \left(r - q - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\,Z_t, \quad Z_t \sim \mathcal{N}(0,1)$$

#### Merton Jump-Diffusion

Adds compound Poisson jumps to the GBM:

$$\frac{dS_t}{S_{t^-}} = (r - q - \lambda\kappa)\,dt + \sigma\,dW_t + dJ_t$$

where $J_t = \sum_{i=1}^{N_t}(e^{Y_i} - 1)$, $N_t \sim \text{Poisson}(\lambda\Delta t)$, $Y_i \sim \mathcal{N}(\mu_J, \sigma_J^2)$, and $\kappa = e^{\mu_J + \sigma_J^2/2} - 1$ ensures risk-neutral drift.

| Parameter | Symbol | Default |
|---|---|---|
| Initial spot | S₀ | 100 |
| Risk-free rate | r | 3% |
| BS volatility | σ | 20% |
| Jump intensity | λ | 1.0 /year |
| Mean log-jump | μ_J | −0.10 |
| Std log-jump | σ_J | 0.30 |
| Transaction cost | c | 0.02% |

---

### 2. Deep Hedging Architecture

#### State Features

At each rebalancing date $t$, the policy observes:

| Feature | Formula | Meaning |
|---|---|---|
| **Relative spot** | $S_t / S_0$ | Normalised price level |
| **Time fraction** | $t / T$ | Elapsed time (0 → 1) |

#### Policy Network (MLP)

```
Input (B, T, 2) → [Linear → ReLU → Dropout] × depth → Linear → Output (B, T)
```

| Hyperparameter | Default |
|---|---|
| Hidden dimension | 32 |
| Depth | 2 layers |
| Dropout | 10% |
| Output | Δ-position (action) per step |

The network outputs **delta-actions** (position changes), which are cumulated: $\delta_t = \sum_{s=0}^{t} a_s$.

#### Benchmark — Delta Black-Scholes Policy

An analytical policy computing the BS delta at each step, compatible with the same environment:

$$\Delta_t^{BS} = e^{-qT_{rem}} \cdot \mathcal{N}(d_1), \quad d_1 = \frac{\ln(S_t/K) + (r - q + \sigma^2/2)T_{rem}}{\sigma\sqrt{T_{rem}}}$$

---

### 3. P&L Computation & Transaction Costs

The hedging environment computes the total **gains** for each path:

$$\text{Gains} = \underbrace{\text{Payoff}}_{\max(S_T - K, 0)} + \underbrace{\text{PnL}}_{\sum_t \delta_t \cdot \Delta S_t} - \underbrace{\text{Cost}}_{c \sum_t |a_t| \cdot S_t}$$

| Component | Formula | Description |
|---|---|---|
| **PnL** | $\sum_{t=0}^{T-1} \delta_t \cdot (S_{t+1} - S_t)$ | Trading gains from hedging |
| **Cost** | $c \sum_{t=0}^{T-1} \lvert a_t \rvert \cdot S_t$ | Proportional transaction costs |
| **Gains** | Payoff + PnL − Cost | Net hedging outcome |

---

### 4. Loss Function — CVaR Optimisation

The network is trained to **maximise** the monetary utility of gains, equivalently **minimise** the loss:

$$\mathcal{L} = -\text{CVaR}_\alpha(\text{Gains}) + \lambda \cdot \frac{1}{BT}\sum_{b,t} a_{b,t}^2$$

where CVaR at level $\alpha$ is the **expected value of the worst $\alpha$-fraction** of gains:

$$\text{CVaR}_\alpha(X) = \mathbb{E}[X \mid X \leq \text{VaR}_\alpha(X)]$$

| Utility Mode | Formula | Use Case |
|---|---|---|
| **CVaR** (default) | Mean of worst α% | Tail risk minimisation |
| **Entropic** | $-\frac{1}{\lambda}\ln\mathbb{E}[e^{-\lambda X}]$ | Exponential risk aversion |
| **Mean** | $\mathbb{E}[X]$ | Risk-neutral baseline |

Default: **α = 2.5%** (focus on the 2.5% worst scenarios).

---

### 5. Training Loop

| Feature | Implementation |
|---|---|
| **Batching** | PyTorch DataLoader, batch_size = 10,000 |
| **Optimiser** | Adam (lr = 1e-3) |
| **Gradient clipping** | Max norm = 1.0 |
| **Early stopping** | Patience = 5 epochs on validation loss |
| **Best model** | Restored from checkpoint with lowest val_loss |
| **Regularisation** | L2 penalty on actions (λ = 1e-5) + Dropout |

---

### 6. Evaluation — Multi-Scenario Benchmarking

Three cross-model experiments to assess **generalisation** and **robustness**:

| Scenario | Training World | Test World | Purpose |
|---|---|---|---|
| **BS → BS** | Black-Scholes | Black-Scholes | In-sample performance |
| **BS → Merton** | Black-Scholes | Merton (jumps) | Robustness to model mis-specification |
| **Merton → Merton** | Merton | Merton | Optimal under jump dynamics |

For each scenario, both **Deep Hedging** and **Delta BS** strategies are evaluated on 20,000 fresh paths.

---

### 7. Risk Analytics

The `RiskMetrics` class provides a full risk analysis toolkit:

| Analysis | Method |
|---|---|
| **VaR** | Empirical quantile at level α |
| **CVaR** | Mean of observations ≤ VaR |
| **Histogram** | Distribution with VaR/CVaR markers |
| **KDE** | Kernel Density Estimation (Gaussian kernel) |
| **QQ-Plot** | Quantile-quantile vs Normal distribution |
| **Left Tail Zoom** | Focus on worst 30% of scenarios |

---

## ⚙️ Configuration

All parameters are set via **Python dataclasses** in `deep_hedging/config.py`:

```python
from deep_hedging import DeepHedgingConfig, MarketConfig, TrainingConfig
from dataclasses import replace

# Default configuration
cfg = DeepHedgingConfig()

# Custom BS configuration (no jumps)
cfg_bs = DeepHedgingConfig(
    market=replace(cfg.market, use_jumps=False),
    training=TrainingConfig(n_epochs=50, batch_size=10_000, cvar_alpha=0.025),
)
```

### Market Parameters

| Parameter | Field | Default | Description |
|---|---|---|---|
| S₀ | `S0` | 100.0 | Initial spot price |
| r | `r` | 0.03 | Risk-free rate |
| q | `q` | 0.0 | Dividend yield |
| σ | `sigma` | 0.2 | BS volatility |
| T | `T` | 1.0 | Option maturity (years) |
| K | `K` | 100.0 | Strike price |
| n_steps | `n_steps` | 52 | Rebalancing frequency |
| cost | `cost_s` | 0.0002 | Transaction cost (bps) |
| λ | `lambda_jump` | 1.0 | Poisson jump intensity |
| μ_J | `mu_J` | −0.1 | Mean log-jump size |
| σ_J | `sigma_J` | 0.3 | Std of log-jump size |

### Training Parameters

| Parameter | Field | Default |
|---|---|---|
| Epochs | `n_epochs` | 50 |
| Batch size | `batch_size` | 10,000 |
| Learning rate | `lr` | 1e-3 |
| CVaR level | `cvar_alpha` | 0.025 |
| Train paths | `n_paths_train` | 200,000 |
| Val paths | `n_paths_val` | 50,000 |

---

## 📊 Experiments & Results

### Training Convergence

The CVaR loss converges within ~30–50 epochs with early stopping:

- **BS training**: Smooth convergence, stable validation loss
- **Merton training**: Slightly noisier due to jump discontinuities

### Comparison Table (Traffic Light System)

The final synthesis table compares all strategies across all scenarios using a **colour-coded ranking**:

| Metric | Higher is Better? | Measures |
|---|---|---|
| RMSE (vs payoff) | ❌ | Hedging accuracy |
| Mean Gains | ✅ | Average profitability |
| Std Gains | ❌ | P&L volatility |
| Mean PnL | ✅ | Trading profit |
| CVaR | ✅ | Tail risk (worst-case) |
| Max Drawdown | ❌ | Worst cumulative loss |
| Mean Cost | ❌ | Transaction costs |

🟢 Green = best · 🟡 Orange = intermediate · 🔴 Red = worst

### Key Findings

1. **BS → BS**: Deep Hedging matches or slightly outperforms Delta in a frictionless BS world, with lower transaction costs due to learned cost-aware rebalancing.

2. **BS → Merton**: Delta Hedging degrades significantly when the test world includes jumps. Deep Hedging trained on BS is also affected but typically maintains better tail risk.

3. **Merton → Merton**: Deep Hedging trained on Merton data **significantly outperforms** Delta Hedging, especially on CVaR and tail metrics — the network learns to anticipate jump risk.

---

## 🛠 Technical Stack

### Core Libraries

| Category | Packages |
|---|---|
| **Deep Learning** | `torch` (PyTorch) — models, training, GPU/MPS |
| **Numerical** | `numpy`, `scipy` (stats, optimisation) |
| **Data** | `pandas` |
| **Visualisation** | `matplotlib` |

### Device Support

| Device | Status | Detection |
|---|---|---|
| **CPU** | ✅ Always available | Fallback |
| **CUDA** | ✅ Auto-detected | NVIDIA GPUs |
| **MPS** | ✅ Auto-detected | Apple Silicon (M1/M2/M3) |

Auto-detection in `deep_hedging/config.py`:

```python
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
```

### Models Implemented

| Model | Module | Usage |
|---|---|---|
| **PolicyMLP** | `policies.py` | Neural hedging policy (MLP) |
| **DeltaBSPolicy** | `policies.py` | Analytical BS delta (benchmark) |
| **Black-Scholes** | `worlds.py` | GBM price simulation |
| **Merton Jump-Diffusion** | `worlds.py` | Jump-diffusion simulation |
| **CVaR (Rockafellar-Uryasev)** | `losses.py` | Tail risk loss function |
| **Entropic Utility** | `losses.py` | Exponential risk measure |

---

## 🔒 Reproducibility

| Guarantee | Implementation |
|---|---|
| **Determinism** | Fixed seeds in `RandomConfig` (default: 1234) |
| **No external data** | Fully synthetic — no market data download needed |
| **Device-agnostic** | Auto-detection CPU/CUDA/MPS |
| **Centralised config** | All parameters in `DeepHedgingConfig` dataclass |
| **Modular code** | Each module is independently importable and testable |
| **Early stopping** | Best model checkpoint restored automatically |

### Reproducing Results

```bash
# Full pipeline (in Jupyter)
jupyter notebook DeepHedging.ipynb
# → Run All Cells

# Or programmatically
python3 -c "
from deep_hedging import *
from dataclasses import replace

cfg = DeepHedgingConfig(market=replace(MarketConfig(), use_jumps=False))
policy = PolicyMLP(d_in=2, d_hidden=32, depth=2, dropout=0.1)
res = train_deep_hedging(cfg, policy, patience=5)
eval_res = evaluate_strategies_env_world(cfg, res['policy'], SimpleWorldBS)
table = build_comparison_table(cfg, eval_res)
print(table)
"
```

---

## 📚 References

### Deep Hedging
- Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). *Deep Hedging*. Quantitative Finance, 19(8), 1271–1291.
- Buehler, H., Gonon, L., Teichmann, J., Wood, B., Mohan, B., & Kochems, J. (2019). *Deep Hedging: Hedging Derivatives Under Generic Market Frictions Using Reinforcement Learning*. Swiss Finance Institute Research Paper No. 19-80.

### Risk Measures
- Rockafellar, R.T., & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk, 2, 21–41.
- Artzner, P., Delbaen, F., Eber, J.M., & Heath, D. (1999). *Coherent Measures of Risk*. Mathematical Finance, 9(3), 203–228.

### Option Pricing & Hedging
- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy, 81(3), 637–654.
- Merton, R.C. (1976). *Option Pricing When Underlying Stock Returns Are Discontinuous*. Journal of Financial Economics, 3(1–2), 125–144.

### Neural Networks in Finance
- Ruf, J., & Wang, W. (2020). *Neural Networks for Option Pricing and Hedging: A Literature Review*. Journal of Computational Finance, 24(1).
- Cao, J., Chen, J., Hull, J., & Poulos, Z. (2021). *Deep Hedging of Derivatives Using Reinforcement Learning*. Journal of Financial Data Science, 3(1), 10–27.

### Utility Theory
- Ben-Tal, A., & Teboulle, M. (2007). *An Old-New Concept of Convex Risk Measures: The Optimized Certainty Equivalent*. Mathematical Finance, 17(3), 449–476.
- Föllmer, H., & Schied, A. (2016). *Stochastic Finance: An Introduction in Discrete Time* (4th ed.). De Gruyter.

---

## 👤 Author

**Gianni Carena**   
**Course**: Numerical Methods (Machine Learning)  
**Degree**: Master 2 Finance  
**Year**: 2024–2025  

---

<p align="center"><b>Status</b>: ✅ Pipeline complet — BS & Merton, training, evaluation, 3 scénarios<br><i>Dernière mise à jour : Février 2025</i></p>
