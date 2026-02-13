import json

cells = []

def md(source):
    lines = source.strip().split("\n")
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]})

def code(source):
    lines = source.strip().split("\n")
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]})

md("""# Deep Hedging V2 — Black-Scholes, Merton & Heston

Pipeline complet de Deep Hedging avec le package `deep_hedging/`.

### Nouveautes V2 vs V1
| Feature | V1 | V2 |
|---|---|---|
| Features d'etat | 2 (S_rel, time) | 6 (log-moneyness, time, delta_{t-1}, vol realisee, BS delta, d1) |
| Architectures | MLP uniquement | MLP + LSTM (recurrent) |
| Mondes | BS, Merton | BS, Merton, **Heston** (vol stochastique) |
| Payoffs | Call/Put | Call, Put, **Asian, Straddle, Lookback** |
| Loss | CVaR empirique | CVaR + **OCE parametrique** (seuil appris) |
| Metriques | 7 | **10** (+ Hedging Error, Sharpe PnL, Cost/Payoff) |
| Baselines | Delta BS | Delta BS + **No Hedge** |
| LR scheduler | Non | **ReduceLROnPlateau** |
| Action clipping | Non | **Oui** (plus/moins 2.0) |

### Pipeline
1. Configuration
2. Visualisation des trajectoires (BS, Merton, Heston)
3. Training MLP et LSTM (BS puis Merton)
4. Evaluation multi-scenarios (6 scenarios)
5. Analyse de risque (VaR, CVaR, KDE, QQ-plot)
6. Tableau de synthese colore (10 metriques)
7. Comparaison MLP vs LSTM
8. Bonus : payoffs exotiques""")

md("## 1. Imports et Configuration")

code("""import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import replace

from deep_hedging import (
    DEVICE, DTYPE, N_FEATURES,
    DeepHedgingConfig, MarketConfig, TrainingConfig, RandomConfig,
    SimpleWorldBS, SimpleWorldMerton, SimpleWorldHeston,
    DeepHedgingEnv, PolicyMLP, PolicyLSTM, DeltaBSPolicy,
    MonetaryUtility, OCEUtility,
    train_deep_hedging,
    evaluate_strategies_env_world, build_comparison_table,
    RiskMetrics,
    plot_training_history, plot_gains_hist, plot_payoff_vs_gains,
    plot_simulated_paths, traffic_light_style,
)

print(f"Device         : {DEVICE}")
print(f"Features / step: {N_FEATURES}")
print(f"Dtype          : {DTYPE}")""")

md("## 2. Configuration globale")

code("""cfg = DeepHedgingConfig()

print("=== Marche ===")
print(f"  S0={cfg.market.S0}, sigma={cfg.market.sigma}, K={cfg.market.K}, T={cfg.market.T}")
print(f"  n_steps={cfg.market.n_steps}, cost_s={cfg.market.cost_s}")
print()
print("=== Merton ===")
print(f"  lambda_jump={cfg.market.lambda_jump}, mu_J={cfg.market.mu_J}, sigma_J={cfg.market.sigma_J}")
print()
print("=== Heston ===")
print(f"  kappa={cfg.market.heston_kappa}, theta={cfg.market.heston_theta}")
print(f"  xi={cfg.market.heston_xi}, rho={cfg.market.heston_rho}, v0={cfg.market.heston_v0}")
print()
print("=== Training ===")
print(f"  epochs={cfg.training.n_epochs}, batch_size={cfg.training.batch_size}")
print(f"  lr={cfg.training.lr}, CVaR alpha={cfg.training.cvar_alpha}")
print(f"  train paths={cfg.market.n_paths_train:,}, val paths={cfg.market.n_paths_val:,}")""")

md("""## 3. Visualisation des trajectoires simulees

Comparaison visuelle des trois mondes avec les memes parametres.""")

md("### 3.1 Black-Scholes")
code("""world_bs = SimpleWorldBS(cfg.market)
data_bs = world_bs.simulate_paths(200, seed=42)
plot_simulated_paths(data_bs["S"], n_paths_to_plot=20)
print(f"Payoff call moyen : {data_bs['payoff'].mean():.2f}")""")

md("### 3.2 Merton Jump-Diffusion")
code("""world_merton = SimpleWorldMerton(cfg.market)
data_merton = world_merton.simulate_paths(200, seed=42)
plot_simulated_paths(data_merton["S"], n_paths_to_plot=20)
print(f"Payoff call moyen : {data_merton['payoff'].mean():.2f}")""")

md("### 3.3 Heston Stochastic Volatility")
code("""world_heston = SimpleWorldHeston(cfg.market)
data_heston = world_heston.simulate_paths(200, seed=42)
plot_simulated_paths(data_heston["S"], n_paths_to_plot=20)
print(f"Payoff call moyen : {data_heston['payoff'].mean():.2f}")

# Trajectoires de variance v_t
fig, ax = plt.subplots(figsize=(10, 4))
for i in range(20):
    ax.plot(data_heston["v"][i], linewidth=0.8, alpha=0.7)
ax.axhline(cfg.market.heston_theta, color="red", linestyle="--",
           label=f"theta={cfg.market.heston_theta}")
ax.set_xlabel("Time steps")
ax.set_ylabel("Variance $v_t$")
ax.set_title("Heston - Variance Paths (mean-reverting)")
ax.legend()
ax.grid(alpha=0.4)
plt.tight_layout()
plt.show()""")

md("""## 4. Entrainement Deep Hedging

On entraine 4 modeles :
- MLP sous Black-Scholes
- LSTM sous Black-Scholes
- MLP sous Merton
- LSTM sous Merton

Chacun utilise les **6 features enrichies**, le **LR scheduler**, et l'**action clipping**.""")

md("### 4.1 MLP - Black-Scholes")
code("""cfg_bs = DeepHedgingConfig(
    market=replace(cfg.market, use_jumps=False, payoff_type="call"),
    training=cfg.training,
    random=cfg.random,
    device=cfg.device,
    dtype=cfg.dtype,
)

policy_mlp_bs = PolicyMLP(
    d_in=N_FEATURES, d_hidden=32, depth=2, dropout=0.1, clip=2.0
)
print(f"MLP : {sum(p.numel() for p in policy_mlp_bs.parameters())} params")

res_mlp_bs = train_deep_hedging(
    cfg_bs, policy_mlp_bs,
    utility=MonetaryUtility(kind="cvar", alpha=cfg_bs.training.cvar_alpha),
    patience=5, min_delta=1e-3, use_scheduler=True,
)
plot_training_history(res_mlp_bs["history"])""")

md("### 4.2 LSTM - Black-Scholes")
code("""policy_lstm_bs = PolicyLSTM(
    d_in=N_FEATURES, d_hidden=32, n_layers=1, dropout=0.0, clip=2.0
)
print(f"LSTM : {sum(p.numel() for p in policy_lstm_bs.parameters())} params")

res_lstm_bs = train_deep_hedging(
    cfg_bs, policy_lstm_bs,
    utility=MonetaryUtility(kind="cvar", alpha=cfg_bs.training.cvar_alpha),
    patience=5, min_delta=1e-3, use_scheduler=True,
)
plot_training_history(res_lstm_bs["history"])""")

md("### 4.3 MLP - Merton (avec sauts)")
code("""cfg_merton = DeepHedgingConfig(
    market=replace(cfg.market, use_jumps=True, payoff_type="call"),
    training=cfg.training,
    random=cfg.random,
    device=cfg.device,
    dtype=cfg.dtype,
)

policy_mlp_merton = PolicyMLP(
    d_in=N_FEATURES, d_hidden=32, depth=2, dropout=0.1, clip=2.0
)

res_mlp_merton = train_deep_hedging(
    cfg_merton, policy_mlp_merton,
    utility=MonetaryUtility(kind="cvar", alpha=cfg_merton.training.cvar_alpha),
    patience=5, min_delta=1e-3, use_scheduler=True,
)
plot_training_history(res_mlp_merton["history"])""")

md("### 4.4 LSTM - Merton (avec sauts)")
code("""policy_lstm_merton = PolicyLSTM(
    d_in=N_FEATURES, d_hidden=32, n_layers=1, dropout=0.0, clip=2.0
)

res_lstm_merton = train_deep_hedging(
    cfg_merton, policy_lstm_merton,
    utility=MonetaryUtility(kind="cvar", alpha=cfg_merton.training.cvar_alpha),
    patience=5, min_delta=1e-3, use_scheduler=True,
)
plot_training_history(res_lstm_merton["history"])""")

md("""## 5. Evaluation multi-scenarios

Six scenarios pour mesurer performance et robustesse :

| Scenario | Train | Test | Mesure |
|---|---|---|---|
| MLP BS->BS | BS | BS | Performance in-sample |
| LSTM BS->BS | BS | BS | LSTM vs MLP |
| MLP BS->Merton | BS | Merton | Robustesse aux sauts |
| MLP Merton->Merton | Merton | Merton | Performance in-sample |
| LSTM Merton->Merton | Merton | Merton | LSTM vs MLP |
| MLP Merton->Heston | Merton | Heston | Cross-model |

Chaque evaluation inclut **No Hedge**, **Delta BS**, **Deep Hedging**.""")

md("### 5.1 MLP : BS -> BS")
code("""eval_mlp_bs_bs = evaluate_strategies_env_world(
    cfg_bs, res_mlp_bs["policy"],
    world_class=SimpleWorldBS,
    n_paths_eval=20_000,
    seed_eval=cfg_bs.random.seed_eval_bs,
)
print("=== MLP : BS -> BS ===")
print(f"  CVaR Deep    : {eval_mlp_bs_bs['cvar_deep']:.4f}")
print(f"  CVaR Delta   : {eval_mlp_bs_bs['cvar_delta']:.4f}")
print(f"  CVaR No Hedge: {eval_mlp_bs_bs['cvar_no_hedge']:.4f}")
plot_gains_hist(eval_mlp_bs_bs)""")

md("### 5.2 LSTM : BS -> BS")
code("""eval_lstm_bs_bs = evaluate_strategies_env_world(
    cfg_bs, res_lstm_bs["policy"],
    world_class=SimpleWorldBS,
    n_paths_eval=20_000,
    seed_eval=cfg_bs.random.seed_eval_bs,
)
print("=== LSTM : BS -> BS ===")
print(f"  CVaR Deep  : {eval_lstm_bs_bs['cvar_deep']:.4f}")
print(f"  CVaR Delta : {eval_lstm_bs_bs['cvar_delta']:.4f}")
plot_gains_hist(eval_lstm_bs_bs)""")

md("### 5.3 MLP : BS -> Merton (robustesse)")
code("""eval_mlp_bs_merton = evaluate_strategies_env_world(
    cfg_bs, res_mlp_bs["policy"],
    world_class=SimpleWorldMerton,
    n_paths_eval=20_000,
    seed_eval=cfg_bs.random.seed_eval_bs_merton,
)
print("=== MLP : BS -> Merton ===")
print(f"  CVaR Deep    : {eval_mlp_bs_merton['cvar_deep']:.4f}")
print(f"  CVaR Delta   : {eval_mlp_bs_merton['cvar_delta']:.4f}")
print(f"  CVaR No Hedge: {eval_mlp_bs_merton['cvar_no_hedge']:.4f}")
plot_gains_hist(eval_mlp_bs_merton)""")

md("### 5.4 MLP : Merton -> Merton")
code("""eval_mlp_merton_merton = evaluate_strategies_env_world(
    cfg_merton, res_mlp_merton["policy"],
    world_class=SimpleWorldMerton,
    n_paths_eval=20_000,
    seed_eval=cfg_merton.random.seed_eval_merton,
)
print("=== MLP : Merton -> Merton ===")
print(f"  CVaR Deep  : {eval_mlp_merton_merton['cvar_deep']:.4f}")
print(f"  CVaR Delta : {eval_mlp_merton_merton['cvar_delta']:.4f}")
plot_gains_hist(eval_mlp_merton_merton)""")

md("### 5.5 LSTM : Merton -> Merton")
code("""eval_lstm_merton_merton = evaluate_strategies_env_world(
    cfg_merton, res_lstm_merton["policy"],
    world_class=SimpleWorldMerton,
    n_paths_eval=20_000,
    seed_eval=cfg_merton.random.seed_eval_merton,
)
print("=== LSTM : Merton -> Merton ===")
print(f"  CVaR Deep  : {eval_lstm_merton_merton['cvar_deep']:.4f}")
print(f"  CVaR Delta : {eval_lstm_merton_merton['cvar_delta']:.4f}")
plot_gains_hist(eval_lstm_merton_merton)""")

md("### 5.6 MLP : Merton -> Heston (cross-model)")
code("""eval_mlp_merton_heston = evaluate_strategies_env_world(
    cfg_merton, res_mlp_merton["policy"],
    world_class=SimpleWorldHeston,
    n_paths_eval=20_000,
    seed_eval=cfg_merton.random.seed_eval_merton,
)
print("=== MLP : Merton -> Heston ===")
print(f"  CVaR Deep    : {eval_mlp_merton_heston['cvar_deep']:.4f}")
print(f"  CVaR Delta   : {eval_mlp_merton_heston['cvar_delta']:.4f}")
print(f"  CVaR No Hedge: {eval_mlp_merton_heston['cvar_no_hedge']:.4f}")
plot_gains_hist(eval_mlp_merton_heston)""")

md("""## 6. Analyse de risque detaillee

Focus sur **Merton -> Merton** : comparaison Delta BS / MLP / LSTM.""")

code("""rm = RiskMetrics(alpha=cfg.training.cvar_alpha)

gains_delta = eval_mlp_merton_merton["gains_delta"]
gains_mlp   = eval_mlp_merton_merton["gains_deep"]
gains_lstm  = eval_lstm_merton_merton["gains_deep"]

for name, g in [("Delta BS", gains_delta), ("Deep MLP", gains_mlp), ("Deep LSTM", gains_lstm)]:
    s = rm.summary(g)
    print(f"{name:>10s} | Mean={s['Mean']:+.2f}  Std={s['Std']:.2f}  VaR={s['VaR']:.2f}  CVaR={s['CVaR']:.2f}")""")

md("### 6.1 Histogrammes avec VaR")
code("""rm.plot_hist_with_var(gains_delta, title="Delta Hedging - Merton")
rm.plot_hist_with_var(gains_mlp,  title="Deep Hedging MLP - Merton")
rm.plot_hist_with_var(gains_lstm, title="Deep Hedging LSTM - Merton")""")

md("### 6.2 KDE : comparaisons deux a deux")
code("""rm.plot_kde(gains_delta, gains_mlp, label_a="Delta", label_b="Deep MLP")
rm.plot_kde(gains_delta, gains_lstm, label_a="Delta", label_b="Deep LSTM")
rm.plot_kde(gains_mlp, gains_lstm, label_a="MLP", label_b="LSTM")""")

md("### 6.3 QQ-Plots vs distribution normale")
code("""rm.plot_qq(gains_mlp, title="Deep MLP - QQ-plot vs Normal")
rm.plot_qq(gains_lstm, title="Deep LSTM - QQ-plot vs Normal")""")

md("### 6.4 Queue gauche (tail risk)")
code("""rm.plot_left_tail(gains_delta, gains_mlp, label_a="Delta", label_b="Deep MLP")
rm.plot_left_tail(gains_delta, gains_lstm, label_a="Delta", label_b="Deep LSTM")""")

md("""## 7. Tableau de synthese

Comparaison coloree (traffic light) : 3 strategies x 6 scenarios x 10 metriques.""")

code("""scenarios = {
    "MLP BS->BS":           (cfg_bs,     eval_mlp_bs_bs),
    "LSTM BS->BS":          (cfg_bs,     eval_lstm_bs_bs),
    "MLP BS->Merton":       (cfg_bs,     eval_mlp_bs_merton),
    "MLP Merton->Merton":   (cfg_merton, eval_mlp_merton_merton),
    "LSTM Merton->Merton":  (cfg_merton, eval_lstm_merton_merton),
    "MLP Merton->Heston":   (cfg_merton, eval_mlp_merton_heston),
}

all_tables = []
for scenario_name, (c, ev) in scenarios.items():
    t = build_comparison_table(c, ev).copy()
    t.insert(0, "Scenario", scenario_name)
    all_tables.append(t)

summary_df = pd.concat(all_tables, axis=0)
summary_df = summary_df.reset_index().rename(columns={"index": "Strategy"})
summary_df = summary_df.set_index(["Scenario", "Strategy"])

styled = (
    summary_df.astype(float)
    .style
    .apply(traffic_light_style, axis=None)
    .format("{:.4f}")
)
display(styled)""")

md("""## 8. Comparaison directe MLP vs LSTM

Resume des performances cles pour les deux architectures.""")

code("""comparison_data = {
    "Scenario": ["BS->BS", "BS->BS", "Merton->Merton", "Merton->Merton"],
    "Architecture": ["MLP", "LSTM", "MLP", "LSTM"],
    "CVaR Deep": [
        eval_mlp_bs_bs["cvar_deep"],
        eval_lstm_bs_bs["cvar_deep"],
        eval_mlp_merton_merton["cvar_deep"],
        eval_lstm_merton_merton["cvar_deep"],
    ],
    "Std Deep": [
        eval_mlp_bs_bs["std_deep"],
        eval_lstm_bs_bs["std_deep"],
        eval_mlp_merton_merton["std_deep"],
        eval_lstm_merton_merton["std_deep"],
    ],
    "CVaR Delta (ref)": [
        eval_mlp_bs_bs["cvar_delta"],
        eval_lstm_bs_bs["cvar_delta"],
        eval_mlp_merton_merton["cvar_delta"],
        eval_lstm_merton_merton["cvar_delta"],
    ],
}

comp_df = pd.DataFrame(comparison_data).set_index(["Scenario", "Architecture"])
print(comp_df.round(4).to_string())
print()
print("Note: un CVaR plus eleve (moins negatif) = meilleur tail risk.")""")

md("""## 9. Bonus : Deep Hedging sur payoffs exotiques

Entrainement rapide du MLP sur 5 payoffs differents sous Black-Scholes :
- **Call** : max(S_T - K, 0)
- **Put** : max(K - S_T, 0)
- **Straddle** : |S_T - K|
- **Asian** : max(mean(S) - K, 0)  *(path-dependent)*
- **Lookback** : max(max(S) - K, 0)  *(path-dependent)*""")

code("""exotic_results = {}

for ptype in ["call", "put", "straddle", "asian", "lookback"]:
    print(f"\\n--- Payoff: {ptype.upper()} ---")
    cfg_exotic = DeepHedgingConfig(
        market=replace(cfg.market, use_jumps=False, payoff_type=ptype,
                       n_paths_train=50_000, n_paths_val=10_000),
        training=replace(cfg.training, n_epochs=30, print_every=10),
        random=cfg.random,
        device=cfg.device,
        dtype=cfg.dtype,
    )

    pol = PolicyMLP(d_in=N_FEATURES, d_hidden=32, depth=2, dropout=0.1, clip=2.0)
    res = train_deep_hedging(cfg_exotic, pol, patience=5, use_scheduler=True)

    ev = evaluate_strategies_env_world(
        cfg_exotic, res["policy"],
        world_class=SimpleWorldBS,
        n_paths_eval=10_000,
        seed_eval=42,
    )
    exotic_results[ptype] = ev

# Tableau recap
exotic_summary = pd.DataFrame({
    ptype: {
        "CVaR Deep": ev["cvar_deep"],
        "CVaR Delta": ev["cvar_delta"],
        "CVaR No Hedge": ev["cvar_no_hedge"],
        "Std Deep": ev["std_deep"],
        "Std Delta": ev["std_delta"],
    }
    for ptype, ev in exotic_results.items()
}).T

print("\\n=== Recap payoffs exotiques ===")
print(exotic_summary.round(4).to_string())""")

md("## 10. Sauvegarde des modeles")

code("""checkpoint = {
    "policy_mlp_bs": res_mlp_bs["policy"].state_dict(),
    "policy_lstm_bs": res_lstm_bs["policy"].state_dict(),
    "policy_mlp_merton": res_mlp_merton["policy"].state_dict(),
    "policy_lstm_merton": res_lstm_merton["policy"].state_dict(),
    "config_bs": cfg_bs,
    "config_merton": cfg_merton,
    "history_mlp_bs": res_mlp_bs["history"],
    "history_lstm_bs": res_lstm_bs["history"],
    "history_mlp_merton": res_mlp_merton["history"],
    "history_lstm_merton": res_lstm_merton["history"],
}

torch.save(checkpoint, "deep_hedging_models_v2.pt")
print(f"Checkpoint sauvegarde : deep_hedging_models_v2.pt")
print(f"Contient {len(checkpoint)} elements")""")

md("""## Conclusion

Ce notebook V2 orchestre le pipeline complet de Deep Hedging enrichi :

| Composant | Details |
|---|---|
| **Mondes** | Black-Scholes, Merton Jump-Diffusion, Heston Stochastic Vol |
| **Architectures** | MLP (feed-forward) et LSTM (recurrent) |
| **Features** | 6D enrichies : log-moneyness, time, prev_delta, realized_vol, BS delta, d1 |
| **Losses** | CVaR empirique, entropique, OCE parametrique |
| **Payoffs** | Call, Put, Straddle, Asian, Lookback |
| **Evaluation** | 3 strategies x 6 scenarios x 10 metriques |
| **Risk** | VaR, CVaR, KDE, QQ-plot, queue gauche |

### Structure du package

```
deep_hedging/
  __init__.py      # Public API
  config.py        # BS + Merton + Heston + payoff_type
  worlds.py        # BS, Merton, Heston + payoffs exotiques
  env.py           # 6 features enrichies
  policies.py      # MLP, LSTM, DeltaBS (avec clipping)
  losses.py        # CVaR, entropique, OCE parametrique
  training.py      # DataLoader + early stop + LR scheduler
  evaluation.py    # No Hedge + Delta + Deep, 10 metriques
  risk_metrics.py  # VaR, CVaR, histogrammes, KDE, QQ-plot
  plotting.py      # Visualisations + traffic light
  utils.py         # Fonctions utilitaires
```""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("DeepHedgingV2.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"DeepHedgingV2.ipynb cree avec {len(cells)} cellules.")
