# Pistes d'amélioration — Deep Hedging

> Basé sur l'analyse du repo de référence [`hansbuehler/deephedging`](https://github.com/hansbuehler/deephedging)
> et la littérature Deep Hedging (Buehler et al., 2019).

---

## Table des matières

1. [Impact fort — Prioritaires](#-impact-fort--prioritaires)
2. [Impact moyen — Recommandés](#-impact-moyen--recommandés)
3. [Quick wins — Faciles](#-quick-wins--faciles)
4. [Résumé priorisé](#-résumé-priorisé)

---

## 🔴 Impact fort — Prioritaires

### 1. Réseau récurrent (LSTM / GRU)

**Problème** : Le `PolicyMLP` actuel traite chaque pas de temps de manière indépendante. Il ne peut pas mémoriser l'historique des positions ni capturer des dépendances temporelles.

**Solution** : Buehler a ajouté le support LSTM/GRU en février 2023 dans son repo. Un réseau récurrent peut :
- Mémoriser les positions passées et l'historique des prix
- S'adapter dynamiquement à la trajectoire observée
- Mieux performer sur les **options path-dependent** (asiatiques, barriers, forward-started)

**Implémentation** :
- Créer une classe `PolicyLSTM` dans `policies.py`
- Architecture : LSTM → Linear → sigmoid/tanh pour borner les actions
- Comparaison MLP vs LSTM sur les mêmes scénarios

**Référence** : `notebooks/trainer-recurrent-fwdstart.ipynb` dans le repo Buehler

---

### 2. Features d'état enrichies

**Problème** : L'environnement n'utilise que 2 features : `S_t / S₀` (prix relatif) et `t / T` (temps écoulé). C'est très limité.

**Solution** : Ajouter des features informatives dans `env.py` :

| Feature | Formule | Justification |
|---|---|---|
| **Moneyness** | `log(S_t / K)` | Plus informatif que S/S₀ pour le hedging |
| **Position courante** | `δ_{t-1}` | Le réseau doit savoir sa position pour ajuster |
| **Vol réalisée glissante** | `std(log-returns, window=10)` | Signal de régime de vol |
| **Delta BS initial** | `Δ_BS(S_t, K, T-t)` | Feature-engineering de qualité |
| **Log-moneyness / vol√τ** | `log(S/K) / (σ√τ)` | Variable d1 standardisée |

**Impact** : Le réseau converge plus vite et atteint de meilleures performances avec des features pré-traitées.

---

### 3. Hedging multi-instruments

**Problème** : Le projet ne hedge qu'avec le sous-jacent spot. Le repo Buehler supporte le hedging avec **spot + option ATM**.

**Solution** :
- Ajouter un 2e instrument de hedging : option ATM vanille
- Le réseau output 2 actions par pas de temps : `(Δ_spot, Δ_option)`
- Permet de capturer le risque **gamma** et **vega** directement

**Complexité** : Nécessite de pricer l'option à chaque step (Black-Scholes) et de gérer les retours `DH_t` pour chaque instrument.

---

## 🟡 Impact moyen — Recommandés

### 4. Monde à volatilité stochastique (Heston)

**Problème** : BS = vol constante, Merton = vol constante + sauts. Aucun ne capture le **smile de volatilité dynamique**.

**Solution** : Implémenter le modèle de Heston :

$$dS_t = S_t \sqrt{v_t}\, dW_t^S$$
$$dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v$$

avec corrélation $\rho$ entre $W^S$ et $W^v$.

**Implémentation** : `SimpleWorldHeston` dans `worlds.py`

**Référence** : Buehler utilise un monde avec stochastic vol + mean-reverting drift.

---

### 5. OCE paramétrique (Optimized Certainty Equivalent)

**Problème** : Le CVaR actuel utilise un quantile fixe. L'OCE de Buehler optimise le seuil VaR conjointement avec la policy.

**Solution** : Le vrai OCE dual est :

$$\text{OCE}(X) = \sup_w \left\{ w + \mathbb{E}[u(X - w)] \right\}$$

où $w$ (le "VaR level") est un paramètre `nn.Parameter` appris par gradient descent.

**Impact** : Convergence plus stable et loss plus semantiquement correcte.

---

### 6. Payoffs exotiques

**Problème** : Le projet ne traite que les calls/puts européens.

**Solution** : Ajouter des payoffs dans `env.py` :

| Payoff | Formule | Intérêt |
|---|---|---|
| **Put européen** | `max(K - S_T, 0)` | Symétrie call/put |
| **Asiatique** | `max(S̄ - K, 0)` avec S̄ = moyenne | Path-dependent → LSTM nécessaire |
| **Barrier (knock-out)** | `max(S_T - K, 0) · 𝟙{S_t < B ∀t}` | Discontinuité → challenge pour NN |
| **Straddle** | `|S_T - K|` | Couverture delta-neutre |
| **Forward-started** | `max(S_T/S_{T/2} - 1, 0)` | Ref Buehler notebooks |

---

### 7. Benchmarks supplémentaires

**Problème** : Le seul benchmark est le Delta BS. C'est insuffisant pour évaluer le Deep Hedging.

**Solution** :

| Benchmark | Description |
|---|---|
| **Delta-Vega Hedging** | Delta BS + hedge en vega (avec une option) |
| **No Hedge** | Aucun hedging (payoff brut) → mesure la valeur ajoutée |
| **Variance Optimal** | Couverture quadratique minimale (Föllmer-Schweizer) |
| **Delta Merton** | Delta ajusté pour les sauts (si monde Merton) |

---

## 🟢 Quick wins — Faciles

### 8. Learning Rate Scheduler

**Actuel** : Learning rate fixe à 1e-3.

**Amélioration** :
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
# ou
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.training.n_epochs
)
```

---

### 9. Action clipping / Contraintes sur les positions

**Problème** : Aucune borne sur les actions → le réseau peut prendre des positions irréalistes.

**Solution** :
```python
# Dans PolicyMLP.forward()
out = self.net(X_flat)
out = torch.clamp(out, -2.0, 2.0)  # position max ±200%
```

Buehler applique des bornes configurables sur les actions.

---

### 10. Initial Delta Hedge (agent séparé)

**Problème** : La première action (t=0) est fondamentalement différente des suivantes (on part de position 0).

**Solution** : Buehler apprend un **agent séparé pour le hedge initial** :
- `init_delta_agent` : réseau small qui apprend δ₀
- Le réseau principal apprend les ajustements ∆δ_t pour t ≥ 1

Config : `config.gym.agent.init_delta.active = True`

---

### 11. Métriques supplémentaires

| Métrique | Formule | Mesure |
|---|---|---|
| **Hedging Error** | `std(Gains − Payoff)` | Qualité du hedge (plus bas = mieux) |
| **Sharpe du PnL** | `mean(PnL) / std(PnL)` | Efficience risque/rendement |
| **Coût / Payoff** | `mean(Cost) / mean(Payoff)` | Efficience en coûts |
| **P&L attribution** | Décomposition delta + gamma + vega + theta | Comprendre d'où vient le PnL |

---

## 📊 Résumé priorisé

| # | Amélioration | Module | Difficulté | Impact |
|---|---|---|---|---|
| 1 | Features enrichies (moneyness, vol, δ_{t-1}) | `env.py` | ⭐ | ⭐⭐⭐ |
| 2 | LSTM/GRU Policy | `policies.py` | ⭐⭐ | ⭐⭐⭐ |
| 3 | Action clipping + LR scheduler | `policies.py`, `training.py` | ⭐ | ⭐⭐ |
| 4 | Métriques supplémentaires | `evaluation.py`, `risk_metrics.py` | ⭐ | ⭐⭐ |
| 5 | Payoffs exotiques | `env.py` | ⭐⭐ | ⭐⭐ |
| 6 | Heston stochastic vol | `worlds.py` | ⭐⭐ | ⭐⭐ |
| 7 | Benchmarks (no hedge, delta-vega) | `evaluation.py` | ⭐ | ⭐⭐ |
| 8 | OCE paramétrique | `losses.py` | ⭐⭐ | ⭐⭐ |
| 9 | Hedging multi-instruments | `env.py`, `policies.py` | ⭐⭐⭐ | ⭐⭐⭐ |
| 10 | Initial delta hedge agent | `policies.py` | ⭐⭐ | ⭐ |

---

## Ordre d'implémentation recommandé

```
Phase 1 (Quick wins)           Phase 2 (Core)              Phase 3 (Avancé)
─────────────────              ──────────────              ────────────────
• Features enrichies           • LSTM/GRU Policy           • Hedging multi-instruments
• Action clipping              • Heston World              • OCE paramétrique
• LR scheduler                 • Payoffs exotiques         • Initial delta agent
• Métriques suppl.             • Benchmarks suppl.
```
