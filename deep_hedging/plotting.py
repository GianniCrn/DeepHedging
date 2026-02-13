"""
Fonctions de visualisation : historique d'entrainement, gains, trajectoires, synthese coloree.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Historique d'entrainement
# ---------------------------------------------------------------------------

def plot_training_history(history: dict):
    """Trace les courbes de loss train / val + learning rate."""
    has_lr = "lr" in history and len(history["lr"]) > 0
    n_plots = 2 if has_lr else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Loss curves
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (-Utility)")
    ax.legend()
    ax.set_title("Deep Hedging — Training History")
    ax.grid(True)

    # LR curve
    if has_lr:
        ax2 = axes[1]
        ax2.plot(history["lr"], color="tab:orange")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Distribution des gains
# ---------------------------------------------------------------------------

def plot_gains_hist(eval_res: dict, bins: int = 50):
    """Histogramme superposes Delta vs Deep Hedging."""
    plt.figure(figsize=(6, 4))
    plt.hist(eval_res["gains_delta"], bins=bins, alpha=0.5, label="Delta")
    plt.hist(eval_res["gains_deep"], bins=bins, alpha=0.5, label="Deep Hedging")
    plt.xlabel("Gains")
    plt.ylabel("Frequency")
    plt.title("Distribution des gains : Delta vs Deep Hedging")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_payoff_vs_gains(eval_res: dict, n_points: int = 5_000):
    """Scatter payoff vs gains pour les deux strategies."""
    payoff = eval_res["payoff"]
    gains_deep = eval_res["gains_deep"]
    gains_delta = eval_res["gains_delta"]

    idx = np.random.choice(len(payoff), size=min(n_points, len(payoff)), replace=False)

    plt.figure(figsize=(6, 4))
    plt.scatter(payoff[idx], gains_delta[idx], s=10, alpha=0.4, label="Delta")
    plt.scatter(payoff[idx], gains_deep[idx], s=10, alpha=0.4, label="Deep Hedging")
    plt.xlabel("Payoff")
    plt.ylabel("Gains")
    plt.legend()
    plt.title("Payoff final vs Gains (Delta / Deep Hedging)")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# Trajectoires simulees
# ---------------------------------------------------------------------------

def plot_simulated_paths(S, n_paths_to_plot: int = 20):
    """Trace un sous-echantillon de trajectoires simulees."""
    n_paths, T_plus_1 = S.shape
    t = np.arange(T_plus_1)
    idx = np.random.choice(n_paths, size=min(n_paths_to_plot, n_paths), replace=False)

    plt.figure(figsize=(10, 5))
    for i in idx:
        plt.plot(t, S[i], linewidth=1)
    plt.xlabel("Time steps")
    plt.ylabel("Spot price S_t")
    plt.title("Simulated Price Paths")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Tableau de synthese colore (traffic light)
# ---------------------------------------------------------------------------

def traffic_light_style(df: pd.DataFrame, metrics_cfg: dict = None) -> pd.DataFrame:
    """
    Applique un code couleur vert/orange/rouge colonne par colonne.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame numerique a colorer.
    metrics_cfg : dict, optional
        {col_name: {"higher_is_better": bool}}. Si None, utilise un defaut.
    """
    if metrics_cfg is None:
        metrics_cfg = {
            "RMSE (vs payoff)": {"higher_is_better": False},
            "Mean Gains": {"higher_is_better": True},
            "Std Gains": {"higher_is_better": False},
            "Mean PnL": {"higher_is_better": True},
            "CVaR": {"higher_is_better": True},
            "Max Drawdown": {"higher_is_better": False},
            "Mean Cost": {"higher_is_better": False},
            "Hedging Error": {"higher_is_better": False},
            "Sharpe PnL": {"higher_is_better": True},
            "Cost / Payoff": {"higher_is_better": False},
        }

    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for col in df.columns:
        cfg_col = metrics_cfg.get(col)
        if cfg_col is None:
            continue

        col_values = df[col].astype(float)
        vmin, vmax = col_values.min(), col_values.max()

        for idx, val in col_values.items():
            if np.isnan(val):
                continue
            ratio = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            if not cfg_col["higher_is_better"]:
                ratio = 1.0 - ratio

            if ratio <= 1 / 3:
                color = "#ff4d4d"  # rouge
            elif ratio <= 2 / 3:
                color = "#ffa500"  # orange
            else:
                color = "#4caf50"  # vert

            styles.loc[idx, col] = f"background-color: {color}; color: white;"

    return styles
