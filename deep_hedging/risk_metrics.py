"""
Metriques de risque : VaR, CVaR empiriques, histogrammes, KDE, QQ-plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, probplot


class RiskMetrics:
    """
    Outil centralise pour analyser la distribution des gains :
        - VaR / CVaR empiriques
        - Histogrammes avec marqueurs VaR
        - KDE (densite estimee)
        - QQ-Plot vs loi normale
        - Zoom sur la queue gauche
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    # ------------------------------------------------------------------
    # VaR / CVaR
    # ------------------------------------------------------------------
    def empirical_var(self, samples: np.ndarray) -> float:
        """VaR empirique (quantile alpha, queue gauche)."""
        samples = np.asarray(samples, dtype=float)
        return float(np.quantile(samples, self.alpha))

    def empirical_cvar(self, samples: np.ndarray) -> float:
        """CVaR empirique = moyenne des valeurs <= VaR."""
        samples = np.asarray(samples, dtype=float)
        var = self.empirical_var(samples)
        tail = samples[samples <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    def summary(self, samples: np.ndarray) -> dict:
        """Dictionnaire resume : mean, std, VaR, CVaR."""
        samples = np.asarray(samples, dtype=float)
        return {
            "Mean": float(samples.mean()),
            "Std": float(samples.std()),
            "VaR": self.empirical_var(samples),
            "CVaR": self.empirical_cvar(samples),
        }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_hist_with_var(self, gains, title: str = "Distribution des gains", bins: int = 60):
        gains = np.asarray(gains, dtype=float)
        var = self.empirical_var(gains)
        cvar = self.empirical_cvar(gains)

        plt.figure(figsize=(7, 5))
        plt.hist(gains, bins=bins, alpha=0.7, edgecolor="black", linewidth=0.5)
        plt.axvline(var, color="red", linestyle="--", linewidth=2,
                    label=f"VaR {self.alpha:.0%} = {var:.2f}")
        plt.axvline(cvar, color="orange", linestyle="--", linewidth=2,
                    label=f"CVaR {self.alpha:.0%} = {cvar:.2f}")
        plt.xlabel("Gains")
        plt.ylabel("Frequence")
        plt.title(title)
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.show()

    def plot_overlap_with_var(self, gains_a, gains_b, label_a="Delta", label_b="Deep", bins=60):
        gains_a = np.asarray(gains_a, dtype=float)
        gains_b = np.asarray(gains_b, dtype=float)

        var_a = self.empirical_var(gains_a)
        var_b = self.empirical_var(gains_b)

        plt.figure(figsize=(7, 5))
        plt.hist(gains_a, bins=bins, alpha=0.5, label=f"{label_a} (VaR={var_a:.2f})")
        plt.hist(gains_b, bins=bins, alpha=0.5, label=f"{label_b} (VaR={var_b:.2f})")
        plt.axvline(var_a, color="blue", linestyle="--", linewidth=1.5)
        plt.axvline(var_b, color="orange", linestyle="--", linewidth=1.5)
        plt.xlabel("Gains")
        plt.ylabel("Frequence")
        plt.title("Overlap Delta vs Deep Hedging")
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.show()

    def plot_kde(self, gains_a, gains_b, label_a="Delta", label_b="Deep"):
        gains_a = np.asarray(gains_a, dtype=float)
        gains_b = np.asarray(gains_b, dtype=float)

        x_min = min(gains_a.min(), gains_b.min())
        x_max = max(gains_a.max(), gains_b.max())
        x_grid = np.linspace(x_min, x_max, 400)

        kde_a = gaussian_kde(gains_a)
        kde_b = gaussian_kde(gains_b)

        plt.figure(figsize=(7, 5))
        plt.plot(x_grid, kde_a(x_grid), label=f"{label_a} (KDE)")
        plt.plot(x_grid, kde_b(x_grid), label=f"{label_b} (KDE)")
        plt.xlabel("Gains")
        plt.ylabel("Densite")
        plt.title("Densite estimee (KDE)")
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.show()

    def plot_qq(self, gains, title: str = "QQ-plot vs Normal"):
        gains = np.asarray(gains, dtype=float)
        plt.figure(figsize=(6, 5))
        probplot(gains, dist="norm", plot=plt.gca())
        plt.title(title)
        plt.grid(True, alpha=0.4)
        plt.show()

    def plot_left_tail(self, gains_a, gains_b, label_a="Delta", label_b="Deep", q_tail=0.30):
        gains_a = np.asarray(gains_a, dtype=float)
        gains_b = np.asarray(gains_b, dtype=float)

        cut_a = np.quantile(gains_a, q_tail)
        cut_b = np.quantile(gains_b, q_tail)

        var_a = self.empirical_var(gains_a)
        var_b = self.empirical_var(gains_b)

        plt.figure(figsize=(7, 5))
        plt.hist(gains_a[gains_a <= cut_a], bins=40, alpha=0.6, label=label_a)
        plt.hist(gains_b[gains_b <= cut_b], bins=40, alpha=0.6, label=label_b)
        plt.axvline(var_a, linestyle="--", label=f"VaR {label_a} {self.alpha:.0%}")
        plt.axvline(var_b, linestyle="--", label=f"VaR {label_b} {self.alpha:.0%}")
        plt.xlabel("Gains (zone pertes)")
        plt.ylabel("Frequence")
        plt.title(f"Queue gauche ({int(q_tail * 100)}% pires scenarios)")
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.show()
