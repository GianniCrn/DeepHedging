"""
Fonctions utilitaires : statistiques ponderees, RMSE, max drawdown.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Statistiques ponderees (PyTorch)
# ---------------------------------------------------------------------------

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
    """Moyenne ponderee : sum(w*x) / sum(w). Si w is None, simple moyenne."""
    if w is None:
        return x.mean()
    w = w / w.sum()
    return (w * x).sum()


def weighted_var(x: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
    """Variance ponderee."""
    m = weighted_mean(x, w)
    if w is None:
        return ((x - m) ** 2).mean()
    w = w / w.sum()
    return (w * (x - m) ** 2).sum()


def weighted_std(x: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
    """Ecart-type pondere."""
    return torch.sqrt(weighted_var(x, w))


# ---------------------------------------------------------------------------
# Metriques numpy
# ---------------------------------------------------------------------------

def compute_rmse(pred, target) -> float:
    """RMSE empirique."""
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(target)) ** 2)))


def compute_max_drawdown(series) -> float:
    """Max drawdown d'une serie de gains ou PnL."""
    series = np.asarray(series, dtype=float)
    cummax = np.maximum.accumulate(series)
    dd = (series - cummax) / np.where(cummax == 0, 1.0, cummax)
    return float(dd.min())
