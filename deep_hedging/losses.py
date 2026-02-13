"""
Fonctions de perte et utilitaires monetaires :
    - cvar_from_samples : CVaR empirique
    - MonetaryUtility   : CVaR, entropique, moyenne
    - OCEUtility        : Optimized Certainty Equivalent (parametre appris)
"""

import torch
import torch.nn as nn


def cvar_from_samples(x: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    CVaR empirique au niveau *alpha* sur un vecteur de gains.

    Convention : x = gains (positif = bon).
    On regarde les *alpha* pires valeurs (queue gauche).
    """
    sorted_x, _ = torch.sort(x)
    n = len(sorted_x)
    k = max(1, int(alpha * n))
    tail = sorted_x[:k]
    return tail.mean()


class MonetaryUtility:
    """
    Utilite monetaire simplifiee.

    Modes :
        - ``"cvar"``      : CVaR empirique (queue gauche).
        - ``"entropic"``  : -1/lambda * log E[exp(-lambda * X)].
        - ``"mean"``      : simple moyenne.
    """

    def __init__(self, kind: str = "cvar", lmbda: float = 1.0, alpha: float = 0.05):
        self.kind = kind.lower()
        self.lmbda = lmbda
        self.alpha = alpha

    def __repr__(self):
        return f"MonetaryUtility(kind={self.kind}, lambda={self.lmbda}, alpha={self.alpha})"

    def utility(self, x: torch.Tensor) -> torch.Tensor:
        """Renvoie U(X) (scalaire)."""
        if self.kind in ("exp", "entropic"):
            lmbda = self.lmbda
            return -(1.0 / lmbda) * torch.log(torch.mean(torch.exp(-lmbda * x)))
        elif self.kind == "cvar":
            return cvar_from_samples(x, alpha=self.alpha)
        elif self.kind == "mean":
            return x.mean()
        else:
            raise ValueError(f"Unknown utility kind: {self.kind}")

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Loss a minimiser = -U(X)."""
        return -self.utility(x)


# =========================================================================
# OCEUtility — Optimized Certainty Equivalent (Buehler-style)
# =========================================================================

class OCEUtility(nn.Module):
    """
    Optimized Certainty Equivalent (OCE) avec seuil w appris.

    La formulation duale de la CVaR est :
        CVaR_alpha(X) = sup_w { w + (1/alpha) * E[ min(X - w, 0) ] }

    On parametrise w comme un nn.Parameter optimise conjointement avec
    la politique de hedging.

    Avantage : loss differentiable, converge mieux que la CVaR empirique
    basee sur un tri.
    """

    def __init__(self, alpha: float = 0.05, init_w: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.w = nn.Parameter(torch.tensor(init_w))

    def __repr__(self):
        return f"OCEUtility(alpha={self.alpha}, w={self.w.item():.4f})"

    def utility(self, x: torch.Tensor) -> torch.Tensor:
        """
        OCE(X) = w + (1/alpha) * E[min(X - w, 0)]
               = w - (1/alpha) * E[max(w - X, 0)]
        """
        shortfall = torch.clamp(self.w - x, min=0.0)
        return self.w - shortfall.mean() / self.alpha

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Loss a minimiser = -OCE(X)."""
        return -self.utility(x)
