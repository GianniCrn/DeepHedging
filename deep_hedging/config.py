"""
Configuration dataclasses pour le projet Deep Hedging.

Ce module est le point d'ancrage pour DEVICE et DTYPE afin d'eviter
les imports circulaires. Les autres modules importent depuis ici.
"""

from dataclasses import dataclass, field
import torch

# ---------------------------------------------------------------------------
# Device auto-detection (defini ici pour eviter les imports circulaires)
# ---------------------------------------------------------------------------
DTYPE = torch.float32

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MarketConfig:
    """Parametres de marche (Black-Scholes / Merton / Heston)."""

    S0: float = 100.0
    r: float = 0.03
    q: float = 0.0
    sigma: float = 0.2

    # Merton jump-diffusion
    lambda_jump: float = 1.0
    mu_J: float = -0.1
    sigma_J: float = 0.3
    use_jumps: bool = True

    # Heston stochastic volatility
    heston_kappa: float = 2.0       # vitesse de retour a la moyenne
    heston_theta: float = 0.04      # variance long-terme (theta = sigma^2 = 0.2^2)
    heston_xi: float = 0.3          # vol-of-vol
    heston_rho: float = -0.7        # correlation spot-vol
    heston_v0: float = 0.04         # variance initiale

    T: float = 1.0
    n_steps: int = 52  # rebalancement hebdomadaire

    K: float = 100.0
    is_call: bool = True
    payoff_type: str = "call"       # call, put, asian, straddle, lookback

    cost_s: float = 0.0002
    n_paths_train: int = 200_000
    n_paths_val: int = 50_000


@dataclass
class TrainingConfig:
    """Hyper-parametres d'entrainement."""

    n_epochs: int = 50
    batch_size: int = 10_000
    lr: float = 1e-3
    cvar_alpha: float = 0.025
    print_every: int = 5


@dataclass
class RandomConfig:
    """Seeds pour reproductibilite."""

    seed_global: int = 1234
    seed_train: int = 1234
    seed_val: int = 1234
    seed_eval_merton: int = 1234
    seed_eval_bs: int = 1234
    seed_eval_bs_merton: int = 1234
    seed_eval_heston: int = 1234


@dataclass
class DeepHedgingConfig:
    """Configuration globale regroupant marche, training, random et device."""

    market: MarketConfig = field(default_factory=MarketConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    random: RandomConfig = field(default_factory=RandomConfig)
    device: torch.device = DEVICE
    dtype: torch.dtype = DTYPE
