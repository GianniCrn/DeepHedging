"""
Deep Hedging — Package modulaire
=================================
Strategie de couverture par reseaux de neurones sous Black-Scholes,
Merton Jump-Diffusion et Heston Stochastic Volatility.
"""

# ---------------------------------------------------------------------------
# Device & dtype (re-exported from config to keep backward compatibility)
# ---------------------------------------------------------------------------
from deep_hedging.config import DEVICE, DTYPE

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
from deep_hedging.config import MarketConfig, TrainingConfig, RandomConfig, DeepHedgingConfig
from deep_hedging.worlds import SimpleWorldBS, SimpleWorldMerton, SimpleWorldHeston
from deep_hedging.env import (
    DeepHedgingEnv,
    N_FEATURES,
    FEAT_LOG_MONEYNESS,
    FEAT_TIME_FRAC,
    FEAT_PREV_DELTA,
    FEAT_REALIZED_VOL,
    FEAT_BS_DELTA_HINT,
    FEAT_D1_STD,
)
from deep_hedging.policies import PolicyMLP, PolicyLSTM, DeltaBSPolicy
from deep_hedging.losses import MonetaryUtility, OCEUtility, cvar_from_samples
from deep_hedging.training import generate_datasets, deep_hedging_loss_batch, train_deep_hedging
from deep_hedging.evaluation import (
    bs_delta_call,
    evaluate_strategies_env_world,
    build_comparison_table,
)
from deep_hedging.risk_metrics import RiskMetrics
from deep_hedging.plotting import (
    plot_training_history,
    plot_gains_hist,
    plot_payoff_vs_gains,
    plot_simulated_paths,
    traffic_light_style,
)
from deep_hedging.utils import (
    weighted_mean,
    weighted_var,
    weighted_std,
    compute_rmse,
    compute_max_drawdown,
)

__all__ = [
    # constants
    "DEVICE", "DTYPE",
    # config
    "MarketConfig", "TrainingConfig", "RandomConfig", "DeepHedgingConfig",
    # worlds
    "SimpleWorldBS", "SimpleWorldMerton", "SimpleWorldHeston",
    # env
    "DeepHedgingEnv", "N_FEATURES",
    "FEAT_LOG_MONEYNESS", "FEAT_TIME_FRAC", "FEAT_PREV_DELTA",
    "FEAT_REALIZED_VOL", "FEAT_BS_DELTA_HINT", "FEAT_D1_STD",
    # policies
    "PolicyMLP", "PolicyLSTM", "DeltaBSPolicy",
    # losses
    "MonetaryUtility", "OCEUtility", "cvar_from_samples",
    # training
    "generate_datasets", "deep_hedging_loss_batch", "train_deep_hedging",
    # evaluation
    "bs_delta_call", "evaluate_strategies_env_world", "build_comparison_table",
    # risk
    "RiskMetrics",
    # plotting
    "plot_training_history", "plot_gains_hist", "plot_payoff_vs_gains",
    "plot_simulated_paths", "traffic_light_style",
    # utils
    "weighted_mean", "weighted_var", "weighted_std",
    "compute_rmse", "compute_max_drawdown",
]
