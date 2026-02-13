"""
Evaluation : comparaison Deep Hedging vs Delta Hedging + No Hedge + tableau de synthese.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from deep_hedging.config import DeepHedgingConfig
from deep_hedging.env import DeepHedgingEnv
from deep_hedging.policies import DeltaBSPolicy
from deep_hedging.losses import MonetaryUtility
from deep_hedging.utils import compute_rmse, compute_max_drawdown
from deep_hedging.config import DEVICE, DTYPE


# ---------------------------------------------------------------------------
# Delta BS analytique (numpy, pour usage standalone)
# ---------------------------------------------------------------------------

def bs_delta_call(S, K, T, r, q, sigma):
    """Delta BS d'un call europeen (numpy)."""
    S = np.asarray(S, dtype=float)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# ---------------------------------------------------------------------------
# Evaluation multi-strategies
# ---------------------------------------------------------------------------

def evaluate_strategies_env_world(
    cfg: DeepHedgingConfig,
    policy_trained,
    world_class,
    n_paths_eval: int = 20_000,
    seed_eval: int = None,
) -> dict:
    """
    Compare Deep Hedging (policy_trained) vs Delta BS analytique vs No Hedge
    dans un monde donne (SimpleWorldBS, SimpleWorldMerton, ou SimpleWorldHeston).

    Retourne un dict avec gains, pnl, cost, metriques pour chaque strategie.
    """
    if seed_eval is None:
        seed_eval = cfg.random.seed_eval_merton

    world = world_class(cfg.market)
    data = world.simulate_paths(n_paths_eval, seed=seed_eval)

    S = torch.tensor(data["S"], dtype=cfg.dtype, device=cfg.device)
    payoff = torch.tensor(data["payoff"], dtype=cfg.dtype, device=cfg.device)

    env = DeepHedgingEnv(cfg)

    # Deep Hedging
    policy_trained = policy_trained.to(cfg.device)
    policy_trained.eval()
    with torch.no_grad():
        out_deep = env.rollout(policy_trained, S, payoff)
        gains_deep = out_deep["gains"].detach().cpu()
        pnl_deep = out_deep["pnl"].detach().cpu()
        cost_deep = out_deep["cost"].detach().cpu()

    # Delta BS
    delta_policy = DeltaBSPolicy(cfg.market, device=cfg.device, dtype=cfg.dtype).to(cfg.device)
    delta_policy.eval()
    with torch.no_grad():
        out_delta = env.rollout(delta_policy, S, payoff)
        gains_delta = out_delta["gains"].detach().cpu()
        pnl_delta = out_delta["pnl"].detach().cpu()
        cost_delta = out_delta["cost"].detach().cpu()

    # No Hedge (payoff brut, aucun trading)
    gains_no_hedge = payoff.detach().cpu()
    pnl_no_hedge = torch.zeros_like(gains_no_hedge)
    cost_no_hedge = torch.zeros_like(gains_no_hedge)

    # Metriques
    util = MonetaryUtility(kind="cvar", alpha=cfg.training.cvar_alpha)
    cvar_delta = util.utility(gains_delta).item()
    cvar_deep = util.utility(gains_deep).item()
    cvar_no_hedge = util.utility(gains_no_hedge).item()

    return {
        # Deep Hedging
        "gains_deep": gains_deep.numpy(),
        "pnl_deep": pnl_deep.numpy(),
        "cost_deep": cost_deep.numpy(),
        "cvar_deep": cvar_deep,
        "std_deep": gains_deep.std().item(),
        # Delta BS
        "gains_delta": gains_delta.numpy(),
        "pnl_delta": pnl_delta.numpy(),
        "cost_delta": cost_delta.numpy(),
        "cvar_delta": cvar_delta,
        "std_delta": gains_delta.std().item(),
        # No Hedge
        "gains_no_hedge": gains_no_hedge.numpy(),
        "pnl_no_hedge": pnl_no_hedge.numpy(),
        "cost_no_hedge": cost_no_hedge.numpy(),
        "cvar_no_hedge": cvar_no_hedge,
        "std_no_hedge": gains_no_hedge.std().item(),
        # Common
        "payoff": data["payoff"],
    }


# ---------------------------------------------------------------------------
# Tableau comparatif enrichi
# ---------------------------------------------------------------------------

def build_comparison_table(cfg: DeepHedgingConfig, eval_res: dict) -> pd.DataFrame:
    """
    Construit un DataFrame comparant No Hedge, Delta Hedging et Deep Hedging.

    Metriques :
        - RMSE (vs payoff)
        - Mean Gains, Std Gains
        - Mean PnL
        - CVaR
        - Max Drawdown
        - Mean Cost
        - Hedging Error = std(Gains - Payoff)
        - Sharpe PnL = mean(PnL) / std(PnL)
        - Cost / Payoff = mean(Cost) / mean(|Payoff|)
    """
    payoff = eval_res["payoff"]

    def _row(gains, label):
        pnl = eval_res[f"pnl_{label}"]
        cost = eval_res[f"cost_{label}"]
        hedging_error = float(np.std(gains - payoff))
        pnl_std = float(np.std(pnl))
        pnl_mean = float(np.mean(pnl))
        sharpe_pnl = pnl_mean / pnl_std if pnl_std > 1e-10 else 0.0
        mean_payoff_abs = float(np.mean(np.abs(payoff)))
        cost_over_payoff = float(np.mean(cost)) / mean_payoff_abs if mean_payoff_abs > 1e-10 else 0.0

        return {
            "RMSE (vs payoff)": compute_rmse(gains, payoff),
            "Mean Gains": float(np.mean(gains)),
            "Std Gains": float(np.std(gains)),
            "Mean PnL": pnl_mean,
            "CVaR": eval_res[f"cvar_{label}"],
            "Max Drawdown": compute_max_drawdown(gains),
            "Mean Cost": float(np.mean(cost)),
            "Hedging Error": hedging_error,
            "Sharpe PnL": sharpe_pnl,
            "Cost / Payoff": cost_over_payoff,
        }

    df = pd.DataFrame(
        [
            _row(eval_res["gains_no_hedge"], "no_hedge"),
            _row(eval_res["gains_delta"], "delta"),
            _row(eval_res["gains_deep"], "deep"),
        ],
        index=["No Hedge", "Delta Hedging", "Deep Hedging"],
    )
    return df
