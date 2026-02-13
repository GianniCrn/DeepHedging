"""
Environnement de hedging : construction des features, rollout et calcul du PnL.

Features d'etat (6 dimensions) :
    [0] log_moneyness   = log(S_t / K)
    [1] time_frac       = t / T
    [2] prev_delta       = delta_{t-1} (0 au premier pas)
    [3] realized_vol    = vol realisee glissante (fenetre 10)
    [4] bs_delta_hint   = delta BS analytique (feature-engineering)
    [5] d1_standardized = log(S/K) / (sigma * sqrt(T-t))
"""

import math

import torch
import torch.nn as nn

from deep_hedging.config import DeepHedgingConfig

# Indices des features (pour usage externe, ex: DeltaBSPolicy)
FEAT_LOG_MONEYNESS = 0
FEAT_TIME_FRAC = 1
FEAT_PREV_DELTA = 2
FEAT_REALIZED_VOL = 3
FEAT_BS_DELTA_HINT = 4
FEAT_D1_STD = 5
N_FEATURES = 6


def _bs_delta_numpy_torch(S, K, T_rem, r, q, sigma, is_call=True):
    """Delta BS vectorise en PyTorch (pour feature engineering)."""
    eps = 1e-8
    S = torch.clamp(S, min=eps)
    T_rem = torch.clamp(T_rem, min=eps)

    K_t = torch.tensor(K, dtype=S.dtype, device=S.device)
    r_t = torch.tensor(r, dtype=S.dtype, device=S.device)
    q_t = torch.tensor(q, dtype=S.dtype, device=S.device)
    sigma_t = torch.tensor(sigma, dtype=S.dtype, device=S.device)

    d1 = (torch.log(S / K_t) + (r_t - q_t + 0.5 * sigma_t ** 2) * T_rem) / (
        sigma_t * torch.sqrt(T_rem)
    )
    Nd1 = 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))
    disc_q = torch.exp(-q_t * T_rem)

    if is_call:
        return disc_q * Nd1
    else:
        return disc_q * (Nd1 - 1.0)


class DeepHedgingEnv:
    """
    Environnement de hedging avec features enrichies (6D).

    Hypotheses :
        - On ne hedge que le spot.
        - La politique renvoie des delta-positions (actions) cumulees en delta_t.
        - PnL  = sum_t delta_t * dS_t
        - Cost = sum_t cost_s * |action_t| * S_t
        - Gains = payoff - Cost + PnL
    """

    def __init__(self, cfg: DeepHedgingConfig):
        self.cfg = cfg
        self.mcfg = cfg.market
        self.device = cfg.device
        self.dtype = cfg.dtype

    @property
    def n_features(self) -> int:
        """Nombre de features d'etat."""
        return N_FEATURES

    # ------------------------------------------------------------------
    # Features enrichies
    # ------------------------------------------------------------------
    def build_state_features(
        self, S_batch: torch.Tensor, prev_deltas: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Construction de 6 features d'etat.

        Entree  : S_batch (B, T+1), prev_deltas (B, T) optionnel
        Sortie  : X       (B, T, 6)
        """
        m = self.mcfg
        B, T_plus_1 = S_batch.shape
        T = T_plus_1 - 1

        S_t = S_batch[:, :-1]  # (B, T)

        # [0] log-moneyness
        log_moneyness = torch.log(S_t / m.K)  # (B, T)

        # [1] fraction de temps ecoulee
        t_grid = torch.linspace(0, 1.0, T, device=S_batch.device, dtype=S_batch.dtype)
        time_frac = t_grid.unsqueeze(0).expand(B, -1)  # (B, T)

        # [2] position precedente (0 au premier pas)
        if prev_deltas is None:
            prev_delta = torch.zeros(B, T, device=S_batch.device, dtype=S_batch.dtype)
        else:
            prev_delta = prev_deltas

        # [3] vol realisee glissante (fenetre = 10 pas)
        window = min(10, T)
        log_ret = torch.log(S_batch[:, 1:] / S_batch[:, :-1])  # (B, T)
        # Padding : pour les premiers pas, on utilise ce qui est disponible
        realized_vol = torch.zeros(B, T, device=S_batch.device, dtype=S_batch.dtype)
        for t in range(T):
            start = max(0, t - window + 1)
            chunk = log_ret[:, start : t + 1]  # (B, <=window)
            if chunk.shape[1] > 1:
                realized_vol[:, t] = chunk.std(dim=1)
            else:
                realized_vol[:, t] = abs(m.sigma) * math.sqrt(self.mcfg.T / self.mcfg.n_steps)

        # [4] BS delta hint
        T_rem = m.T * (1.0 - time_frac)
        T_rem = torch.clamp(T_rem, min=1e-6)
        bs_delta = _bs_delta_numpy_torch(S_t, m.K, T_rem, m.r, m.q, m.sigma, m.is_call)

        # [5] d1 standardise = log(S/K) / (sigma * sqrt(T-t))
        d1_std = log_moneyness / (m.sigma * torch.sqrt(T_rem) + 1e-8)

        X = torch.stack(
            [log_moneyness, time_frac, prev_delta, realized_vol, bs_delta, d1_std],
            dim=-1,
        )  # (B, T, 6)
        return X

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def rollout(self, policy: nn.Module, S: torch.Tensor, payoff: torch.Tensor) -> dict:
        """
        Execute la strategie de hedging sur un batch.

        Entrees :
            S      : (B, T+1)
            payoff : (B,)

        Sorties : dict avec actions, deltas, pnl, cost, gains, payoff.
        """
        mcfg = self.mcfg
        B, T_plus_1 = S.shape
        T = T_plus_1 - 1

        # Premiere passe sans prev_delta (on met 0)
        X = self.build_state_features(S, prev_deltas=None)
        delta_actions = policy(X)  # (B, T)

        if delta_actions.shape != (B, T):
            raise ValueError(
                f"policy(X) doit renvoyer shape (B, T), obtenu {delta_actions.shape}"
            )

        deltas = torch.cumsum(delta_actions, dim=1)  # (B, T)

        # Mettre a jour prev_delta et re-faire un forward si necessaire
        # (optionnel — ici on fait un seul pass pour performance)

        dS = S[:, 1:] - S[:, :-1]  # (B, T)
        pnl = torch.sum(deltas * dS, dim=1)  # (B,)

        cost_s = torch.tensor(mcfg.cost_s, device=S.device, dtype=S.dtype)
        S_mid = S[:, :-1]
        cost = torch.sum(cost_s * torch.abs(delta_actions) * S_mid, dim=1)  # (B,)

        gains = payoff - cost + pnl

        return {
            "actions": delta_actions,
            "deltas": deltas,
            "pnl": pnl,
            "cost": cost,
            "gains": gains,
            "payoff": payoff,
        }
