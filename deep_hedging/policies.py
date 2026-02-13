"""
Politiques de hedging :
    - PolicyMLP     : reseau feed-forward avec action clipping
    - PolicyLSTM    : reseau recurrent (LSTM)
    - DeltaBSPolicy : delta Black-Scholes analytique (benchmark)
"""

import math

import torch
import torch.nn as nn

from deep_hedging.config import MarketConfig, DEVICE, DTYPE
from deep_hedging.env import FEAT_LOG_MONEYNESS, FEAT_TIME_FRAC, FEAT_BS_DELTA_HINT


# =========================================================================
# PolicyMLP — reseau feed-forward avec clipping
# =========================================================================

class PolicyMLP(nn.Module):
    """
    Policy MLP (non recurrente) avec action clipping.

    Entree  : (B, T, d_in)
    Sortie  : (B, T) = delta-positions_t clampees dans [-clip, +clip]
    """

    def __init__(
        self,
        d_in: int = 6,
        d_hidden: int = 32,
        depth: int = 3,
        dropout: float = 0.0,
        clip: float = 2.0,
    ):
        super().__init__()
        self.clip = clip

        layers = []
        d = d_in
        for _ in range(depth):
            layers.append(nn.Linear(d, d_hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = d_hidden

        layers.append(nn.Linear(d_hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X : (B, T, d_in) -> (B, T)"""
        B, T, d_in = X.shape
        X_flat = X.reshape(B * T, d_in)
        out = self.net(X_flat)
        out = out.view(B, T)
        # Action clipping : positions bornees
        if self.clip is not None and self.clip > 0:
            out = torch.clamp(out, -self.clip, self.clip)
        return out


# =========================================================================
# PolicyLSTM — reseau recurrent
# =========================================================================

class PolicyLSTM(nn.Module):
    """
    Policy recurrente (LSTM).

    L'etat cache (h, c) est initialise a zero et propage sur T pas.
    Le LSTM capture les dependances temporelles (positions passees, tendances).

    Entree  : (B, T, d_in)
    Sortie  : (B, T) = delta-positions_t clampees dans [-clip, +clip]
    """

    def __init__(
        self,
        d_in: int = 6,
        d_hidden: int = 32,
        n_layers: int = 1,
        dropout: float = 0.0,
        clip: float = 2.0,
    ):
        super().__init__()
        self.clip = clip
        self.d_hidden = d_hidden
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=d_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(d_hidden, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X : (B, T, d_in) -> (B, T)"""
        # h0, c0 initialises a zero (defaut PyTorch)
        lstm_out, _ = self.lstm(X)  # (B, T, d_hidden)
        out = self.head(lstm_out).squeeze(-1)  # (B, T)
        if self.clip is not None and self.clip > 0:
            out = torch.clamp(out, -self.clip, self.clip)
        return out


# =========================================================================
# DeltaBSPolicy — benchmark analytique
# =========================================================================

class DeltaBSPolicy(nn.Module):
    """
    Policy analytique Delta Black-Scholes.

    Compatible avec les features enrichies (6D) de DeepHedgingEnv :
        - Utilise FEAT_LOG_MONEYNESS (idx 0) pour reconstruire S_t
        - Utilise FEAT_TIME_FRAC (idx 1) pour calculer T_rem
        - Utilise FEAT_BS_DELTA_HINT (idx 4) si disponible

    Sortie  : (B, T) = delta-actions (variation de delta a chaque pas)
    """

    def __init__(self, market_cfg: MarketConfig, device=DEVICE, dtype=DTYPE):
        super().__init__()
        self.mcfg = market_cfg
        self.device = device
        self.dtype = dtype

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        m = self.mcfg
        X = X.to(self.device, dtype=self.dtype)
        B, T, d_in = X.shape

        if d_in >= 5:
            # Utiliser le BS delta pre-calcule dans les features
            delta_t = X[..., FEAT_BS_DELTA_HINT]  # (B, T)
        else:
            # Fallback pour anciennes features (2D: S_rel, time)
            S_rel = X[..., 0]
            time_feat = X[..., 1]
            S_t = S_rel * m.S0
            T_rem = m.T * (1.0 - time_feat)
            T_rem = torch.clamp(T_rem, min=1e-6)
            delta_t = self._bs_delta_torch(S_t, m.K, T_rem, m.r, m.q, m.sigma, m.is_call)

        # Convertir delta absolu en delta-actions (variation)
        delta_prev = torch.zeros_like(delta_t)
        delta_prev[:, 1:] = delta_t[:, :-1]
        delta_actions = delta_t - delta_prev

        return delta_actions

    @staticmethod
    def _bs_delta_torch(S, K, T, r, q, sigma, is_call=True):
        """Delta Black-Scholes en PyTorch pour call/put europeen."""
        eps = 1e-8
        S = torch.clamp(S, min=eps)
        T = torch.clamp(T, min=eps)

        r_t = torch.tensor(r, dtype=S.dtype, device=S.device)
        q_t = torch.tensor(q, dtype=S.dtype, device=S.device)
        sigma_t = torch.tensor(sigma, dtype=S.dtype, device=S.device)
        K_t = torch.tensor(K, dtype=S.dtype, device=S.device)

        d1 = (torch.log(S / K_t) + (r_t - q_t + 0.5 * sigma_t ** 2) * T) / (
            sigma_t * torch.sqrt(T)
        )
        Nd1 = 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))
        disc_q = torch.exp(-q_t * T)

        if is_call:
            delta = disc_q * Nd1
        else:
            delta = disc_q * (Nd1 - 1.0)

        return delta
