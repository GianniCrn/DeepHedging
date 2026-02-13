"""
Simulateurs de trajectoires :
    - SimpleWorldBS      : Black-Scholes (GBM)
    - SimpleWorldMerton  : Merton Jump-Diffusion
    - SimpleWorldHeston  : Heston Stochastic Volatility
"""

import math
import numpy as np

from deep_hedging.config import MarketConfig


# =========================================================================
# Calcul du payoff generique
# =========================================================================

def _compute_payoff(S: np.ndarray, mcfg: MarketConfig) -> np.ndarray:
    """
    Calcule le payoff selon le type configurer dans mcfg.payoff_type.

    Payoffs supportes :
        - call      : max(S_T - K, 0)
        - put       : max(K - S_T, 0)
        - straddle  : |S_T - K|
        - asian     : max(S_mean - K, 0)  ou max(K - S_mean, 0)
        - lookback  : max(S_max - K, 0)   (call lookback sur max)
    """
    ptype = mcfg.payoff_type.lower()

    if ptype == "call":
        return np.maximum(S[:, -1] - mcfg.K, 0.0)
    elif ptype == "put":
        return np.maximum(mcfg.K - S[:, -1], 0.0)
    elif ptype == "straddle":
        return np.abs(S[:, -1] - mcfg.K)
    elif ptype == "asian":
        S_mean = S.mean(axis=1)
        if mcfg.is_call:
            return np.maximum(S_mean - mcfg.K, 0.0)
        else:
            return np.maximum(mcfg.K - S_mean, 0.0)
    elif ptype == "lookback":
        S_max = S.max(axis=1)
        return np.maximum(S_max - mcfg.K, 0.0)
    else:
        raise ValueError(f"Payoff type inconnu: {ptype}")


# =========================================================================
# SimpleWorldBS — Black-Scholes
# =========================================================================

class SimpleWorldBS:
    """
    Monde Black-Scholes :
        dS_t = S_t * ((r - q) dt + sigma * dW_t)
    """

    def __init__(self, mcfg: MarketConfig):
        self.cfg = mcfg
        self.dt = mcfg.T / mcfg.n_steps
        self.t_grid = np.linspace(0, mcfg.T, mcfg.n_steps + 1)

    def simulate_paths(self, n_paths: int, seed: int = 1234) -> dict:
        rng = np.random.default_rng(seed)
        m = self.cfg
        dt = self.dt

        Z = rng.standard_normal(size=(n_paths, m.n_steps))

        S = np.zeros((n_paths, m.n_steps + 1), dtype=np.float32)
        S[:, 0] = m.S0

        drift = (m.r - m.q - 0.5 * m.sigma ** 2) * dt
        vol = m.sigma * math.sqrt(dt)

        for t in range(m.n_steps):
            S[:, t + 1] = S[:, t] * np.exp(drift + vol * Z[:, t])

        dS = S[:, 1:] - S[:, :-1]
        payoff = _compute_payoff(S, m)

        return {"S": S, "dS": dS, "payoff": payoff, "t_grid": self.t_grid}


# =========================================================================
# SimpleWorldMerton — Merton Jump-Diffusion
# =========================================================================

class SimpleWorldMerton:
    """
    Monde Merton Jump-Diffusion :
        dS_t / S_{t-} = (r - q - lambda*kappa) dt + sigma dW_t + dJ_t

    avec J_t = sum(e^{Y_i} - 1), Y_i ~ N(mu_J, sigma_J^2).
    """

    def __init__(self, mcfg: MarketConfig):
        self.cfg = mcfg
        self.dt = mcfg.T / mcfg.n_steps
        self.t_grid = np.linspace(0, mcfg.T, mcfg.n_steps + 1)

        self.kappa = math.exp(mcfg.mu_J + 0.5 * mcfg.sigma_J ** 2) - 1.0
        self.drift = (
            mcfg.r - mcfg.q - mcfg.lambda_jump * self.kappa - 0.5 * mcfg.sigma ** 2
        )

    def simulate_paths(self, n_paths: int, seed: int = 1234) -> dict:
        rng = np.random.default_rng(seed)
        m = self.cfg
        dt = self.dt

        S = np.zeros((n_paths, m.n_steps + 1), dtype=np.float64)
        S[:, 0] = m.S0
        vol = m.sigma * math.sqrt(dt)

        for t in range(m.n_steps):
            Z = rng.standard_normal(size=n_paths)

            N_jump = rng.poisson(m.lambda_jump * dt, size=n_paths)
            has_jump = N_jump > 0

            logJ = np.zeros(n_paths, dtype=np.float64)
            if np.any(has_jump):
                n_tot = N_jump[has_jump].sum()
                Y = rng.normal(loc=m.mu_J, scale=m.sigma_J, size=n_tot)

                idx = np.cumsum(np.concatenate([[0], N_jump[has_jump]]))
                for k in range(len(idx) - 1):
                    a, b = idx[k], idx[k + 1]
                    logJ[np.where(has_jump)[0][k]] = Y[a:b].sum()

            dlogS = self.drift * dt + vol * Z + logJ
            S[:, t + 1] = S[:, t] * np.exp(dlogS)

        dS = S[:, 1:] - S[:, :-1]
        payoff = _compute_payoff(S, m)

        return {"S": S, "dS": dS, "payoff": payoff, "t_grid": self.t_grid}


# =========================================================================
# SimpleWorldHeston — Stochastic Volatility
# =========================================================================

class SimpleWorldHeston:
    """
    Monde Heston a volatilite stochastique :
        dS_t = S_t * sqrt(v_t) * dW_t^S
        dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_t^v

    avec corr(dW^S, dW^v) = rho.
    Discretisation d'Euler avec troncature v_t >= 0.
    """

    def __init__(self, mcfg: MarketConfig):
        self.cfg = mcfg
        self.dt = mcfg.T / mcfg.n_steps
        self.t_grid = np.linspace(0, mcfg.T, mcfg.n_steps + 1)

        # Verification condition de Feller
        feller = 2 * mcfg.heston_kappa * mcfg.heston_theta
        xi2 = mcfg.heston_xi ** 2
        if feller < xi2:
            import warnings
            warnings.warn(
                f"Condition de Feller non satisfaite : "
                f"2*kappa*theta={feller:.4f} < xi^2={xi2:.4f}. "
                f"La variance peut atteindre zero."
            )

    def simulate_paths(self, n_paths: int, seed: int = 1234) -> dict:
        rng = np.random.default_rng(seed)
        m = self.cfg
        dt = self.dt
        kappa = m.heston_kappa
        theta = m.heston_theta
        xi = m.heston_xi
        rho = m.heston_rho
        v0 = m.heston_v0

        S = np.zeros((n_paths, m.n_steps + 1), dtype=np.float64)
        v = np.zeros((n_paths, m.n_steps + 1), dtype=np.float64)
        S[:, 0] = m.S0
        v[:, 0] = v0

        for t in range(m.n_steps):
            Z1 = rng.standard_normal(size=n_paths)
            Z2 = rng.standard_normal(size=n_paths)
            # Browniens correles
            W_S = Z1
            W_v = rho * Z1 + math.sqrt(1.0 - rho ** 2) * Z2

            v_t = np.maximum(v[:, t], 0.0)
            sqrt_v = np.sqrt(v_t)

            # Euler pour v
            v[:, t + 1] = (
                v_t + kappa * (theta - v_t) * dt + xi * sqrt_v * math.sqrt(dt) * W_v
            )
            v[:, t + 1] = np.maximum(v[:, t + 1], 0.0)  # troncature

            # Euler pour S (drift risque-neutre)
            dlogS = (m.r - m.q - 0.5 * v_t) * dt + sqrt_v * math.sqrt(dt) * W_S
            S[:, t + 1] = S[:, t] * np.exp(dlogS)

        dS = S[:, 1:] - S[:, :-1]
        payoff = _compute_payoff(S, m)

        return {
            "S": S.astype(np.float32),
            "dS": dS.astype(np.float32),
            "payoff": payoff.astype(np.float32),
            "t_grid": self.t_grid,
            "v": v.astype(np.float32),  # trajectoires de variance
        }
