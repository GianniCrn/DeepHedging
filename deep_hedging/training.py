"""
Boucle d'entrainement : generation de datasets, loss batch, training
avec early stopping et learning rate scheduler.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deep_hedging.config import DeepHedgingConfig
from deep_hedging.worlds import SimpleWorldBS, SimpleWorldMerton
from deep_hedging.env import DeepHedgingEnv
from deep_hedging.losses import MonetaryUtility


# ---------------------------------------------------------------------------
# Generation de datasets
# ---------------------------------------------------------------------------

def generate_datasets(cfg: DeepHedgingConfig):
    """
    Genere les datasets train / val a partir de la configuration.

    Retourne un dict avec S_train, payoff_train, S_val, payoff_val (tensors).
    """
    if cfg.market.use_jumps:
        world = SimpleWorldMerton(cfg.market)
    else:
        world = SimpleWorldBS(cfg.market)

    data_train = world.simulate_paths(
        cfg.market.n_paths_train, seed=cfg.random.seed_train
    )
    S_train = torch.tensor(data_train["S"], dtype=cfg.dtype, device=cfg.device)
    payoff_train = torch.tensor(data_train["payoff"], dtype=cfg.dtype, device=cfg.device)

    data_val = world.simulate_paths(
        cfg.market.n_paths_val, seed=cfg.random.seed_val
    )
    S_val = torch.tensor(data_val["S"], dtype=cfg.dtype, device=cfg.device)
    payoff_val = torch.tensor(data_val["payoff"], dtype=cfg.dtype, device=cfg.device)

    return {
        "S_train": S_train,
        "payoff_train": payoff_train,
        "S_val": S_val,
        "payoff_val": payoff_val,
    }


# ---------------------------------------------------------------------------
# Loss batch
# ---------------------------------------------------------------------------

def deep_hedging_loss_batch(
    policy: nn.Module,
    env: DeepHedgingEnv,
    S_batch: torch.Tensor,
    payoff_batch: torch.Tensor,
    utility: MonetaryUtility,
) -> torch.Tensor:
    """Calcule la loss sur un batch."""
    out = env.rollout(policy, S_batch, payoff_batch)
    gains = out["gains"]
    actions = out["actions"]

    base_loss = utility.loss(gains)

    # Regularisation sur la taille des actions
    lambda_pen = 1e-5
    penalty = lambda_pen * (actions ** 2).mean()

    return base_loss + penalty


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_deep_hedging(
    cfg: DeepHedgingConfig,
    policy: nn.Module,
    utility: Optional[MonetaryUtility] = None,
    patience: int = 5,
    min_delta: float = 1e-3,
    use_scheduler: bool = True,
) -> dict:
    """
    Entraine la policy avec DataLoader + early stopping + LR scheduler.

    Retourne un dict avec 'policy', 'history' (train_loss, val_loss, lr).
    """
    if utility is None:
        utility = MonetaryUtility(kind="cvar", alpha=cfg.training.cvar_alpha)

    policy = policy.to(cfg.device)
    env = DeepHedgingEnv(cfg)

    # ---------- Donnees ----------
    datasets = generate_datasets(cfg)
    S_train = datasets["S_train"]
    payoff_train = datasets["payoff_train"]
    S_val = datasets["S_val"]
    payoff_val = datasets["payoff_val"]

    train_ds = TensorDataset(S_train, payoff_train)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)

    # ---------- Optimiseur + Scheduler ----------
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.lr)

    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6,
        )

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    # ---------- Boucle ----------
    for epoch in range(1, cfg.training.n_epochs + 1):
        policy.train()
        epoch_losses = []

        for S_b, p_b in train_loader:
            optimizer.zero_grad()
            loss = deep_hedging_loss_batch(policy, env, S_b, p_b, utility)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = sum(epoch_losses) / len(epoch_losses)
        history["train_loss"].append(train_loss)

        # Validation
        policy.eval()
        with torch.no_grad():
            val_loss = deep_hedging_loss_batch(
                policy, env, S_val, payoff_val, utility
            ).item()
        history["val_loss"].append(val_loss)

        # Learning rate scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
        if scheduler is not None:
            scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(policy.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch % cfg.training.print_every == 0:
            print(
                f"Epoch {epoch:3d}/{cfg.training.n_epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"lr={current_lr:.2e} | patience={wait}/{patience}"
            )

        if wait >= patience:
            print(f"Early stopping a l'epoch {epoch}.")
            break

    # Restaurer le meilleur modele
    if best_state is not None:
        policy.load_state_dict(best_state)

    return {"policy": policy, "history": history}
