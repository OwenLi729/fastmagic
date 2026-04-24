# AI-generated: Claude, 2026-04-21
# The actor-weighting helper follows the original IQL paper / official IQL
# implementation convention: weights are computed as `exp(beta * adv)` and clipped.
"""Loss functions for Implicit Q-Learning (IQL)."""

from __future__ import annotations

import torch
from torch import Tensor


def expectile_loss(diff: Tensor, tau: float) -> Tensor:
    """Compute vectorized expectile regression loss.

    Args:
        diff: Difference tensor defined as Q(s, a) - V(s).
        tau: Expectile coefficient in (0, 1), commonly 0.7 to 0.9.

    Returns:
        Scalar tensor representing mean expectile loss.
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")

    tau_tensor = torch.as_tensor(tau, device=diff.device, dtype=diff.dtype)
    weights = torch.where(diff > 0, tau_tensor, 1.0 - tau_tensor)
    return (weights * diff.pow(2)).mean()


def awr_policy_loss(
    advantages: Tensor,
    log_prob: Tensor,
    beta: float,
    max_weight: float = 100.0,
) -> Tensor:
    """Compute vectorized advantage-weighted behavior cloning loss.

    Args:
        advantages: Advantage tensor A(s, a) = Q(s, a) - V(s).
        log_prob: Log-probabilities log pi(a | s) from current policy.
        beta: Inverse-temperature parameter from the IQL paper.
        max_weight: Maximum clipping value for exp(beta * advantage).

    Returns:
        Scalar tensor representing mean negative weighted log-likelihood.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if max_weight <= 0.0:
        raise ValueError(f"max_weight must be > 0, got {max_weight}")

    weights = awr_weights(advantages=advantages, beta=beta, max_weight=max_weight)
    return -(weights * log_prob).mean()


def awr_weights(advantages: Tensor, beta: float, max_weight: float = 100.0) -> Tensor:
    """Compute detached advantage weights for the actor update.

    This follows the original IQL paper and official implementation. RLKit's
    `beta` naming is reciprocal to this convention.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if max_weight <= 0.0:
        raise ValueError(f"max_weight must be > 0, got {max_weight}")

    beta_tensor = torch.as_tensor(beta, device=advantages.device, dtype=advantages.dtype)
    return torch.exp(beta_tensor * advantages.detach()).clamp(max=max_weight)
