# AI-generated: Claude, 2026-04-21
# Portions of `TanhNormal` are adapted from RLKit's distribution helpers, while
# the default `GaussianPolicy` behavior follows the official IQL implementation
# pattern: bounded means, Gaussian log-probabilities, and clipped actions.
"""Neural network modules for Implicit Q-Learning (IQL)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Independent, Normal


LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -10.0


class TanhNormal:
    """Distribution of `tanh(z)` where `z ~ Normal(mean, std)`.

    This mirrors the distribution used by RLKit's IQL policy and provides
    numerically stable log-probabilities for squashed actions.
    """

    def __init__(self, mean: Tensor, std: Tensor, epsilon: float = 1e-6) -> None:
        self.normal_mean = mean
        self.normal_std = std
        self.normal = Normal(mean, std)
        self.epsilon = epsilon

    def _log_prob_from_pre_tanh(self, pre_tanh_value: Tensor) -> Tensor:
        log_prob = self.normal.log_prob(pre_tanh_value)
        log_two = torch.log(torch.tensor(2.0, device=pre_tanh_value.device, dtype=pre_tanh_value.dtype))
        correction = -2.0 * (log_two - pre_tanh_value - torch.nn.functional.softplus(-2.0 * pre_tanh_value))
        return log_prob + correction

    def log_prob(self, value: Tensor, pre_tanh_value: Tensor | None = None) -> Tensor:
        if pre_tanh_value is None:
            value = torch.clamp(value, -1.0 + self.epsilon, 1.0 - self.epsilon)
            pre_tanh_value = torch.atanh(value)
        return self._log_prob_from_pre_tanh(pre_tanh_value)

    def rsample_with_pretanh(self) -> tuple[Tensor, Tensor]:
        pre_tanh_value = self.normal.rsample()
        return torch.tanh(pre_tanh_value), pre_tanh_value

    def rsample_and_logprob(self) -> tuple[Tensor, Tensor]:
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_prob = self.log_prob(value, pre_tanh_value)
        return value, log_prob

    def sample(self) -> Tensor:
        value, _ = self.rsample_with_pretanh()
        return value.detach()

    @property
    def mean(self) -> Tensor:
        return torch.tanh(self.normal_mean)

    @property
    def stddev(self) -> Tensor:
        return self.normal_std


def _build_mlp(input_dim: int, output_dim: int, hidden_dim: int, n_hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(n_hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def _initialize_last_layer(linear: nn.Linear, init_w: float = 1e-3) -> None:
    linear.weight.data.uniform_(-init_w, init_w)
    linear.bias.data.uniform_(-init_w, init_w)


# AI-generated: Claude, 2026-04-21
class ValueNetwork(nn.Module):
    """State-value function V(s)."""

    def __init__(self, state_dim: int, hidden_dim: int, n_hidden_layers: int) -> None:
        super().__init__()
        self.net = _build_mlp(
            input_dim=state_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.net(states)


# AI-generated: Claude, 2026-04-21
class QNetwork(nn.Module):
    """Action-value function Q(s, a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, n_hidden_layers: int) -> None:
        super().__init__()
        self.net = _build_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        return self.net(torch.cat([states, actions], dim=-1))


# AI-generated: Claude, 2026-04-21
class TwinQNetwork(nn.Module):
    """Twin critics for clipped double-Q training."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, n_hidden_layers: int) -> None:
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim, n_hidden_layers)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim, n_hidden_layers)

    def forward(self, states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(states, actions), self.q2(states, actions)

    def min_q(self, states: Tensor, actions: Tensor) -> Tensor:
        q1, q2 = self.forward(states, actions)
        return torch.minimum(q1, q2)


# AI-generated: Claude, 2026-04-21
class GaussianPolicy(nn.Module):
    """Gaussian actor used for IQL advantage-weighted regression."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        log_std_bounds: Sequence[float] = (-6.0, 0.0),
        init_w: float = 1e-3,
        tanh_squash_distribution: bool = False,
    ) -> None:
        super().__init__()
        if len(log_std_bounds) != 2:
            raise ValueError("log_std_bounds must contain (min, max)")

        self.log_std_min = float(log_std_bounds[0])
        self.log_std_max = float(log_std_bounds[1])
        self.tanh_squash_distribution = tanh_squash_distribution

        self.backbone = _build_mlp(
            input_dim=state_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        _initialize_last_layer(self.mu_layer, init_w=init_w)
        _initialize_last_layer(self.log_std_layer, init_w=init_w)

    def _distribution(self, states: Tensor) -> TanhNormal | Independent:
        features = self.backbone(states)
        raw_mu = self.mu_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(
            log_std,
            max=min(self.log_std_max, LOG_SIG_MAX),
            min=max(self.log_std_min, LOG_SIG_MIN),
        )
        std = log_std.exp()
        if self.tanh_squash_distribution:
            return TanhNormal(raw_mu, std)

        mean = torch.tanh(raw_mu)
        return Independent(Normal(mean, std), 1)

    def forward(self, states: Tensor) -> TanhNormal | Independent:
        return self._distribution(states)

    def sample(self, states: Tensor) -> tuple[Tensor, Tensor]:
        dist = self._distribution(states)
        if self.tanh_squash_distribution:
            action, log_prob = dist.rsample_and_logprob()
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        else:
            raw_action = dist.rsample()
            action = torch.clamp(raw_action, -1.0, 1.0)
            log_prob = dist.log_prob(raw_action).unsqueeze(-1)
        return action, log_prob

    def log_prob(self, states: Tensor, actions: Tensor) -> Tensor:
        dist = self._distribution(states)
        log_prob = dist.log_prob(actions)
        if log_prob.dim() == 1:
            return log_prob.unsqueeze(-1)
        return log_prob.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def act(self, state: Tensor, deterministic: bool = True) -> Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        dist = self._distribution(state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        return torch.clamp(action, -1.0, 1.0)
