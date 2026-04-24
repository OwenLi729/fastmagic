# AI-generated: Claude, 2026-04-21
"""Evaluation helpers for IQL policies on D4RL tasks."""

from __future__ import annotations

from typing import Any

import gym
import numpy as np
import torch
from torch import Tensor

from networks import GaussianPolicy
from utils import CudaEventTimer


def _reset_env(env: gym.Env) -> np.ndarray:
    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        observation, _ = reset_output
        return observation
    return reset_output


def _step_env(env: gym.Env, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
    step_output = env.step(action)
    if len(step_output) == 5:
        next_observation, reward, terminated, truncated, _ = step_output
        done = bool(terminated or truncated)
        return next_observation, float(reward), done

    next_observation, reward, done, _ = step_output
    return next_observation, float(reward), bool(done)


def evaluate_policy(
    policy: GaussianPolicy,
    env_name: str,
    device: torch.device,
    n_episodes: int = 5,
) -> float:
    """Run policy rollouts and return mean D4RL normalized score."""
    env = gym.make(env_name)
    episode_returns: list[float] = []

    policy.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs = _reset_env(env)
            done = False
            episode_return = 0.0

            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action = policy.act(obs_tensor, deterministic=True)
                action_np = action.squeeze(0).cpu().numpy()
                obs, reward, done = _step_env(env, action_np)
                episode_return += reward

            episode_returns.append(episode_return)

    raw_score = float(np.mean(episode_returns))
    if hasattr(env, "get_normalized_score"):
        normalized = float(env.get_normalized_score(raw_score) * 100.0)
    else:
        normalized = raw_score

    policy.train()
    return normalized


def measure_inference_time_ms(
    policy: GaussianPolicy,
    states: Tensor,
    device: torch.device,
    n_repeats: int = 100,
) -> float:
    """Measure mean deterministic inference latency in milliseconds."""
    timer = CudaEventTimer(enabled=(device.type == "cuda"))

    policy.eval()
    with torch.no_grad():
        timer.start()
        for _ in range(n_repeats):
            _ = policy.act(states, deterministic=True)
        elapsed_ms = timer.stop()
    policy.train()

    if elapsed_ms == 0.0:
        return 0.0
    return float(elapsed_ms / n_repeats)
