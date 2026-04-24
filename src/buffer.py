# AI-generated: Claude, 2026-04-21
# Reward preprocessing is adapted from the official IQL repository's D4RL data
# handling (`train_offline.py` and `dataset_utils.py`).
"""Replay buffer utilities for D4RL offline datasets."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import Tensor


@dataclass
class ReplayBatch:
    """Container for a sampled batch from replay data."""

    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    terminals: Tensor

class GPUReplayBuffer:
    """Replay buffer that stores full D4RL dataset as tensors on a target device."""

    def __init__(
        self,
        env_name: str,
        device: torch.device,
        storage_device: torch.device | None = None,
    ) -> None:
        import d4rl  # pylint: disable=import-outside-toplevel
        import gym  # pylint: disable=import-outside-toplevel

        self.env_name = env_name
        self.device = device
        self.storage_device = storage_device if storage_device is not None else device

        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
        rewards = dataset["rewards"].astype(np.float32)
        dones_float = self._compute_dones_float(dataset)

        dataset["actions"] = np.clip(dataset["actions"].astype(np.float32), -1.0 + 1e-5, 1.0 - 1e-5)

        if "antmaze" in env_name:
            rewards = rewards - 1.0
        elif any(name in env_name for name in ("halfcheetah", "walker2d", "hopper")):
            rewards = self._normalize_rewards(rewards, dones_float)

        self.observations = self._to_tensor(dataset["observations"])
        self.actions = self._to_tensor(dataset["actions"])
        self.rewards = self._to_tensor(rewards).unsqueeze(-1)
        self.next_observations = self._to_tensor(dataset["next_observations"])
        self.terminals = self._to_tensor(dataset["terminals"]).unsqueeze(-1)

        self.size = self.observations.shape[0]
        env.close()

    def _compute_dones_float(self, dataset: dict[str, np.ndarray]) -> np.ndarray:
        dones_float = np.zeros_like(dataset["rewards"], dtype=np.float32)
        for index in range(len(dones_float) - 1):
            observation_gap = np.linalg.norm(
                dataset["observations"][index + 1] - dataset["next_observations"][index]
            )
            if observation_gap > 1e-6 or dataset["terminals"][index] == 1.0:
                dones_float[index] = 1.0
        dones_float[-1] = 1.0
        return dones_float

    def _normalize_rewards(self, rewards: np.ndarray, dones_float: np.ndarray) -> np.ndarray:
        returns: list[float] = []
        episode_return = 0.0
        for reward, done in zip(rewards, dones_float):
            episode_return += float(reward)
            if done == 1.0:
                returns.append(episode_return)
                episode_return = 0.0

        if not returns:
            return rewards

        reward_range = max(returns) - min(returns)
        if reward_range <= 1e-8:
            return rewards
        return rewards / reward_range * 1000.0

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.storage_device)

    def _batch_to_training_device(self, batch: ReplayBatch) -> ReplayBatch:
        if self.storage_device == self.device:
            return batch
        return ReplayBatch(
            observations=batch.observations.to(self.device, non_blocking=True),
            actions=batch.actions.to(self.device, non_blocking=True),
            rewards=batch.rewards.to(self.device, non_blocking=True),
            next_observations=batch.next_observations.to(self.device, non_blocking=True),
            terminals=batch.terminals.to(self.device, non_blocking=True),
        )

    def sample(self, batch_size: int) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        indices = torch.randint(0, self.size, (batch_size,), device=self.storage_device)
        batch = ReplayBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            terminals=self.terminals[indices],
        )
        return self._batch_to_training_device(batch)

    def sample_with_throughput(self, batch_size: int) -> tuple[ReplayBatch, float]:
        """Sample a batch and report throughput in samples/sec (CUDA timing)."""
        if self.device.type != "cuda":
            start = time.perf_counter()
            batch = self.sample(batch_size)
            elapsed = time.perf_counter() - start
            throughput = float(batch_size / max(elapsed, 1e-8))
            return batch, throughput

        if self.storage_device.type != "cuda":
            start = time.perf_counter()
            batch = self.sample(batch_size)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            throughput = float(batch_size / max(elapsed, 1e-8))
            return batch, throughput

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        batch = self.sample(batch_size)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        throughput = float(batch_size / max(elapsed_ms / 1000.0, 1e-8))
        return batch, throughput

    @property
    def state_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[-1])

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "terminals": self.terminals,
        }
