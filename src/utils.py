# AI-generated: Claude, 2026-04-21
"""Utility functions for timing, checkpointing, and serialization."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class CudaEventTimer:
    """Simple CUDA event timer for accurate GPU timing in milliseconds."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and torch.cuda.is_available()
        self._start = torch.cuda.Event(enable_timing=True) if self.enabled else None
        self._end = torch.cuda.Event(enable_timing=True) if self.enabled else None

    def start(self) -> None:
        if self.enabled and self._start is not None:
            self._start.record()

    def stop(self) -> float:
        if not self.enabled or self._start is None or self._end is None:
            return 0.0
        self._end.record()
        torch.cuda.synchronize()
        return float(self._start.elapsed_time(self._end))


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    checkpoint_path: str | Path,
    step: int,
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    config: dict[str, Any],
) -> None:
    """Save training checkpoint and JSON config metadata."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "step": step,
        "config": config,
        "models": {name: model.state_dict() for name, model in models.items()},
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
    }

    torch.save(payload, checkpoint_path)

    config_path = checkpoint_path.with_suffix(".json")
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


