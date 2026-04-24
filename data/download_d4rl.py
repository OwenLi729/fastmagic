# AI-generated: Claude, 2026-04-21
"""Download and cache D4RL datasets for offline RL training."""

from __future__ import annotations

import argparse
from pathlib import Path

import d4rl
import gym
import numpy as np


# AI-generated: Claude, 2026-04-21
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache a D4RL dataset")
    parser.add_argument("--env", type=str, default="hopper-medium-v2")
    parser.add_argument("--output", type=str, default="data/cache")
    parser.add_argument("--save_npz", action="store_true")
    return parser.parse_args()


# AI-generated: Claude, 2026-04-21
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env)
    dataset = d4rl.qlearning_dataset(env)

    print(f"Cached dataset for {args.env}")
    print(f"observations: {dataset['observations'].shape}")
    print(f"actions: {dataset['actions'].shape}")
    print(f"rewards: {dataset['rewards'].shape}")

    if args.save_npz:
        output_path = output_dir / f"{args.env}.npz"
        np.savez_compressed(output_path, **dataset)
        print(f"Saved dataset snapshot to {output_path}")


if __name__ == "__main__":
    main()
