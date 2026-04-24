# AI-generated: Claude, 2026-04-23
# Benchmark presets are adapted from the official IQL repository's published
# offline settings for MuJoCo and AntMaze tasks.
"""Run multi-seed IQL benchmark sweeps and aggregate results."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from train import ANTMAZE_BENCHMARK_ENVS, MUJOCO_BENCHMARK_ENVS


PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    "mujoco": {
        "envs": list(MUJOCO_BENCHMARK_ENVS),
        "tau": 0.7,
        "beta": 3.0,
        "eval_interval": 5_000,
        "eval_episodes": 10,
        "train_steps": 1_000_000,
    },
    "antmaze": {
        "envs": list(ANTMAZE_BENCHMARK_ENVS),
        "tau": 0.9,
        "beta": 10.0,
        "eval_interval": 100_000,
        "eval_episodes": 100,
        "train_steps": 1_000_000,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark sweeps."""
    parser = argparse.ArgumentParser(description="Run paper-style IQL benchmark sweeps")
    parser.add_argument("--preset", type=str, choices=sorted(PRESET_CONFIGS.keys()), default="mujoco")
    parser.add_argument("--envs", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train_steps", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_hidden_layers", type=int, default=2)
    parser.add_argument("--checkpoint_root", type=str, default="models/benchmarks")
    parser.add_argument("--results_root", type=str, default="results/benchmarks")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--replay_device", type=str, choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--deterministic_torch", action="store_true")
    parser.add_argument("--max_envs", type=int, default=None)
    return parser.parse_args()


def choose_envs(args: argparse.Namespace) -> list[str]:
    """Resolve the environment list for the sweep."""
    envs = list(args.envs) if args.envs else list(PRESET_CONFIGS[args.preset]["envs"])
    if args.max_envs is not None:
        envs = envs[: args.max_envs]
    return envs


def build_train_command(args: argparse.Namespace, env_name: str, seed: int) -> list[str]:
    """Construct the subprocess command for one train run."""
    preset = PRESET_CONFIGS[args.preset]
    train_steps = args.train_steps or preset["train_steps"]
    eval_interval = args.eval_interval or preset["eval_interval"]
    eval_episodes = args.eval_episodes or preset["eval_episodes"]
    tau = preset["tau"]
    beta = preset["beta"]
    run_name = f"{args.preset}_{env_name}_seed{seed}"

    command = [
        sys.executable,
        "-u",
        "-X",
        "faulthandler",
        "src/train.py",
        "--env",
        env_name,
        "--seed",
        str(seed),
        "--tau",
        str(tau),
        "--beta",
        str(beta),
        "--train_steps",
        str(train_steps),
        "--eval_interval",
        str(eval_interval),
        "--eval_episodes",
        str(eval_episodes),
        "--batch_size",
        str(args.batch_size),
        "--hidden_dim",
        str(args.hidden_dim),
        "--n_hidden_layers",
        str(args.n_hidden_layers),
        "--checkpoint_dir",
        args.checkpoint_root,
        "--results_dir",
        args.results_root,
        "--run_name",
        run_name,
        "--replay_device",
        args.replay_device,
    ]
    if args.baseline:
        command.append("--baseline")
    if args.mixed_precision:
        command.append("--mixed_precision")
    if args.profile:
        command.append("--profile")
    if args.deterministic_torch:
        command.append("--deterministic_torch")
    return command


def read_summary(results_root: Path, preset: str, env_name: str, seed: int) -> dict[str, Any]:
    """Load the summary JSON produced by a single train run."""
    run_name = f"{preset}_{env_name}_seed{seed}"
    summary_path = results_root / run_name / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate final and best scores across seeds for each environment."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["env"]), []).append(row)

    aggregate: list[dict[str, Any]] = []
    for env_name, env_rows in grouped.items():
        final_scores = [float(row["final_d4rl_normalized_score"]) for row in env_rows if row["final_d4rl_normalized_score"] is not None]
        best_scores = [float(row["best_d4rl_normalized_score"]) for row in env_rows if row["best_d4rl_normalized_score"] is not None]
        aggregate.append(
            {
                "env": env_name,
                "num_seeds": len(env_rows),
                "final_score_mean": statistics.mean(final_scores) if final_scores else None,
                "final_score_std": statistics.pstdev(final_scores) if len(final_scores) > 1 else 0.0 if final_scores else None,
                "best_score_mean": statistics.mean(best_scores) if best_scores else None,
                "best_score_std": statistics.pstdev(best_scores) if len(best_scores) > 1 else 0.0 if best_scores else None,
            }
        )
    return aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries as CSV."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join("" if row[key] is None else str(row[key]) for key in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the configured benchmark sweep."""
    args = parse_args()
    envs = choose_envs(args)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    for env_name in envs:
        for seed in args.seeds:
            command = build_train_command(args=args, env_name=env_name, seed=seed)
            print(f"[benchmark] running env={env_name} seed={seed}")
            run_result = subprocess.run(command, check=False, capture_output=True, text=True)
            if run_result.stdout:
                print(run_result.stdout, end="")
            if run_result.returncode != 0:
                print(f"[benchmark] failed env={env_name} seed={seed} exit={run_result.returncode}")
                if run_result.stderr:
                    print("[benchmark] child stderr:")
                    print(run_result.stderr, end="")
                raise RuntimeError(f"train failed for env={env_name} seed={seed} with exit code {run_result.returncode}")
            raw_rows.append(read_summary(results_root=results_root, preset=args.preset, env_name=env_name, seed=seed))

    aggregate_rows_data = aggregate_rows(raw_rows)
    write_csv(results_root / f"{args.preset}_raw_runs.csv", raw_rows)
    write_csv(results_root / f"{args.preset}_aggregate.csv", aggregate_rows_data)

    print(f"[benchmark] wrote raw runs to {results_root / f'{args.preset}_raw_runs.csv'}")
    print(f"[benchmark] wrote aggregate results to {results_root / f'{args.preset}_aggregate.csv'}")


if __name__ == "__main__":
    main()
