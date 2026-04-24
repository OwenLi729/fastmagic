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
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument(
        "--compile_mode",
        type=str,
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--compile_backend", type=str, default=None)
    parser.add_argument("--parallel_vq_updates", action="store_true")
    parser.add_argument("--tau_values", nargs="*", type=float, default=None)
    parser.add_argument("--beta_values", nargs="*", type=float, default=None)
    parser.add_argument("--n_hidden_layer_values", nargs="*", type=int, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--deterministic_torch", action="store_true")
    parser.add_argument("--max_envs", type=int, default=None)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip runs that already have results_root/<run_name>/summary.json.",
    )
    return parser.parse_args()


def choose_envs(args: argparse.Namespace) -> list[str]:
    """Resolve the environment list for the sweep."""
    envs = list(args.envs) if args.envs else list(PRESET_CONFIGS[args.preset]["envs"])
    if args.max_envs is not None:
        envs = envs[: args.max_envs]
    return envs


def build_train_command(
    args: argparse.Namespace,
    env_name: str,
    seed: int,
    tau: float,
    beta: float,
    n_hidden_layers: int,
    run_name: str,
) -> list[str]:
    """Construct the subprocess command for one train run."""
    preset = PRESET_CONFIGS[args.preset]
    train_steps = args.train_steps or preset["train_steps"]
    eval_interval = args.eval_interval or preset["eval_interval"]
    eval_episodes = args.eval_episodes or preset["eval_episodes"]
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
        str(n_hidden_layers),
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
    if args.torch_compile:
        command.append("--torch_compile")
        command.extend(["--compile_mode", args.compile_mode])
        if args.compile_backend:
            command.extend(["--compile_backend", args.compile_backend])
    if args.parallel_vq_updates:
        command.append("--parallel_vq_updates")
    if args.profile:
        command.append("--profile")
    if args.deterministic_torch:
        command.append("--deterministic_torch")
    return command


def read_summary(results_root: Path, run_name: str) -> dict[str, Any]:
    """Load the summary JSON produced by a single train run."""
    summary_path = results_root / run_name / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# AI-generated: GitHub Copilot, 2026-04-24
def is_completed_summary(summary: dict[str, Any]) -> bool:
    """Return True when a run summary is from a fully completed training run."""
    explicit_complete = summary.get("is_complete")
    if isinstance(explicit_complete, bool):
        return explicit_complete

    # Backward-compatible heuristic for older summaries that do not carry
    # `is_complete`: final summaries include training-time aggregate metrics.
    return summary.get("avg_wall_clock_per_update_ms") is not None


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate final and best scores across seeds for each environment."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["env"]), str(row.get("ablation_type", "default")), str(row.get("ablation_value", "default")))
        grouped.setdefault(key, []).append(row)

    aggregate: list[dict[str, Any]] = []
    for (env_name, ablation_type, ablation_value), env_rows in grouped.items():
        final_scores = [float(row["final_d4rl_normalized_score"]) for row in env_rows if row["final_d4rl_normalized_score"] is not None]
        best_scores = [float(row["best_d4rl_normalized_score"]) for row in env_rows if row["best_d4rl_normalized_score"] is not None]
        avg_update_ms = [float(row["avg_wall_clock_per_update_ms"]) for row in env_rows if row.get("avg_wall_clock_per_update_ms") is not None]
        avg_replay_sps = [float(row["avg_replay_buffer_throughput"]) for row in env_rows if row.get("avg_replay_buffer_throughput") is not None]
        avg_ratio = [float(row["avg_critic_actor_update_ratio"]) for row in env_rows if row.get("avg_critic_actor_update_ratio") is not None]
        avg_inference_ms = [float(row["avg_inference_time_ms"]) for row in env_rows if row.get("avg_inference_time_ms") is not None]
        aggregate.append(
            {
                "env": env_name,
                "ablation_type": ablation_type,
                "ablation_value": ablation_value,
                "num_seeds": len(env_rows),
                "final_score_mean": statistics.mean(final_scores) if final_scores else None,
                "final_score_std": statistics.pstdev(final_scores) if len(final_scores) > 1 else 0.0 if final_scores else None,
                "best_score_mean": statistics.mean(best_scores) if best_scores else None,
                "best_score_std": statistics.pstdev(best_scores) if len(best_scores) > 1 else 0.0 if best_scores else None,
                "avg_wall_clock_per_update_ms": statistics.mean(avg_update_ms) if avg_update_ms else None,
                "avg_replay_buffer_throughput": statistics.mean(avg_replay_sps) if avg_replay_sps else None,
                "avg_critic_actor_update_ratio": statistics.mean(avg_ratio) if avg_ratio else None,
                "avg_inference_time_ms": statistics.mean(avg_inference_ms) if avg_inference_ms else None,
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

    preset = PRESET_CONFIGS[args.preset]

    experiment_specs: list[dict[str, Any]] = [
        {
            "ablation_type": "default",
            "ablation_value": "default",
            "tau": float(preset["tau"]),
            "beta": float(preset["beta"]),
            "n_hidden_layers": int(args.n_hidden_layers),
        }
    ]

    if args.tau_values:
        experiment_specs.extend(
            {
                "ablation_type": "tau",
                "ablation_value": str(tau),
                "tau": float(tau),
                "beta": float(preset["beta"]),
                "n_hidden_layers": int(args.n_hidden_layers),
            }
            for tau in args.tau_values
        )

    if args.beta_values:
        experiment_specs.extend(
            {
                "ablation_type": "beta",
                "ablation_value": str(beta),
                "tau": float(preset["tau"]),
                "beta": float(beta),
                "n_hidden_layers": int(args.n_hidden_layers),
            }
            for beta in args.beta_values
        )

    if args.n_hidden_layer_values:
        experiment_specs.extend(
            {
                "ablation_type": "n_hidden_layers",
                "ablation_value": str(n_hidden_layers),
                "tau": float(preset["tau"]),
                "beta": float(preset["beta"]),
                "n_hidden_layers": int(n_hidden_layers),
            }
            for n_hidden_layers in args.n_hidden_layer_values
        )

    raw_rows: list[dict[str, Any]] = []
    skipped_runs = 0
    for spec in experiment_specs:
        for env_name in envs:
            for seed in args.seeds:
                run_name = (
                    f"{args.preset}_{env_name}_seed{seed}"
                    f"_{spec['ablation_type']}_{spec['ablation_value']}"
                )
                summary_path = results_root / run_name / "summary.json"
                if args.skip_existing and summary_path.exists():
                    summary = read_summary(results_root=results_root, run_name=run_name)
                    if not is_completed_summary(summary):
                        print(
                            "[benchmark] rerunning incomplete "
                            f"ablation={spec['ablation_type']} value={spec['ablation_value']} "
                            f"env={env_name} seed={seed}"
                        )
                    else:
                        print(
                            "[benchmark] skipping existing "
                            f"ablation={spec['ablation_type']} value={spec['ablation_value']} "
                            f"env={env_name} seed={seed}"
                        )
                        summary["ablation_type"] = spec["ablation_type"]
                        summary["ablation_value"] = spec["ablation_value"]
                        raw_rows.append(summary)
                        skipped_runs += 1
                        continue
                command = build_train_command(
                    args=args,
                    env_name=env_name,
                    seed=seed,
                    tau=float(spec["tau"]),
                    beta=float(spec["beta"]),
                    n_hidden_layers=int(spec["n_hidden_layers"]),
                    run_name=run_name,
                )
                print(
                    "[benchmark] running "
                    f"ablation={spec['ablation_type']} value={spec['ablation_value']} "
                    f"env={env_name} seed={seed}"
                )
                run_result = subprocess.run(command, check=False, capture_output=True, text=True)
                if run_result.stdout:
                    print(run_result.stdout, end="")
                if run_result.returncode != 0:
                    print(f"[benchmark] failed env={env_name} seed={seed} exit={run_result.returncode}")
                    if run_result.stderr:
                        print("[benchmark] child stderr:")
                        print(run_result.stderr, end="")
                    raise RuntimeError(f"train failed for env={env_name} seed={seed} with exit code {run_result.returncode}")
                summary = read_summary(results_root=results_root, run_name=run_name)
                summary["ablation_type"] = spec["ablation_type"]
                summary["ablation_value"] = spec["ablation_value"]
                raw_rows.append(summary)

    aggregate_rows_data = aggregate_rows(raw_rows)
    write_csv(results_root / f"{args.preset}_raw_runs.csv", raw_rows)
    write_csv(results_root / f"{args.preset}_aggregate.csv", aggregate_rows_data)

    print(f"[benchmark] wrote raw runs to {results_root / f'{args.preset}_raw_runs.csv'}")
    print(f"[benchmark] wrote aggregate results to {results_root / f'{args.preset}_aggregate.csv'}")
    if skipped_runs > 0:
        print(f"[benchmark] skipped {skipped_runs} existing run(s)")


if __name__ == "__main__":
    main()
