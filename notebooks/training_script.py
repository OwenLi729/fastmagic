""" Training script used to generate results for Fastmagic
    Used x1 H200
    AI-generated using Claude Sonnet 4.6
"""

#!/usr/bin/env python3
"""
run_benchmark.py  —  Fastmagic IQL benchmark runner for DCC SLURM

Ablation axes:
  Axis 1 — Value Network Depth   : n_hidden_layers in {1, 2}  (architectural choice)
  Axis 2 — Policy Conservatism   : beta            in {3.0, 10.0}  (methodological choice)
  tau is held fixed at 0.9 (best-performing value from the IQL paper) across
  all ablations so these two axes are cleanly isolated.

Usage (normally called by fastmagic.sbatch, but can be run directly):
  python run_benchmark.py [options]

Key flags:
  --preset         mujoco | antmaze   (default: mujoco)
  --max_envs       N                  (default: 3)
  --train_steps    N                  (default: 300000)
  --seeds          0 1 2              (default: 0 1)
  --eval_interval  N                  (default: 50000)
  --eval_episodes  N                  (default: 3)
    --no_resume                         rerun all runs (disable skip-existing)
  --no_compile                        disable torch.compile
  --no_ablations                      skip ablation sweep
  --no_profile                        skip profiling
  --work_dir       PATH               override $FASTMAGIC_WORK_DIR / ~/fastmagic_dcc
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fastmagic IQL DCC benchmark runner")
    p.add_argument("--preset",          default="mujoco", choices=["mujoco", "antmaze"])
    p.add_argument("--max_envs",        type=int,   default=3)
    p.add_argument("--train_steps",     type=int,   default=300_000)
    p.add_argument("--seeds",           type=int,   nargs="+", default=[0, 1, 2])
    p.add_argument("--eval_interval",   type=int,   default=50_000)
    p.add_argument("--eval_episodes",   type=int,   default=3)
    p.add_argument("--no_resume",       action="store_true")
    p.add_argument("--no_compile",      action="store_true")
    p.add_argument("--compile_mode",    default="reduce-overhead",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--compile_backend", default=None)
    p.add_argument("--no_parallel_vq",  action="store_true")
    p.add_argument("--no_profile",      action="store_true")
    p.add_argument("--no_ablations",    action="store_true")
    p.add_argument("--work_dir",        default=None)
    return p.parse_args()


# ── Ablation config ───────────────────────────────────────────────────────────
#   Axis 1: architectural  — value network depth (n_hidden_layers)
#   Axis 2: methodological — policy conservatism (beta / AWR temperature)
# tau is fixed at 0.9 (IQL paper default) — NOT swept, so it doesn't dilute the axes.

ABLATION_TAU_FIXED     = 0.9          # held constant across all ablations
ABLATION_HIDDEN_VALUES = [1, 2]       # Axis 1: shallow (1 layer) vs deep (2 layers)
ABLATION_BETA_VALUES   = [3.0, 10.0]  # Axis 2: moderate vs high conservatism


# ── Environment sets ──────────────────────────────────────────────────────────

PRESET_ENVS = {
    "mujoco": [
        "halfcheetah-medium-v2",
        "hopper-medium-v2",
        "walker2d-medium-v2",
        "halfcheetah-medium-replay-v2",
        "hopper-medium-replay-v2",
        "walker2d-medium-replay-v2",
        "halfcheetah-medium-expert-v2",
        "hopper-medium-expert-v2",
        "walker2d-medium-expert-v2",
    ],
    "antmaze": [
        "antmaze-umaze-v2",
        "antmaze-umaze-diverse-v2",
        "antmaze-medium-play-v2",
        "antmaze-medium-diverse-v2",
        "antmaze-large-play-v2",
        "antmaze-large-diverse-v2",
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_mujoco_env(mujoco_dir: Path) -> dict:
    """Return os.environ copy with all MuJoCo / D4RL vars set."""
    env = dict(os.environ)
    env["MUJOCO_GL"]                = "egl"
    env["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    env["MUJOCO_PY_MUJOCO_PATH"]    = str(mujoco_dir / "mujoco210")
    lib = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"]          = f"{lib}:{mujoco_dir / 'mujoco210' / 'bin'}"
    return env


def run_subprocess(cmd: list, log_file: Path, env: dict, cwd: Path, label: str):
    """Run cmd, tee stdout+stderr to log_file. Raises RuntimeError on failure."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"  cwd : {cwd}")
    print(f"  cmd : {' '.join(str(c) for c in cmd)}")
    print(f"  log : {log_file}")
    print(f"{'='*64}\n")

    with log_file.open("w") as fh:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(cwd),
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()

    if proc.returncode != 0:
        tail = log_file.read_text(errors="replace").splitlines()
        print(f"\n--- {label} FAILED (exit {proc.returncode}) — last 80 lines of log ---")
        print("\n".join(tail[-80:]))
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")

    print(f"\n✓ {label} complete.\n")


def common_eval_flags(args) -> list:
    """Flags shared by all three benchmark phases."""
    return [
        "--eval_interval", str(args.eval_interval),
        "--eval_episodes",  str(args.eval_episodes),
    ]


def compile_flags(args) -> list:
    """torch.compile flags, empty list if disabled."""
    if args.no_compile:
        return []
    flags = ["--torch_compile", "--compile_mode", args.compile_mode]
    if args.compile_backend:
        flags += ["--compile_backend", args.compile_backend]
    return flags


def resume_flags(args) -> list:
    """Resume flags, empty list when explicit full rerun is requested."""
    if args.no_resume:
        return []
    return ["--skip_existing"]


# ── Pipeline steps ────────────────────────────────────────────────────────────

def cache_datasets(selected_envs: list, env_py: str, repo_root: Path, bench_env: dict):
    print("\n[Step 1/5] Caching D4RL datasets...")
    for env_name in selected_envs:
        print(f"  → {env_name}")
        result = subprocess.run(
            [env_py, "data/download_d4rl.py", "--env", env_name],
            capture_output=True, text=True,
            cwd=str(repo_root), env=bench_env,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(f"Dataset cache failed for: {env_name}")
    print("  All datasets cached.")


def run_baseline(args, env_py: str, repo_root: Path, bench_env: dict):
    """Standard IQL — no optimizations (CPU replay, no compile, no mixed prec)."""
    cmd = [
        env_py, "src/benchmark_iql.py",
        "--preset",          args.preset,
        "--seeds",           *[str(s) for s in args.seeds],
        "--train_steps",     str(args.train_steps),
        "--results_root",    "data/results/benchmarks_baseline",
        "--checkpoint_root", "models/benchmarks_baseline",
        "--baseline",
        "--replay_device",   "cpu",
    ]
    cmd += common_eval_flags(args)
    cmd += resume_flags(args)
    if args.max_envs is not None:
        cmd += ["--max_envs", str(args.max_envs)]
    if not args.no_profile:
        cmd.append("--profile")

    log = repo_root / "data" / "results" / "benchmarks_baseline" / "run.log"
    run_subprocess(cmd, log, bench_env, repo_root, "Baseline (standard IQL)")


def run_improved(args, env_py: str, repo_root: Path, bench_env: dict):
    """Improved IQL — mixed precision + GPU replay + torch.compile + parallel VQ."""
    cmd = [
        env_py, "src/benchmark_iql.py",
        "--preset",          args.preset,
        "--seeds",           *[str(s) for s in args.seeds],
        "--train_steps",     str(args.train_steps),
        "--results_root",    "data/results/benchmarks_improved",
        "--checkpoint_root", "models/benchmarks_improved",
        "--mixed_precision",
        "--replay_device",   "gpu",
    ]
    cmd += common_eval_flags(args)
    cmd += resume_flags(args)
    if args.max_envs is not None:
        cmd += ["--max_envs", str(args.max_envs)]
    if not args.no_profile:
        cmd.append("--profile")
    cmd += compile_flags(args)
    if not args.no_parallel_vq:
        cmd.append("--parallel_vq_updates")

    log = repo_root / "data" / "results" / "benchmarks_improved" / "run.log"
    run_subprocess(cmd, log, bench_env, repo_root, "Improved (mixed prec + GPU replay + compile)")


def run_ablations(args, env_py: str, repo_root: Path, bench_env: dict):
    """
    2×2 ablation sweep:
      Axis 1 — Value Network Depth    : n_hidden_layers ∈ {1, 2}    (architectural)
      Axis 2 — Policy Conservatism    : beta            ∈ {3.0, 10.0} (methodological)
    tau is fixed at 0.9 (NOT swept) so the two axes are independently interpretable.
    Grid: 2 × 2 × 3 envs × 2 seeds = 24 runs × 300K steps each.
    """
    cmd = [
        env_py, "src/benchmark_iql.py",
        "--preset",               args.preset,
        "--seeds",                *[str(s) for s in args.seeds],
        "--train_steps",          str(args.train_steps),
        "--results_root",         "data/results/benchmarks_ablations",
        "--checkpoint_root",      "models/benchmarks_ablations",
        "--mixed_precision",
        "--replay_device",        "gpu",
        "--tau_values",           str(ABLATION_TAU_FIXED),           # fixed, not swept
        "--beta_values",          *[str(x) for x in ABLATION_BETA_VALUES],
        "--n_hidden_layer_values",*[str(x) for x in ABLATION_HIDDEN_VALUES],
    ]
    cmd += common_eval_flags(args)
    cmd += resume_flags(args)
    if args.max_envs is not None:
        cmd += ["--max_envs", str(args.max_envs)]
    if not args.no_profile:
        cmd.append("--profile")
    cmd += compile_flags(args)
    if not args.no_parallel_vq:
        cmd.append("--parallel_vq_updates")

    log = repo_root / "data" / "results" / "benchmarks_ablations" / "run.log"
    run_subprocess(cmd, log, bench_env, repo_root,
                   "Ablations (depth ∈ {1,2} × beta ∈ {3.0,10.0}, tau=0.9 fixed)")


def aggregate_results(args, repo_root: Path, output_dir: Path):
    """Merge baseline/improved CSVs and print comparison. Replaces display()."""
    import pandas as pd

    baseline_agg = repo_root / "data" / "results" / "benchmarks_baseline" / f"{args.preset}_aggregate.csv"
    improved_agg = repo_root / "data" / "results" / "benchmarks_improved" / f"{args.preset}_aggregate.csv"

    if not baseline_agg.exists() or not improved_agg.exists():
        print("WARNING: aggregate CSVs missing — skipping comparison (did benchmark_iql.py produce them?)")
        return

    rename_base = {
        "final_score_mean":              "baseline_final_mean",
        "best_score_mean":               "baseline_best_mean",
        "final_score_std":               "baseline_final_std",
        "best_score_std":                "baseline_best_std",
        "avg_wall_clock_per_update_ms":  "baseline_update_ms",
        "avg_replay_buffer_throughput":  "baseline_replay_sps",
        "avg_critic_actor_update_ratio": "baseline_critic_actor_ratio",
        "avg_inference_time_ms":         "baseline_inference_ms",
    }
    rename_impr = {k: v.replace("baseline_", "improved_") for k, v in rename_base.items()}

    bdf = pd.read_csv(baseline_agg).rename(columns=rename_base)
    idf = pd.read_csv(improved_agg).rename(columns=rename_impr)

    for df in (bdf, idf):
        if "ablation_type" not in df.columns:
            df["ablation_type"] = "default"
        if "ablation_value" not in df.columns:
            df["ablation_value"] = "default"

    cdf = bdf.merge(idf, on=["env", "ablation_type", "ablation_value"], how="inner")
    cdf["delta_final_mean"] = cdf["improved_final_mean"] - cdf["baseline_final_mean"]
    cdf["delta_best_mean"]  = cdf["improved_best_mean"]  - cdf["baseline_best_mean"]
    for col in ["update_ms", "replay_sps", "inference_ms"]:
        b, i = f"baseline_{col}", f"improved_{col}"
        if b in cdf.columns and i in cdf.columns:
            cdf[f"delta_{col}"] = cdf[i] - cdf[b]

    sorted_cdf = cdf.sort_values(["ablation_type", "ablation_value", "env"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{args.preset}_baseline_vs_improved.csv"
    sorted_cdf.to_csv(out_csv, index=False)
    print(f"\nBaseline vs Improved — saved to {out_csv}")
    print(sorted_cdf.to_string())

    # Ablation results
    abl_agg = repo_root / "data" / "results" / "benchmarks_ablations" / f"{args.preset}_aggregate.csv"
    if abl_agg.exists():
        adf = pd.read_csv(abl_agg)
        keep = [c for c in [
            "ablation_type", "ablation_value", "env",
            "final_score_mean", "best_score_mean",
            "avg_wall_clock_per_update_ms",
            "avg_replay_buffer_throughput",
            "avg_critic_actor_update_ratio",
            "avg_inference_time_ms",
        ] if c in adf.columns]
        abl_out = output_dir / f"{args.preset}_ablations.csv"
        adf[keep].sort_values(["ablation_type", "ablation_value", "env"]).to_csv(abl_out, index=False)
        print(f"\nAblation results — saved to {abl_out}")
        print(adf[keep].sort_values(["ablation_type", "ablation_value", "env"]).to_string())
    else:
        print("\nNo ablation aggregate CSV found (expected if --no_ablations was used).")


def copy_artifacts(repo_root: Path, output_dir: Path):
    """Copy results + model checkpoints to output_dir (replaces Google Drive)."""
    print("\nCopying artifacts to output dir...")
    for rel in [
        "data/results/benchmarks_baseline",
        "data/results/benchmarks_improved",
        "data/results/benchmarks_ablations",
        "models/benchmarks_baseline",
        "models/benchmarks_improved",
        "models/benchmarks_ablations",
    ]:
        src = repo_root / rel
        if src.exists():
            dst = output_dir / src.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  {src.name} → {dst}")
    print(f"\nAll artifacts in: {output_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve working directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
    elif "FASTMAGIC_WORK_DIR" in os.environ:
        work_dir = Path(os.environ["FASTMAGIC_WORK_DIR"])
    else:
        work_dir = Path.home() / "fastmagic_dcc"

    repo_root  = work_dir / "fastmagic"
    mujoco_dir = Path.home() / ".mujoco"
    output_dir = work_dir / "outputs"
    env_py     = sys.executable   # whatever Python is active in the conda env

    selected_envs = PRESET_ENVS[args.preset]
    if args.max_envs is not None:
        selected_envs = selected_envs[: args.max_envs]

    bench_env = make_mujoco_env(mujoco_dir)

    ablation_runs = (
        len(ABLATION_BETA_VALUES) * len(ABLATION_HIDDEN_VALUES)
        * len(selected_envs) * len(args.seeds)
    )

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║            Fastmagic IQL — DCC Benchmark Runner              ║
╠══════════════════════════════════════════════════════════════╣
  preset         : {args.preset}
  envs           : {selected_envs}
  seeds          : {args.seeds}
  train_steps    : {args.train_steps:,}
  eval_interval  : {args.eval_interval:,}
  eval_episodes  : {args.eval_episodes}
    resume_mode    : {not args.no_resume} (skip existing summaries)
  torch_compile  : {not args.no_compile}  (mode: {args.compile_mode})
  parallel_vq    : {not args.no_parallel_vq}
  ablations      : {not args.no_ablations}
    ├─ Axis 1 (arch)    n_hidden_layers ∈ {ABLATION_HIDDEN_VALUES}
    ├─ Axis 2 (method)  beta            ∈ {ABLATION_BETA_VALUES}
    ├─ tau fixed at     {ABLATION_TAU_FIXED}
    └─ total runs       {ablation_runs}
  work_dir       : {work_dir}
  repo_root      : {repo_root}
  python         : {env_py}
╚══════════════════════════════════════════════════════════════╝
""")

    # Sanity check: repo must exist (setup_env.sh clones it)
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Repo not found at {repo_root}. Did you run setup_env.sh?"
        )

    # ── Step 1: cache datasets ────────────────────────────────────────────────
    cache_datasets(selected_envs, env_py, repo_root, bench_env)

    # ── Step 2: baseline ──────────────────────────────────────────────────────
    print("\n[Step 2/5] Running baseline (standard IQL, CPU replay)...")
    run_baseline(args, env_py, repo_root, bench_env)

    # ── Step 3: improved ──────────────────────────────────────────────────────
    print("\n[Step 3/5] Running improved config (mixed prec + GPU replay + compile)...")
    run_improved(args, env_py, repo_root, bench_env)

    # ── Step 4: ablations ─────────────────────────────────────────────────────
    if not args.no_ablations:
        print(f"\n[Step 4/5] Running ablation sweep ({ablation_runs} runs)...")
        run_ablations(args, env_py, repo_root, bench_env)
    else:
        print("\n[Step 4/5] Ablations skipped (--no_ablations).")

    # ── Step 5: aggregate + save ──────────────────────────────────────────────
    print("\n[Step 5/5] Aggregating results and saving CSVs...")
    aggregate_results(args, repo_root, output_dir)
    copy_artifacts(repo_root, output_dir)

    print("\n✓ All done. Results in:", output_dir)


if __name__ == "__main__":
    main()
