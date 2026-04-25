# AI-generated: GitHub Copilot, 2026-04-25

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PALETTE: Dict[str, str] = {
    "baseline": "#1f77b4",
    "amp+compile": "#d62728",
    "best_ablation": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visualization figures from the committed benchmark results.")
    parser.add_argument("--results-root", type=Path, default=Path("data/results"))
    parser.add_argument(
        "--baseline-aggregate",
        type=Path,
        default=Path("data/results/benchmarks_baseline/mujoco_aggregate.csv"),
    )
    parser.add_argument(
        "--improved-aggregate",
        type=Path,
        default=Path("data/results/benchmarks_improved/mujoco_aggregate.csv"),
    )
    parser.add_argument(
        "--ablation-aggregate",
        type=Path,
        default=Path("data/results/benchmarks_ablations/mujoco_aggregate.csv"),
    )
    parser.add_argument(
        "--baseline-raw",
        type=Path,
        default=Path("data/results/benchmarks_baseline/mujoco_raw_runs.csv"),
    )
    parser.add_argument(
        "--improved-raw",
        type=Path,
        default=Path("data/results/benchmarks_improved/mujoco_raw_runs.csv"),
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--improved-label", type=str, default="amp+compile")
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def make_variant_label(ablation_type: str, ablation_value: str, improved_label: str) -> str:
    if ablation_type == "default":
        return improved_label
    if ablation_type == "tau":
        return f"tau={ablation_value}"
    if ablation_type == "beta":
        return f"beta={ablation_value}"
    if ablation_type == "n_hidden_layers":
        return f"layers={ablation_value}"
    return f"{ablation_type}={ablation_value}"


def parse_run_dir_name(name: str) -> tuple[str, int, str, str]:
    if not name.startswith("mujoco_"):
        raise ValueError(f"Unexpected run directory name: {name}")

    rest = name[len("mujoco_") :]
    env, right = rest.split("_seed", maxsplit=1)
    seed_text, _, ablation_part = right.partition("_")
    seed = int(seed_text)

    if ablation_part.startswith("default_default"):
        return env, seed, "default", "default"
    if ablation_part.startswith("tau_"):
        return env, seed, "tau", ablation_part[len("tau_") :]
    if ablation_part.startswith("beta_"):
        return env, seed, "beta", ablation_part[len("beta_") :]
    if ablation_part.startswith("n_hidden_layers_"):
        return env, seed, "n_hidden_layers", ablation_part[len("n_hidden_layers_") :]

    raise ValueError(f"Could not parse ablation info from run directory: {name}")


def collect_eval_histories(results_root: Path, improved_label: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for group_name in ["benchmarks_baseline", "benchmarks_improved", "benchmarks_ablations"]:
        group_dir = results_root / group_name
        if not group_dir.exists():
            continue

        for eval_path in sorted(group_dir.glob("mujoco_*/eval_history.csv")):
            env, seed, ablation_type, ablation_value = parse_run_dir_name(eval_path.parent.name)
            variant = "baseline" if group_name == "benchmarks_baseline" else (
                improved_label if group_name == "benchmarks_improved" else make_variant_label(ablation_type, ablation_value, improved_label)
            )
            df = pd.read_csv(eval_path).rename(columns={"d4rl_normalized_score": "normalized_score"})
            df["env"] = env
            df["seed"] = seed
            df["variant"] = variant
            rows.append(df[["env", "seed", "step", "normalized_score", "variant"]])

    if not rows:
        raise ValueError("No eval histories found under results root.")
    return pd.concat(rows, ignore_index=True)


def env_order_from_aggregates(baseline_agg: pd.DataFrame) -> list[str]:
    return list(baseline_agg["env"].tolist())


def build_best_ablation_table(ablation_agg: pd.DataFrame, improved_label: str) -> pd.DataFrame:
    df = ablation_agg.copy()
    df["variant"] = df.apply(
        lambda row: make_variant_label(str(row["ablation_type"]), str(row["ablation_value"]), improved_label),
        axis=1,
    )
    best = df.sort_values(["env", "final_score_mean"], ascending=[True, False]).groupby("env", as_index=False).first()
    best = best.rename(
        columns={
            "variant": "best_variant",
            "final_score_mean": "best_ablation_final_mean",
            "final_score_std": "best_ablation_final_std",
            "best_score_mean": "best_ablation_best_mean",
            "best_score_std": "best_ablation_best_std",
        }
    )
    return best[
        [
            "env",
            "best_variant",
            "best_ablation_final_mean",
            "best_ablation_final_std",
            "best_ablation_best_mean",
            "best_ablation_best_std",
        ]
    ]


def plot_learning_curves(
    eval_histories: pd.DataFrame,
    baseline_agg: pd.DataFrame,
    improved_label: str,
    figures_dir: Path,
) -> None:
    envs = env_order_from_aggregates(baseline_agg)
    selected = eval_histories[eval_histories["variant"].isin(["baseline", improved_label])].copy()
    summary = (
        selected.groupby(["env", "variant", "step"])["normalized_score"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 4.8), squeeze=False)
    axes = axes[0]

    for idx, env in enumerate(envs):
        ax = axes[idx]
        env_df = summary[summary["env"] == env]
        for variant in ["baseline", improved_label]:
            var_df = env_df[env_df["variant"] == variant].sort_values("step")
            if var_df.empty:
                continue
            color = PALETTE[variant]
            ax.plot(var_df["step"], var_df["mean"], label=variant, color=color, linewidth=2)
            std = var_df["std"].fillna(0.0)
            ax.fill_between(var_df["step"], var_df["mean"] - std, var_df["mean"] + std, color=color, alpha=0.18)

        ax.set_title(env)
        ax.set_xlabel("Training steps")
        ax.set_ylabel("D4RL normalized score")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    fig.suptitle("Learning Curves: Baseline vs Improved", y=1.14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(figures_dir / "01_learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_score_comparison(
    baseline_agg: pd.DataFrame,
    improved_agg: pd.DataFrame,
    best_ablation: pd.DataFrame,
    improved_label: str,
    figures_dir: Path,
) -> None:
    merged = baseline_agg.merge(improved_agg, on="env", suffixes=("_baseline", "_improved"))
    merged = merged.merge(best_ablation, on="env", how="left")
    envs = env_order_from_aggregates(baseline_agg)
    merged = merged.set_index("env").loc[envs].reset_index()

    x = np.arange(len(envs))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    score_panels = [
        (
            axes[0],
            "final",
            "Final normalized score",
            [
                ("baseline", "final_score_mean_baseline", "final_score_std_baseline"),
                (improved_label, "final_score_mean_improved", "final_score_std_improved"),
                ("best_ablation", "best_ablation_final_mean", "best_ablation_final_std"),
            ],
        ),
        (
            axes[1],
            "best",
            "Best normalized score during training",
            [
                ("baseline", "best_score_mean_baseline", "best_score_std_baseline"),
                (improved_label, "best_score_mean_improved", "best_score_std_improved"),
                ("best_ablation", "best_ablation_best_mean", "best_ablation_best_std"),
            ],
        ),
    ]

    for ax, metric_name, title, columns in score_panels:
        for offset, (label, mean_col, std_col) in zip([-width, 0.0, width], columns):
            means = merged[mean_col].to_numpy(dtype=float)
            bars = ax.bar(x + offset, means, width=width, color=PALETTE[label], label=label)
            if metric_name == "final" and label in {improved_label, "best_ablation"}:
                base = merged["final_score_mean_baseline"].to_numpy(dtype=float)
                for bar, base_val, value in zip(bars, base, means):
                    delta = value - base_val
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f"{delta:+.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(envs, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("D4RL normalized score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    fig.suptitle("Baseline, Improved, and Best Ablation Score Comparison", y=1.14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(figures_dir / "02_score_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_systems_metrics(
    baseline_agg: pd.DataFrame,
    improved_agg: pd.DataFrame,
    improved_label: str,
    figures_dir: Path,
) -> None:
    merged = baseline_agg.merge(improved_agg, on="env", suffixes=("_baseline", "_improved"))
    envs = env_order_from_aggregates(baseline_agg)
    merged = merged.set_index("env").loc[envs].reset_index()
    x = np.arange(len(envs))
    width = 0.35

    metric_specs = [
        ("avg_wall_clock_per_update_ms", "Update time (ms, lower is better)", lambda b, i: (b - i) / b * 100.0),
        ("avg_replay_buffer_throughput", "Replay throughput (samples/sec)", lambda b, i: (i - b) / b * 100.0),
        ("avg_inference_time_ms", "Inference time (ms, lower is better)", lambda b, i: (b - i) / b * 100.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, (metric, title, delta_fn) in zip(axes, metric_specs):
        baseline_vals = merged[f"{metric}_baseline"].to_numpy(dtype=float)
        improved_vals = merged[f"{metric}_improved"].to_numpy(dtype=float)

        ax.bar(x - width / 2, baseline_vals, width=width, color=PALETTE["baseline"], label="baseline")
        bars = ax.bar(x + width / 2, improved_vals, width=width, color=PALETTE[improved_label], label=improved_label)

        for bar, base_val, imp_val in zip(bars, baseline_vals, improved_vals):
            delta_pct = delta_fn(base_val, imp_val)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(baseline_vals.max(), improved_vals.max()),
                f"{delta_pct:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(envs, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    fig.suptitle("Systems Metrics: Baseline vs Improved", y=1.14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(figures_dir / "03_systems_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_heatmap(ablation_agg: pd.DataFrame, improved_label: str, figures_dir: Path) -> None:
    df = ablation_agg.copy()
    df["variant"] = df.apply(
        lambda row: make_variant_label(str(row["ablation_type"]), str(row["ablation_value"]), improved_label),
        axis=1,
    )

    column_order = [improved_label, "tau=0.9", "beta=3.0", "beta=10.0", "layers=1", "layers=2"]
    existing_columns = [col for col in column_order if col in set(df["variant"])]

    means = df.pivot(index="env", columns="variant", values="final_score_mean")
    stds = df.pivot(index="env", columns="variant", values="final_score_std")
    means = means[existing_columns]
    stds = stds[existing_columns]

    annot = means.copy().astype(object)
    for env in means.index:
        for variant in means.columns:
            annot.loc[env, variant] = f"{means.loc[env, variant]:.2f}±{stds.loc[env, variant]:.2f}"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.heatmap(
        means,
        annot=annot,
        fmt="",
        cmap="viridis",
        linewidths=0.4,
        cbar_kws={"label": "Final normalized score"},
        ax=ax,
    )
    ax.set_title("Ablation Study: Final Score ± Std")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Environment")
    fig.tight_layout()
    fig.savefig(figures_dir / "04_ablation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pick_failure_case_env(baseline_raw: pd.DataFrame, improved_raw: pd.DataFrame, improved_label: str) -> str:
    raw = pd.concat(
        [
            baseline_raw.assign(variant="baseline"),
            improved_raw.assign(variant=improved_label),
        ],
        ignore_index=True,
    )
    raw["peak_to_final_drop"] = raw["best_d4rl_normalized_score"] - raw["final_d4rl_normalized_score"]
    worst = raw.groupby("env", as_index=False)["peak_to_final_drop"].mean().sort_values("peak_to_final_drop", ascending=False)
    return str(worst.iloc[0]["env"])


def plot_stability_analysis(
    eval_histories: pd.DataFrame,
    baseline_raw: pd.DataFrame,
    improved_raw: pd.DataFrame,
    improved_label: str,
    figures_dir: Path,
) -> None:
    combined_raw = pd.concat(
        [
            baseline_raw.assign(variant="baseline"),
            improved_raw.assign(variant=improved_label),
        ],
        ignore_index=True,
    )
    combined_raw["peak_to_final_drop"] = combined_raw["best_d4rl_normalized_score"] - combined_raw["final_d4rl_normalized_score"]

    focus_env = pick_failure_case_env(baseline_raw, improved_raw, improved_label)
    focus_histories = eval_histories[
        (eval_histories["env"] == focus_env) & (eval_histories["variant"].isin(["baseline", improved_label]))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    sns.barplot(
        data=combined_raw,
        x="env",
        y="peak_to_final_drop",
        hue="variant",
        errorbar=None,
        palette={"baseline": PALETTE["baseline"], improved_label: PALETTE[improved_label]},
        ax=axes[0],
    )
    sns.stripplot(
        data=combined_raw,
        x="env",
        y="peak_to_final_drop",
        hue="variant",
        dodge=True,
        marker="o",
        size=5,
        linewidth=0,
        palette={"baseline": PALETTE["baseline"], improved_label: PALETTE[improved_label]},
        ax=axes[0],
    )
    axes[0].set_title("Training Stability: Peak-to-Final Drop")
    axes[0].set_xlabel("Environment")
    axes[0].set_ylabel("Best score - final score")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    for variant in ["baseline", improved_label]:
        var_df = focus_histories[focus_histories["variant"] == variant]
        for seed, seed_df in var_df.groupby("seed"):
            axes[1].plot(
                seed_df["step"],
                seed_df["normalized_score"],
                marker="o",
                linewidth=2,
                label=f"{variant} seed={seed}",
                color=PALETTE[variant],
                alpha=0.6 if seed else 1.0,
            )

    axes[1].set_title(f"Failure-Case Learning Curves: {focus_env}")
    axes[1].set_xlabel("Training steps")
    axes[1].set_ylabel("D4RL normalized score")
    axes[1].grid(alpha=0.25)

    handles0, labels0 = axes[0].get_legend_handles_labels()
    dedup0 = dict(zip(labels0, handles0))
    axes[0].legend(dedup0.values(), dedup0.keys(), frameon=False)

    handles1, labels1 = axes[1].get_legend_handles_labels()
    dedup1 = dict(zip(labels1, handles1))
    axes[1].legend(dedup1.values(), dedup1.keys(), frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(figures_dir / "05_stability_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_table(
    baseline_agg: pd.DataFrame,
    improved_agg: pd.DataFrame,
    best_ablation: pd.DataFrame,
    figures_dir: Path,
) -> None:
    merged = baseline_agg.merge(improved_agg, on="env", suffixes=("_baseline", "_improved"))
    merged = merged.merge(best_ablation, on="env", how="left")
    merged["final_delta"] = merged["final_score_mean_improved"] - merged["final_score_mean_baseline"]
    merged["update_speedup"] = merged["avg_wall_clock_per_update_ms_baseline"] / merged["avg_wall_clock_per_update_ms_improved"]
    merged["throughput_gain"] = merged["avg_replay_buffer_throughput_improved"] / merged["avg_replay_buffer_throughput_baseline"]
    merged["inference_speedup"] = merged["avg_inference_time_ms_baseline"] / merged["avg_inference_time_ms_improved"]

    display = pd.DataFrame(
        {
            "Env": merged["env"],
            "Baseline Final": merged["final_score_mean_baseline"].map(lambda x: f"{x:.2f}"),
            "Improved Final": merged["final_score_mean_improved"].map(lambda x: f"{x:.2f}"),
            "Δ Final": merged["final_delta"].map(lambda x: f"{x:+.2f}"),
            "Update Speedup": merged["update_speedup"].map(lambda x: f"{x:.2f}x"),
            "Throughput Gain": merged["throughput_gain"].map(lambda x: f"{x:.2f}x"),
            "Inference Speedup": merged["inference_speedup"].map(lambda x: f"{x:.2f}x"),
            "Best Ablation": merged["best_variant"],
        }
    )

    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.axis("off")
    table = ax.table(cellText=display.values, colLabels=display.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    ax.set_title("Metric Summary for Slides / README", pad=14)
    fig.tight_layout()
    fig.savefig(figures_dir / "06_summary_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figures(args: argparse.Namespace) -> list[Path]:
    baseline_agg = load_csv(args.baseline_aggregate)
    improved_agg = load_csv(args.improved_aggregate)
    ablation_agg = load_csv(args.ablation_aggregate)
    baseline_raw = load_csv(args.baseline_raw)
    improved_raw = load_csv(args.improved_raw)
    eval_histories = collect_eval_histories(args.results_root, args.improved_label)
    best_ablation = build_best_ablation_table(ablation_agg, args.improved_label)

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curves(eval_histories, baseline_agg, args.improved_label, args.figures_dir)
    plot_score_comparison(baseline_agg, improved_agg, best_ablation, args.improved_label, args.figures_dir)
    plot_systems_metrics(baseline_agg, improved_agg, args.improved_label, args.figures_dir)
    plot_ablation_heatmap(ablation_agg, args.improved_label, args.figures_dir)
    plot_stability_analysis(eval_histories, baseline_raw, improved_raw, args.improved_label, args.figures_dir)
    plot_summary_table(baseline_agg, improved_agg, best_ablation, args.figures_dir)

    return [
        args.figures_dir / "01_learning_curves.png",
        args.figures_dir / "02_score_comparison.png",
        args.figures_dir / "03_systems_metrics.png",
        args.figures_dir / "04_ablation_heatmap.png",
        args.figures_dir / "05_stability_analysis.png",
        args.figures_dir / "06_summary_table.png",
    ]


def main() -> None:
    sns.set_theme(style="whitegrid")
    args = parse_args()
    figure_paths = generate_figures(args)
    print("[done] generated figures:")
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
