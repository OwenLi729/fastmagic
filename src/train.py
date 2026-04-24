# AI-generated: Claude, 2026-04-23
# Portions of the update structure and hyperparameter semantics are adapted from
# RLKit's `rlkit/torch/sac/iql_trainer.py` and example IQL launchers.
"""Training entry point for a research-baseline Implicit Q-Learning agent."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import wandb
from tqdm import trange

from buffer import GPUReplayBuffer, ReplayBatch
from evaluate import evaluate_policy
from losses import awr_policy_loss, awr_weights, expectile_loss
from networks import GaussianPolicy, TwinQNetwork, ValueNetwork
from utils import CudaEventTimer, save_checkpoint, set_seed


PAPER_BENCHMARK_ENVS: tuple[str, ...] = (
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
)

MUJOCO_BENCHMARK_ENVS: tuple[str, ...] = tuple(env for env in PAPER_BENCHMARK_ENVS if "antmaze" not in env)
ANTMAZE_BENCHMARK_ENVS: tuple[str, ...] = tuple(env for env in PAPER_BENCHMARK_ENVS if "antmaze" in env)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Apply Polyak averaging from `source` into `target`."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline IQL training."""
    parser = argparse.ArgumentParser(description="Train a faithful IQL baseline on D4RL")
    parser.add_argument("--env", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.7, help="Expectile coefficient for value regression.")
    parser.add_argument(
        "--beta",
        type=float,
        default=3.0,
        help="Actor inverse-temperature from the IQL paper: weights = exp(beta * adv).",
    )
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--target_update_rate", type=float, default=0.005)
    parser.add_argument("--target_update_period", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_hidden_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--qf_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip_score", type=float, default=100.0)
    parser.add_argument("--train_steps", type=int, default=1_000_000)
    parser.add_argument("--eval_interval", type=int, default=5_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=1_000)
    parser.add_argument("--checkpoint_interval", type=int, default=50_000)
    parser.add_argument("--checkpoint_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--deterministic_torch", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="fastmagic-iql")
    return parser.parse_args()


def build_run_directories(args: argparse.Namespace) -> tuple[Path, Path, str]:
    """Create checkpoint/result directories for a single training run."""
    run_name = args.run_name or f"{args.env}_seed{args.seed}"
    checkpoint_dir = Path(args.checkpoint_dir) / run_name
    results_dir = Path(args.results_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, results_dir, run_name


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_eval_history(path: Path, eval_history: list[dict[str, float | int]]) -> None:
    """Persist evaluation scores as CSV for multi-seed aggregation."""
    lines = ["step,d4rl_normalized_score"]
    for row in eval_history:
        lines.append(f"{row['step']},{row['d4rl_normalized_score']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_models(
    replay: GPUReplayBuffer,
    hidden_dim: int,
    n_hidden_layers: int,
    device: torch.device,
) -> tuple[ValueNetwork, TwinQNetwork, TwinQNetwork, GaussianPolicy]:
    """Instantiate IQL networks on the requested device."""
    value_net = ValueNetwork(
        state_dim=replay.state_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
    ).to(device)
    q_net = TwinQNetwork(
        state_dim=replay.state_dim,
        action_dim=replay.action_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
    ).to(device)
    target_q_net = TwinQNetwork(
        state_dim=replay.state_dim,
        action_dim=replay.action_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
    ).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    policy = GaussianPolicy(
        state_dim=replay.state_dim,
        action_dim=replay.action_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
    ).to(device)
    return value_net, q_net, target_q_net, policy


def make_optimizers(
    value_net: ValueNetwork,
    q_net: TwinQNetwork,
    policy: GaussianPolicy,
    policy_lr: float,
    qf_lr: float,
    weight_decay: float,
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
]:
    """Create separate optimizers mirroring RLKit's trainer structure."""
    q1_optimizer = torch.optim.Adam(q_net.q1.parameters(), lr=qf_lr, weight_decay=weight_decay)
    q2_optimizer = torch.optim.Adam(q_net.q2.parameters(), lr=qf_lr, weight_decay=weight_decay)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=qf_lr, weight_decay=weight_decay)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr, weight_decay=weight_decay)
    return q1_optimizer, q2_optimizer, value_optimizer, policy_optimizer


def compute_iql_losses(
    batch: ReplayBatch,
    value_net: ValueNetwork,
    q_net: TwinQNetwork,
    target_q_net: TwinQNetwork,
    policy: GaussianPolicy,
    discount: float,
    reward_scale: float,
    tau: float,
    beta: float,
    clip_score: float,
) -> dict[str, torch.Tensor]:
    """Compute paper/RLKit-aligned IQL losses for a batch."""
    q1_pred, q2_pred = q_net(batch.observations, batch.actions)

    with torch.no_grad():
        target_v = value_net(batch.next_observations)
        q_target = reward_scale * batch.rewards + (1.0 - batch.terminals) * discount * target_v
        q_dataset = target_q_net.min_q(batch.observations, batch.actions)

    v_pred = value_net(batch.observations)
    value_loss = expectile_loss(q_dataset - v_pred, tau=tau)
    q1_loss = F.mse_loss(q1_pred, q_target)
    q2_loss = F.mse_loss(q2_pred, q_target)

    advantages = q_dataset - v_pred
    log_prob = policy.log_prob(batch.observations, batch.actions)
    policy_loss = awr_policy_loss(
        advantages=advantages,
        log_prob=log_prob,
        beta=beta,
        max_weight=clip_score,
    )
    weights = awr_weights(advantages=advantages, beta=beta, max_weight=clip_score)

    return {
        "q1_loss": q1_loss,
        "q2_loss": q2_loss,
        "value_loss": value_loss,
        "policy_loss": policy_loss,
        "q_target": q_target.detach(),
        "advantages": advantages.detach(),
        "weights": weights.detach(),
    }


def train(args: argparse.Namespace) -> None:
    """Train IQL on an offline D4RL dataset."""
    set_seed(args.seed, deterministic=args.deterministic_torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay = GPUReplayBuffer(env_name=args.env, device=device)
    value_net, q_net, target_q_net, policy = build_models(
        replay=replay,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        device=device,
    )
    q1_optimizer, q2_optimizer, value_optimizer, policy_optimizer = make_optimizers(
        value_net=value_net,
        q_net=q_net,
        policy=policy,
        policy_lr=args.policy_lr,
        qf_lr=args.qf_lr,
        weight_decay=args.weight_decay,
    )

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    checkpoint_dir, results_dir, run_name = build_run_directories(args)
    save_json(results_dir / "config.json", dict(vars(args), run_name=run_name))

    autocast_ctx = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if args.mixed_precision and device.type == "cuda"
        else (lambda: nullcontext())
    )

    total_update_ms = 0.0
    total_inference_ms = 0.0
    total_critic_ms = 0.0
    total_actor_ms = 0.0
    profiled_steps = 0
    best_score = float("-inf")
    final_score: float | None = None
    eval_history: list[dict[str, float | int]] = []

    for step in trange(1, args.train_steps + 1, desc="Training IQL baseline"):
        batch, replay_throughput = replay.sample_with_throughput(args.batch_size)

        update_timer = CudaEventTimer(enabled=(device.type == "cuda"))
        critic_timer = CudaEventTimer(enabled=(device.type == "cuda"))
        actor_timer = CudaEventTimer(enabled=(device.type == "cuda"))

        update_timer.start()
        critic_timer.start()
        with autocast_ctx():
            losses = compute_iql_losses(
                batch=batch,
                value_net=value_net,
                q_net=q_net,
                target_q_net=target_q_net,
                policy=policy,
                discount=args.discount,
                reward_scale=args.reward_scale,
                tau=args.tau,
                beta=args.beta,
                clip_score=args.clip_score,
            )

        q1_optimizer.zero_grad(set_to_none=True)
        q2_optimizer.zero_grad(set_to_none=True)
        (losses["q1_loss"] + losses["q2_loss"]).backward(retain_graph=True)
        q1_optimizer.step()
        q2_optimizer.step()

        value_optimizer.zero_grad(set_to_none=True)
        losses["value_loss"].backward()
        value_optimizer.step()
        critic_ms = critic_timer.stop()

        actor_timer.start()
        policy_optimizer.zero_grad(set_to_none=True)
        losses["policy_loss"].backward()
        policy_optimizer.step()
        actor_ms = actor_timer.stop()

        if step % args.target_update_period == 0:
            soft_update(target_q_net.q1, q_net.q1, tau=args.target_update_rate)
            soft_update(target_q_net.q2, q_net.q2, tau=args.target_update_rate)

        wall_clock_per_update_ms = update_timer.stop()

        inference_timer = CudaEventTimer(enabled=(device.type == "cuda"))
        inference_timer.start()
        _ = policy.act(batch.observations[:1], deterministic=True)
        inference_time_ms = inference_timer.stop()

        critic_actor_update_ratio = critic_ms / max(actor_ms, 1e-8) if actor_ms > 0.0 else 0.0

        metrics: dict[str, float] = {
            "wall_clock_per_update_ms": wall_clock_per_update_ms,
            "replay_buffer_throughput": replay_throughput,
            "critic_actor_update_ratio": critic_actor_update_ratio,
            "inference_time_ms": inference_time_ms,
            "value_loss": float(losses["value_loss"].item()),
            "q1_loss": float(losses["q1_loss"].item()),
            "q2_loss": float(losses["q2_loss"].item()),
            "policy_loss": float(losses["policy_loss"].item()),
            "advantage_mean": float(losses["advantages"].mean().item()),
            "advantage_weight_mean": float(losses["weights"].mean().item()),
        }

        if step % args.eval_interval == 0:
            metrics["d4rl_normalized_score"] = evaluate_policy(
                policy=policy,
                env_name=args.env,
                device=device,
                n_episodes=args.eval_episodes,
            )
            final_score = float(metrics["d4rl_normalized_score"])
            best_score = max(best_score, final_score)
            eval_history.append({"step": step, "d4rl_normalized_score": final_score})
            save_eval_history(results_dir / "eval_history.csv", eval_history)
            save_json(
                results_dir / "summary.json",
                {
                    "run_name": run_name,
                    "env": args.env,
                    "seed": args.seed,
                    "best_d4rl_normalized_score": best_score,
                    "final_d4rl_normalized_score": final_score,
                    "last_eval_step": step,
                    "num_evaluations": len(eval_history),
                },
            )

        if args.use_wandb:
            wandb.log(metrics, step=step)

        if step % args.log_interval == 0:
            print(
                f"step={step} "
                f"v_loss={metrics['value_loss']:.4f} "
                f"q1_loss={metrics['q1_loss']:.4f} "
                f"q2_loss={metrics['q2_loss']:.4f} "
                f"pi_loss={metrics['policy_loss']:.4f} "
                f"update_ms={metrics['wall_clock_per_update_ms']:.3f} "
                f"replay_sps={metrics['replay_buffer_throughput']:.1f}"
            )

        if args.profile:
            total_update_ms += wall_clock_per_update_ms
            total_inference_ms += inference_time_ms
            total_critic_ms += critic_ms
            total_actor_ms += actor_ms
            profiled_steps += 1

        if step % args.checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_path=checkpoint_dir / f"iql_step_{step}.pt",
                step=step,
                models={
                    "value_net": value_net,
                    "q_net": q_net,
                    "target_q_net": target_q_net,
                    "policy": policy,
                },
                optimizers={
                    "q1_optimizer": q1_optimizer,
                    "q2_optimizer": q2_optimizer,
                    "value_optimizer": value_optimizer,
                    "policy_optimizer": policy_optimizer,
                },
                config=vars(args),
            )

    if final_score is None:
        save_json(
            results_dir / "summary.json",
            {
                "run_name": run_name,
                "env": args.env,
                "seed": args.seed,
                "best_d4rl_normalized_score": None,
                "final_d4rl_normalized_score": None,
                "last_eval_step": None,
                "num_evaluations": 0,
            },
        )

    if args.use_wandb:
        wandb.finish()

    if args.profile and profiled_steps > 0:
        print(
            "[profile] "
            f"avg_update_ms={total_update_ms / profiled_steps:.4f} "
            f"avg_critic_ms={total_critic_ms / profiled_steps:.4f} "
            f"avg_actor_ms={total_actor_ms / profiled_steps:.4f} "
            f"avg_inference_ms={total_inference_ms / profiled_steps:.4f}"
        )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
