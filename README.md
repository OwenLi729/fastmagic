<!-- AI-assisted: Claude, 2026-04-21 -->

# Fastmagic

## Project Title

**Fastmagic: A PyTorch Reimplementation of Implicit Q-Learning (IQL) on D4RL**

## What it Does

Fastmagic is an offline reinforcement learning project for COMPSCI 372, taught by Dr. Brandon Fain. It reimplements the Implicit Q-Learning Paper (Kostrikov) in PyTorch and tests and measures speed improvements on a Google Colab T4 GPU. It includes:

- IQL core components
- Vectorized IQL losses 
- GPU-resident replay buffer from D4RL data
- BF16 mixed precision update path using `torch.autocast('cuda', dtype=torch.bfloat16)`
- CUDA-event-based timing for wall-clock training and inference metrics

## Quick Start

1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
2. Cache D4RL data:
	```bash
	python data/download_d4rl.py --env hopper-medium-v2
	```
3. Train:
	```bash
	python src/train.py --env hopper-medium-v2 --tau 0.7 --beta 3.0 --mixed_precision --profile
	```
4. Colab workflow: use [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb)

### Baseline Comparison Runs (T4)

- Standard IQL baseline (no optimization targets):
	```bash
	python src/train.py --env hopper-medium-v2 --seed 0 --train_steps 100000 --profile --baseline
	```
- Improved path (mixed precision + GPU replay storage):
	```bash
	python src/train.py --env hopper-medium-v2 --seed 0 --train_steps 100000 --profile --mixed_precision --replay_device gpu
	```

## Video Links

- Demo video: **TBD**
- Technical walkthrough: **TBD**

## Evaluation

### Core Metrics

Training logs include:

- `wall_clock_per_update_ms`
- `replay_buffer_throughput`
- `d4rl_normalized_score`
- `inference_time_ms`

### Results Table (Placeholder)

| Env | Tau | Beta | D4RL Normalized Score | Update Time (ms) | Notes |
|-----|-----|------|-----------------------|------------------|-------|
| hopper-medium-v2 | 0.7 | 3.0 | TBD | TBD | Baseline |

### Ablations (Planned)

- `tau ∈ {0.5, 0.7, 0.8, 0.9}`
- `beta ∈ {1.0, 3.0, 10.0}`
- Value network depth: 1 vs 2 vs 3 hidden layers

## Contributors

- Owen Li

## Attribution Note

AI-assisted scaffolding and code generation are documented in [ATTRIBUTION.md](ATTRIBUTION.md).

Inspired/adapted from the PyTorch implementation of IQL in RLKit:
https://github.com/rail-berkeley/rlkit/
