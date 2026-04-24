<!-- AI-generated: Claude, 2026-04-21 -->

# Fastmagic

## Project Title

**Fastmagic: A PyTorch Reimplementation of Implicit Q-Learning (IQL) on D4RL**

## What it Does

Fastmagic is an offline reinforcement learning project that reimplements IQL in PyTorch and benchmarks performance-oriented training features for Google Colab T4 GPUs. It includes:

- IQL core components: value network, twin Q-networks, and Gaussian policy
- Vectorized IQL losses (expectile regression + AWR policy objective)
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

## Individual Contributions

- Member 1: **TBD**
- Member 2: **TBD**
- Member 3: **TBD**

## Repository Layout

```text
fastmagic/
├── src/
│   ├── networks.py
│   ├── losses.py
│   ├── buffer.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── data/
│   └── download_d4rl.py
├── models/
├── notebooks/
│   └── colab_train.ipynb
├── docs/
├── videos/
├── context/
├── requirements.txt
├── SETUP.md
└── ATTRIBUTION.md
```

## Attribution Note

AI-assisted scaffolding and code generation are documented in [ATTRIBUTION.md](ATTRIBUTION.md).

Inspired/adapted from the PyTorch implementation of IQL in RLKit:
https://github.com/rail-berkeley/rlkit/
