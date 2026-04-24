<!-- AI-generated: Claude, 2026-04-21 -->

# Fastmagic Setup Guide

## Local Development (macOS M2)

1. Create a Python environment (recommended 3.10):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Download a D4RL dataset cache:
   ```bash
   python data/download_d4rl.py --env hopper-medium-v2
   ```
4. Run a quick training sanity check (CPU/GPU depending on machine):
   ```bash
   python src/train.py --env hopper-medium-v2 --train_steps 1000 --log_interval 100 --profile
   ```

## Google Colab (Primary Training Environment)

1. Open [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb).
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run cells in order:
   - Install packages (`!pip install ...`)
   - Mount Google Drive
   - Clone/open repository
   - Choose replication preset (`mujoco` or `antmaze`), seeds, and environment subset
   - Cache all selected D4RL datasets
   - Launch the benchmark sweep
   - Inspect aggregate CSV results
   - Copy checkpoints/results to Drive

## Rigorous Paper Replication Workflow

The notebook now supports paper-style replication with:

- environment-family presets:
  - `mujoco`: `tau=0.7`, `beta=3.0`, `eval_interval=5000`, `eval_episodes=10`
  - `antmaze`: `tau=0.9`, `beta=10.0`, `eval_interval=100000`, `eval_episodes=100`
- multi-seed sweeps (default: seeds `0, 1, 2`)
- aggregate CSV generation across seeds
- per-run `summary.json` and `eval_history.csv`

You can also run the benchmark script directly:

```bash
python src/benchmark_iql.py --preset mujoco --seeds 0 1 2 --mixed_precision
python src/benchmark_iql.py --preset antmaze --seeds 0 1 2 --mixed_precision
```

### Baseline vs Improved on T4 (for rubric comparison)

Run the baseline (standard IQL, no speed targets enabled):

```bash
python src/train.py --env hopper-medium-v2 --seed 0 --train_steps 100000 --profile --baseline
```

Run the improved configuration (speed targets enabled):

```bash
python src/train.py --env hopper-medium-v2 --seed 0 --train_steps 100000 --profile --mixed_precision --replay_device gpu
```

For multi-seed sweeps:

```bash
python src/benchmark_iql.py --preset mujoco --seeds 0 1 2 --baseline --replay_device cpu --profile
python src/benchmark_iql.py --preset mujoco --seeds 0 1 2 --mixed_precision --replay_device gpu --profile
```

This produces directly comparable outputs in `results/` for wall-clock and score deltas.

Outputs are saved under:

- `results/benchmarks/`
- `models/benchmarks/`

## Key Training Flags

- `--env`: D4RL environment id (e.g., `hopper-medium-v2`)
- `--tau`: Expectile parameter for value loss
- `--beta`: Inverse temperature for AWR weighting
- `--mixed_precision`: Enables BF16 autocast updates on CUDA
- `--profile`: Prints aggregate timing metrics
- `--results_dir`: Directory for per-run summaries and evaluation history
- `--run_name`: Stable identifier used for benchmark aggregation

## Notes

- No compiled CUDA extensions are required.
- Checkpoints and config JSON files are saved under `models/` by default.
- Use `context/final_project_handout.html` for grading/rubric checks before final runs.
