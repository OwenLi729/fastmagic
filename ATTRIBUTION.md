<!-- AI-generated: Claude, 2026-04-21 -->

# ATTRIBUTION

This file tracks AI-assisted and external resources used in the Fastmagic course project.

## AI Tools

### Tool Usage Log

| Date | Tool | Prompt / Task Description | Files | Kept / Modified Notes |
|------|------|----------------------------|-------|------------------------|
| 2026-04-21 | Claude | Scaffold full Fastmagic project structure and starter IQL implementation files | `requirements.txt`, `src/losses.py`, `src/networks.py`, `src/buffer.py`, `src/utils.py`, `src/train.py`, `src/evaluate.py`, `data/download_d4rl.py`, `notebooks/colab_train.ipynb`, `README.md`, `SETUP.md`, `ATTRIBUTION.md` | Initial generated scaffold; expected to be iteratively edited by project authors |
| 2026-04-23 | GitHub Copilot (GPT-5.3-Codex) | Fix Colab training notebook dependency/runtime flow after D4RL import failure; add baseline-first benchmark execution and baseline-vs-improved comparison export | `notebooks/colab_train.ipynb` | Replaced conflicting install cells with a Python 3.12-compatible D4RL stack, added validation + dataset cache cell, split benchmark execution into baseline then improved runs, and added comparison/Drive export cell |
| 2026-04-23 | GitHub Copilot (GPT-5.3-Codex) | Add explicit baseline controls in training pipeline for fair paper comparison | `src/train.py`, `src/buffer.py`, `src/benchmark_iql.py`, `README.md`, `SETUP.md` | Added `--baseline` and replay-device controls, CPU replay path for baseline, optional W&B import handling, and documentation commands for baseline vs improved runs |
| 2026-04-24 | GitHub Copilot (GPT-5.3-Codex) | Implement Tier-2 `torch.compile` contribution for benchmarkable training speedups | `src/train.py`, `src/benchmark_iql.py` | Added `--torch_compile` / compile mode/backend flags, compiled forward modules (value/Q/policy) with safe eager fallback, and benchmark CLI passthrough to compare compile on/off under identical settings |
| 2026-04-24 | GitHub Copilot (GPT-5.3-Codex) | Implement Tier-2 Parallel V/Q updates plus Tier-3 ablation sweeps and performance-metric persistence | `src/train.py`, `src/benchmark_iql.py` | Added `--parallel_vq_updates` critic update mode, benchmark ablation sweeps for `tau`, `beta`, and `n_hidden_layers`, and persisted all rubric performance metrics (update time, replay throughput, critic/actor ratio, inference time, normalized score) into run summaries and aggregate CSVs |

### AI-Generated Code Marking 

- AI-generated Python files include a file-level comment header.
- AI-assistance is also noted per-function or per-file.
- AI-generated public functions/classes include in-file comment tags.

## External Libraries

- PyTorch (`torch`)
- Gym (`gym`)
- D4RL (`d4rl`)
- NumPy (`numpy`)
- tqdm (`tqdm`) used?

## Datasets

- D4RL offline RL datasets (e.g., `hopper-medium-v2`)
- Download script: `data/download_d4rl.py`

## Notes for Course Submission

- Keep this file updated as code evolves.
- Record all AI-assisted changes before final submission.
- Ensure AI attribution is consistent with course policy and rubric.

## AI Usage

I used AI heavily for initial code generation while referencing the RLKit PyTorch implementation of Implicit Q-Learning in order to be able to quickly reimplement the algorithm as a baseline. I also used it to generate the training notebook so I could rapidly validate my code and ensure that I could use it to reproduce the results of the paper, and to be able to quickly test any improvements I made. I leveraged AI for debugging throughout the project.
AI was also helpful for initial scaffolding and to create the requirements.txt and quickly establish a directory structure which adhered to the rubric. While I used it to generate the ATTRIBUTION.md and README.md, I heavily augmented both for accuracy.