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

### AI-Generated Code Marking 

- AI-generated Python files include a file-level comment header.
- AI-assistance is also noted per-function or per-file.
- AI-generated public functions/classes include in-file comment tags.

## External Libraries

- PyTorch (`torch`)
- Gym (`gym`)
- D4RL (`d4rl`)
- NumPy (`numpy`)
- Weights & Biases (`wandb`) used?
- tqdm (`tqdm`) used?

## Datasets

- D4RL offline RL datasets (e.g., `hopper-medium-v2`)
- Download script: `data/download_d4rl.py`

## Notes for Course Submission

- Keep this file updated as code evolves.
- Record all AI-assisted changes before final submission.
- Ensure AI attribution is consistent with course policy and rubric.
