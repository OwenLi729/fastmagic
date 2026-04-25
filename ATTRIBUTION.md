<!-- AI-generated: Claude, 2026-04-21 -->

# ATTRIBUTION

This file tracks AI-assisted and external resources used in the Fastmagic course project.

## AI Tools

### Tool Usage Log

| Date | Tool | Prompt / Task Description | Files | Kept / Modified Notes |
|------|------|----------------------------|-------|------------------------|
| 2026-04-21 | Claude Sonnet 4.6 | Scaffold full Fastmagic project structure and starter IQL implementation files | `requirements.txt`, `src/losses.py`, `src/networks.py`, `src/buffer.py`, `src/utils.py`, `src/train.py`, `src/evaluate.py`, `data/download_d4rl.py`, `notebooks/colab_train.ipynb`, `README.md`, `SETUP.md`, `ATTRIBUTION.md` | Initial generated scaffold; expected to be iteratively edited by project authors |
| 2026-04-23 | GitHub Copilot (GPT-5.3-Codex) | Fix Colab training notebook dependency/runtime flow after D4RL import failure; add baseline-first benchmark execution and baseline-vs-improved comparison export | `notebooks/colab_train.ipynb` | Replaced conflicting install cells with a Python 3.12-compatible D4RL stack, added validation + dataset cache cell, split benchmark execution into baseline then improved runs, and added comparison/Drive export cell |
| 2026-04-23 | GitHub Copilot (GPT-5.3-Codex) | Add explicit baseline controls in training pipeline for fair paper comparison | `src/train.py`, `src/buffer.py`, `src/benchmark_iql.py`, `README.md`, `SETUP.md` | Added `--baseline` and replay-device controls, CPU replay path for baseline, optional W&B import handling, and documentation commands for baseline vs improved runs |
| 2026-04-24 | GitHub Copilot (GPT-5.3-Codex) | Implement Tier-2 `torch.compile` contribution for benchmarkable training speedups | `src/train.py`, `src/benchmark_iql.py` | Added `--torch_compile` / compile mode/backend flags, compiled forward modules (value/Q/policy) with safe eager fallback, and benchmark CLI passthrough to compare compile on/off under identical settings |
| 2026-04-24 | GitHub Copilot (GPT-5.3-Codex) | Implement Tier-2 Parallel V/Q updates plus Tier-3 ablation sweeps and performance-metric persistence | `src/train.py`, `src/benchmark_iql.py` | Added `--parallel_vq_updates` critic update mode, benchmark ablation sweeps for `tau`, `beta`, and `n_hidden_layers`, and persisted all rubric performance metrics (update time, replay throughput, critic/actor ratio, inference time, normalized score) into run summaries and aggregate CSVs |
| 2026-04-24 | GitHub Copilot (GPT-5.3-Codex) | Add resume support for interrupted benchmark jobs on DCC/SLURM | `src/benchmark_iql.py`, `notebooks/training_script.py` | Added `--skip_existing` run skipping based on existing `summary.json`, wired default resume behavior via `--no_resume` override, enabling reruns to continue from completed checkpoints/results instead of restarting the whole sweep |
| 2026-04-25 | GitHub Copilot (GPT-5.4) | Audit committed benchmark outputs in `data/results/`, replace placeholder evaluation text with measured results only, and map the reported evidence to the course rubric | `README.md` | Replaced placeholder evaluation content with values copied from benchmark aggregate CSVs, summarized observed gains/limitations without inventing missing runs, and added rubric-oriented interpretation tied to committed result files |
| 2026-04-25 | GitHub Copilot (GPT-5.4) | Generate a reusable plotting pipeline for final-project figures from benchmark CSVs/logs while avoiding hallucinated data when inputs are missing | `src/generate_visualizations.py`, `requirements.txt` | Added a standalone matplotlib/seaborn/pandas visualization script that reconstructs learning curves from `eval_history.csv`, builds comparison/throughput/ablation/stability/summary figures, emits explicit placeholder figures for unavailable CP or paper-score inputs, and added plotting dependencies to `requirements.txt` |
| 2026-04-25 | GitHub Copilot (GPT-5.4) | Refactor the plotting pipeline to only generate figures justified by the currently committed benchmark data and remove unsupported placeholder outputs | `src/generate_visualizations.py`, `figures/` | Simplified the figure set to six data-backed outputs: learning curves, score comparison, systems metrics, ablation heatmap, stability analysis, and a summary table; removed unsupported CP/paper-assumption plots and regenerated the saved PNGs |
| 2026-04-25 | GitHub Copilot (GPT-5.4) | Perform repository-wide documentation and cleanup sweep for rubric alignment, figure integration, path consistency, and stale artifact removal | `README.md`, `SETUP.md`, `src/generate_visualizations.py`, `src/train.py`, `src/benchmark_iql.py`, `src/utils.py`, `notebooks/training_script.py`, `ATTRIBUTION.md`, `figures/` | Rewrote the README around rubric-required sections, embedded generated figures with captions, aligned defaults and docs to `data/results`, improved figure layout for presentation use, removed an unused checkpoint loader, fixed stale path usage in the benchmark helper, removed repo-owned cache artifacts (`.DS_Store`, `__pycache__`), and regenerated the slide-ready PNGs |

### AI-Generated Code Marking 

- AI-generated Python files include a file-level comment header.
- AI-assistance is also noted per-function or per-file.
- AI-generated public functions/classes include in-file comment tags.

## External Libraries

- PyTorch (`torch`)
- Gym (`gym`)
- D4RL (`d4rl`)
- NumPy (`numpy`)
- tqdm (`tqdm`)
- pandas (`pandas`)
- matplotlib (`matplotlib`)
- seaborn (`seaborn`)

## Datasets

- D4RL offline RL datasets (e.g., `hopper-medium-v2`)
- Download script: `data/download_d4rl.py`

## Notes for Course Submission

- Keep this file updated as code evolves.
- Record all AI-assisted changes before final submission.
- Ensure AI attribution is consistent with course policy and rubric.

## AI Usage

In ATTRIBUTION.md, provided a substantive account of how AI development tools were used in the project, including what was generated, what was modified, and what you had to debug, fix, or substantially rework (3 pts)

I used AI heavily for initial code generation while referencing the RLKit PyTorch implementation of Implicit Q-Learning in order to be able to quickly reimplement the algorithm as a baseline. I also used it to generate the training notebook/script so I could rapidly validate my code and ensure that I could use it to reproduce the results of the paper, and to be able to quickly test any improvements I made. I leveraged AI for debugging throughout the project.
AI was also helpful for initial scaffolding and to create the requirements.txt and quickly establish a directory structure which adhered to the rubric. While I used it to generate the ATTRIBUTION.md and README.md, I heavily augmented both for accuracy.
I had to rework the training notework and tune the hyperparameters myself, as well as evaluate the results against the paper. I also had to debug an issue where incomplete runs were being marked as completed during training runs (with AI assistance).
After I completed the runs and collected the training data, I used AI to help quickly parse the data, graph it, and summarize it in the Evaluations section. For the final reporting pass, I specifically used AI to verify which ablation values were actually present in the committed results, update the README with only measured CSV values, and generate reusable plotting code that refuses to fabricate missing CP-threshold or paper-reference data.