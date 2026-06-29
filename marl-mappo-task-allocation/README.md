# marl-mappo-task-allocation

**MAPPO Task Allocation for Heterogeneous Human-Robot Teams**

This repository/distribution is named `marl-mappo-task-allocation`. Python import names remain as root-level modules such as `task_generator.py`, `heterogeneous_team_env.py`, and `train.py`, because hyphens are not valid in Python import identifiers.

This repository contains a multi-agent reinforcement learning framework for natural-language task allocation in a heterogeneous human-robot team. The core experiment uses an RLlib PPO setup with a MAPPO-style centralized-training/decentralized-execution (CTDE) structure: each actor receives local information, while the critic receives a global state.

The task allocation problem is modeled as a procurement-style bidding process. At each step, all agents receive the same task instruction and output a discrete bid from `0` to `10`. A bid of `0` means abstain. Eligible bidders compete, and the environment selects the lowest eligible bid as the winner.

## What is included

```text
marl-mappo-task-allocation/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE_TEMPLATE_MIT.txt
├── make_dataset.py
├── extract_eval_only_npz.py
├── task_generator.py
├── heterogeneous_team_env.py
├── train.py
├── task_allocation_interface.py
├── task_templates/
│   ├── README.md
│   └── task_templates.yaml
├── datasets/
│   ├── README.md
│   └── .gitkeep
├── exported_checkpoints/
│   └── README.md
├── results/
│   └── .gitkeep
└── scripts/
    ├── README.md
    ├── dataset_npz_to_txt.py
    ├── eligibility_report_basic.py
    ├── eligibility_report_env_sampling.py
    └── evaluate_checkpoint_on_evalset.py
```

## Main files

| File | Purpose |
|---|---|
| `task_generator.py` | Encodes task instructions, loads task templates, generates train/eval tasks, and saves/loads `.npz` datasets. |
| `task_templates/task_templates.yaml` | External task-template dataset used by `NLTaskGenerator`. |
| `heterogeneous_team_env.py` | RLlib multi-agent environment for task bidding and reward calculation. |
| `train.py` | Interactive training/debug script for PPO with MAPPO-style centralized critic. |
| `task_allocation_interface.py` | PySide6 GUI for loading a checkpoint, typing a task, optionally inferring requirements, and running deterministic bids. |
| `make_dataset.py` | Generates the static train/eval task dataset. |
| `extract_eval_only_npz.py` | Creates an eval-only `.npz` file from the full dataset. |
| `scripts/evaluate_checkpoint_on_evalset.py` | Evaluates a trained checkpoint on a fixed eval set and writes CSV/JSON results. |
| `scripts/eligibility_report_basic.py` | Prints per-agent eligibility statistics over the generated dataset. |
| `scripts/eligibility_report_env_sampling.py` | Audits the task distribution produced by the environment sampling mode. |
| `scripts/dataset_npz_to_txt.py` | Converts `.npz` datasets into human-readable text reports. |

## Method overview

The environment contains six agents with fixed capability vectors:

| Agent ID | Agent type | Capability vector `[mobility, manipulation, payload]` |
|---:|---|---|
| 0 | Mobile Robot | `[2, 1, 4]` |
| 1 | Mobile Manipulator | `[3, 4, 3]` |
| 2 | Legged Robot | `[4, 1, 1]` |
| 3 | Robotic Arm 1 | `[1, 3, 2]` |
| 4 | Robotic Arm 2 | `[1, 2, 1]` |
| 5 | Human | `[4, 4, 3]` |

Each task has a requirement vector in the same order:

```text
[mobility_class, manipulation_class, payload_class]
```

An agent is eligible if all capability dimensions satisfy:

```text
agent_capability >= task_requirement
```

The actor receives a local observation made from the agent capability vector plus the natural-language task embedding. The centralized critic receives a global state containing all agent capabilities, the task embedding, and eligibility flags.

## Requirement classes

The current task templates use classes from `1` to `4`:

### Mobility

| Class | Meaning |
|---:|---|
| 1 | Stationary / no mobility required |
| 2 | Planar or open-area mobility |
| 3 | High maneuverability in constrained spaces |
| 4 | Uneven terrain, stairs, ramps, obstacles, or irregular ground |

### Manipulation

| Class | Meaning |
|---:|---|
| 1 | No manipulation required |
| 2 | Short reach, up to about 50 cm |
| 3 | Medium reach, up to about 90 cm |
| 4 | Long reach, above about 90 cm |

### Payload

| Class | Meaning |
|---:|---|
| 1 | Up to 3 kg |
| 2 | More than 3 kg and up to 5 kg |
| 3 | More than 5 kg and up to 10 kg |
| 4 | More than 10 kg and up to 50 kg |

## Installation

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- The GUI requires `PySide6`.
- Training and evaluation require `ray[rllib]` and `torch`.
- Text embeddings require `sentence-transformers`.
- Loading `task_templates/task_templates.yaml` requires `PyYAML`.

## Generate the dataset

The task-template dataset is stored in:

```text
task_templates/task_templates.yaml
```

Generate the full train/eval dataset:

```bash
python make_dataset.py
```

This creates:

```text
datasets/tasks_v1.npz
```

Create a smaller eval-only dataset:

```bash
python extract_eval_only_npz.py \
  --in datasets/tasks_v1.npz \
  --out datasets/tasks_v1_eval_only.npz
```

Create human-readable dataset reports:

```bash
python scripts/dataset_npz_to_txt.py \
  --npz datasets/tasks_v1.npz \
  --out datasets/tasks_v1_grouped.txt

python scripts/dataset_npz_to_txt.py \
  --npz datasets/tasks_v1_eval_only.npz \
  --out datasets/eval_only.txt \
  --split eval
```

## Train a checkpoint

Run:

```bash
python train.py
```

The script opens an interactive menu:

```text
1. Full Training (Single Trial, PPO with Tune)
2. Quick Environment Test (TRAIN templates)
3. Quick Environment Test (EVAL templates)
4. Forced Bid Debug (1 step, bid=5 for all agents)
5. RLlib Action Mask Sanity Check
0. Exit
```

Choose option `1` for full training.

By default, the training code expects:

```text
datasets/tasks_v1.npz
```

and exports the best checkpoint to:

```text
exported_checkpoints/best_checkpoint
```

Training results are written to:

```text
results_mappo_bidding/
```

## Use the GUI interface

After training or providing a compatible RLlib checkpoint, run:

```bash
python task_allocation_interface.py
```

The interface lets you:

1. Load a checkpoint directory such as `exported_checkpoints/best_checkpoint`.
2. Enter a natural-language task instruction.
3. Set Mobility / Manipulation / Payload classes manually.
4. Optionally auto-infer the requirement classes from the instruction.
5. Run deterministic inference and inspect each agent's bid, eligibility, and winner status.

The auto-inference is heuristic. For exact parity with dataset-based evaluation, verify the inferred requirement classes before running bids.

## Evaluate a checkpoint automatically

Evaluate a trained checkpoint on the eval-only dataset:

```bash
python scripts/evaluate_checkpoint_on_evalset.py \
  --checkpoint exported_checkpoints/best_checkpoint \
  --eval_npz datasets/tasks_v1_eval_only.npz \
  --out_csv results/eval_rows.csv \
  --out_json results/eval_summary.json \
  --episode_len 50
```

The evaluator writes:

```text
results/eval_rows.csv
results/eval_summary.json
```

The evaluator can also read the original full dataset because it only uses the `eval_*` arrays:

```bash
python scripts/evaluate_checkpoint_on_evalset.py \
  --checkpoint exported_checkpoints/best_checkpoint \
  --eval_npz datasets/tasks_v1.npz \
  --out_csv results/eval_rows.csv \
  --out_json results/eval_summary.json \
  --episode_len 50
```

## Dataset and sampling diagnostics

Check how many tasks each agent is eligible for in the generated dataset:

```bash
python scripts/eligibility_report_basic.py
```

Audit the distribution of tasks actually sampled by the environment:

```bash
python scripts/eligibility_report_env_sampling.py
```

The sampling audit uses `category_weighted` sampling by default and samples many tasks, so it can take longer than the basic report.

## Action masking note

The environment always emits an `action_mask` in the observation. The behavior depends on configuration:

```text
use_action_mask=False  -> action_mask is all ones
use_action_mask=True   -> ineligible agents can only select bid 0
```

When `force_eligible_to_bid=True`, eligible agents are also prevented from selecting bid `0` by the mask.

Important: the environment does not correct invalid actions inside `step()`. The RLlib model must apply the mask to the logits. The model in `train.py` does this when the mask is present.

## What should be committed to GitHub?

Recommended to commit:

```text
README.md
requirements.txt
pyproject.toml
.gitignore
LICENSE or LICENSE_TEMPLATE_MIT.txt
*.py source files
task_templates/task_templates.yaml
task_templates/README.md
datasets/README.md
exported_checkpoints/README.md
scripts/*.py
scripts/README.md
```

Recommended not to commit by default:

```text
datasets/*.npz
datasets/*.txt
results/*
results_mappo_bidding/
exported_checkpoints/best_checkpoint/
ray_results/
__pycache__/
.venv/
```

The generated datasets are deterministic from `task_templates/task_templates.yaml`, so users can regenerate them locally. If you want to share a trained checkpoint, consider GitHub Releases or Git LFS instead of committing large checkpoint files directly.

## Reproducible run order

A complete run from a clean clone is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python make_dataset.py
python extract_eval_only_npz.py --in datasets/tasks_v1.npz --out datasets/tasks_v1_eval_only.npz

python train.py

python scripts/evaluate_checkpoint_on_evalset.py \
  --checkpoint exported_checkpoints/best_checkpoint \
  --eval_npz datasets/tasks_v1_eval_only.npz \
  --out_csv results/eval_rows.csv \
  --out_json results/eval_summary.json \
  --episode_len 50

python task_allocation_interface.py
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'task_generator'`

Run scripts from the repository root. The reviewed scripts in `scripts/` also add the repository root to `sys.path` so this should work:

```bash
python scripts/eligibility_report_basic.py
```

### `Task template dataset not found`

Check that this file exists:

```text
task_templates/task_templates.yaml
```

### `Dataset not found`

Generate it first:

```bash
python make_dataset.py
```

### GUI does not open

Make sure PySide6 is installed:

```bash
pip install PySide6
```

### Checkpoint does not load

Pass either an RLlib checkpoint directory, a folder containing `checkpoint_*` subfolders, or the exported best checkpoint:

```text
exported_checkpoints/best_checkpoint
```

## License

Choose a license before publishing. A MIT license template is included as:

```text
LICENSE_TEMPLATE_MIT.txt
```

Rename it to `LICENSE` and replace `<YOUR NAME OR ORGANIZATION>` if you want to use MIT.
