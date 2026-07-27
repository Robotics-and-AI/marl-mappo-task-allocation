# marl-mappo-task-allocation

MAPPO-based task allocation framework for heterogeneous human-robot teams using natural-language task instructions.

This repository contains a multi-agent reinforcement learning environment built with RLlib PPO in a MAPPO-style centralized-training/decentralized-execution (CTDE) setup. Agents receive natural-language task instructions, submit discrete bids, and the environment assigns the task to the lowest eligible bidder according to capability constraints.


## Main features

- Natural-language task generation from external YAML templates.
- Requirement representation using mobility, manipulation reach, and payload classes.
- Six-agent heterogeneous team with fixed capability vectors.
- RLlib multi-agent PPO training with a MAPPO-style centralized critic.
- Procurement-style bidding task allocation.
- Optional action masking for eligibility-constrained bidding.
- Dataset generation, checkpoint evaluation, sampling diagnostics, and a manual GUI interface.

## Repository structure

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
├── scripts/
│   ├── README.md
│   ├── dataset_npz_to_txt.py
│   ├── eligibility_report_basic.py
│   ├── eligibility_report_env_sampling.py
│   └── evaluate_checkpoint_on_evalset.py
└── docs/
    └── RELEASE_CHECKLIST.md
```

## Core files

| File | Purpose |
|---|---|
| `task_generator.py` | Loads task templates, generates train/eval tasks, encodes instructions, and saves/loads datasets. |
| `task_templates/task_templates.yaml` | External YAML dataset of parameterized task templates and requirement labels. |
| `heterogeneous_team_env.py` | RLlib multi-agent environment for bidding, eligibility checking, and reward calculation. |
| `train.py` | Interactive PPO/MAPPO training and debugging script. |
| `task_allocation_interface.py` | PySide6 GUI for manual checkpoint testing and optional requirement auto-inference. |
| `make_dataset.py` | Generates the full train/eval dataset. |
| `extract_eval_only_npz.py` | Creates an eval-only dataset from the full `.npz` dataset. |
| `scripts/evaluate_checkpoint_on_evalset.py` | Evaluates a checkpoint on a fixed eval set and writes CSV/JSON results. |
| `scripts/eligibility_report_basic.py` | Reports per-agent eligibility over the generated dataset. |
| `scripts/eligibility_report_env_sampling.py` | Audits the task distribution produced by environment sampling. |
| `scripts/dataset_npz_to_txt.py` | Converts `.npz` datasets into human-readable text reports. |

## Method overview

Each task is represented by a natural-language instruction and a requirement vector:

```text
(mobility_level, manipulation_level, payload_level)
```

Each agent has a fixed capability vector in the same order. An agent is eligible for a task when all capability dimensions satisfy:

```text
agent_capability >= task_requirement
```

At each environment step, all agents receive the task instruction and submit a discrete bid from `0` to `10`. A bid of `0` means that the agent chooses not to participate. Among eligible non-zero bidders, the environment selects the lowest bid as the winner.

The actor receives local information: the agent capability vector and the task embedding. The centralized critic receives a global state containing all agent capabilities, the task embedding, and eligibility flags.

## Team capabilities

| Agent ID | Agent type | Capability vector `(mobility, manipulation, payload)` |
|---:|---|---|
| 0 | Mobile Robot | `(2, 1, 4)` |
| 1 | Mobile Manipulator | `(3, 4, 3)` |
| 2 | Legged Robot | `(4, 1, 1)` |
| 3 | Robotic Arm 1 | `(1, 3, 2)` |
| 4 | Robotic Arm 2 | `(1, 2, 1)` |
| 5 | Human | `(4, 4, 3)` |

## Requirement classes

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

Main dependencies include `ray[rllib]`, `torch`, `gymnasium`, `sentence-transformers`, `PySide6`, `PyYAML`, `numpy`, and `tabulate`.

## Generate datasets

The task templates are stored in:

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

Create an eval-only dataset:

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

By default, training expects:

```text
datasets/tasks_v1.npz
```

and exports the best checkpoint to:

```text
exported_checkpoints/best_checkpoint
```

Training outputs are written to:

```text
results_mappo_bidding/
```

## Evaluate a checkpoint

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

The evaluator can also use the original full dataset because it only reads the `eval_*` arrays:

```bash
python scripts/evaluate_checkpoint_on_evalset.py \
  --checkpoint exported_checkpoints/best_checkpoint \
  --eval_npz datasets/tasks_v1.npz \
  --out_csv results/eval_rows.csv \
  --out_json results/eval_summary.json \
  --episode_len 50
```

## Use the GUI interface

After training or providing a compatible RLlib checkpoint, run:

```bash
python task_allocation_interface.py
```

The interface lets you:

1. Load a checkpoint directory, such as `exported_checkpoints/best_checkpoint`.
2. Enter a natural-language task instruction.
3. Set Mobility / Manipulation / Payload classes manually.
4. Optionally auto-infer requirement classes from the instruction.
5. Run deterministic inference and inspect each agent's bid, eligibility, and winner status.

The auto-inference is heuristic. For exact parity with dataset-based evaluation, verify the inferred requirement classes before running bids.

## Diagnostic scripts

Check how many tasks each agent is eligible for in the generated dataset:

```bash
python scripts/eligibility_report_basic.py
```

Audit the distribution of tasks sampled by the environment:

```bash
python scripts/eligibility_report_env_sampling.py
```

The sampling audit uses `category_weighted` sampling by default and samples many tasks, so it can take longer than the basic report.

## Action masking

The environment always emits an `action_mask` in the observation. Its behavior depends on configuration:

```text
use_action_mask=False  -> action_mask is all ones
use_action_mask=True   -> ineligible agents can only select bid 0
```

When `force_eligible_to_bid=True`, eligible agents are also prevented from selecting bid `0` by the mask.

Important: the environment does not correct invalid actions inside `step()`. The RLlib model must apply the mask to the logits. The model in `train.py` does this when the mask is present.

## Clean run order

A typical run from a clean clone is:

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

## Files to commit

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
datasets/.gitkeep
exported_checkpoints/README.md
results/.gitkeep
scripts/*.py
scripts/README.md
docs/RELEASE_CHECKLIST.md
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

## Troubleshooting

### `ModuleNotFoundError: No module named 'task_generator'`

Run scripts from the repository root:

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

If you choose MIT, review the template, fill in the copyright information, and rename it to:

```text
LICENSE
```
