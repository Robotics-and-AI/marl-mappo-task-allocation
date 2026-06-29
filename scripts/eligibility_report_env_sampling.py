"""
Script: eligibility_report_env_sampling.py

What it does
------------
This script compares **agent eligibility** across:
1) The task dataset splits (train/eval) loaded from `./datasets/tasks_v1.npz`, and
2) Tasks **actually sampled by the environment** (`HeterogeneousTeamEnv`) under a given
   dataset sampling configuration (e.g., "permute", "random", "sequential", "category_weighted").

Eligibility is defined as: an agent is eligible for a task if, for each of the 3 requirement
dimensions, `capability >= requirement` (all values are in the raw 1..4 scale).

Outputs include:
- Eligibility counts and percentages per agent for dataset train/eval splits
- Eligibility counts and percentages per agent for empirically sampled env tasks
- "Lift" vs the train dataset baseline (ratio of sampled eligibility % to dataset eligibility %)
- The most frequently sampled task categories during env sampling

How to run
----------
1) Ensure you have the dataset at: ./datasets/tasks_v1.npz
2) Ensure these imports are available in your environment:
   - `task_generator.load_dataset_npz`
   - `heterogeneous_team_env.HeterogeneousTeamEnv`
3) Run:
   python scripts/eligibility_report_env_sampling.py

Notes
-----
- The environment stores task requirements in a normalized form; this script converts them back
  to raw 1..4 using `env._unnormalize(...)` before checking eligibility.
- Actions taken during sampling are trivial (all zeros) because the goal is to audit sampling
  distributions, not to solve tasks.
"""
# Allow running this script from the repository root as:
#   python scripts/<script_name>.py
# Python otherwise puts scripts/ on sys.path, not the project root.
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



import os
import numpy as np
from collections import Counter

from task_generator import load_dataset_npz
from heterogeneous_team_env import HeterogeneousTeamEnv

# Agent capability vectors in the environment's *raw* (unnormalized) scale (1..4),
# one row per agent, with 3 capability dimensions
RAW_CAPS = np.array(
    [
        [2, 1, 4],  # Mobile robot
        [3, 4, 3],  # Mobile manipulator
        [4, 1, 1],  # Legged robot
        [1, 3, 2],  # Robotic arm 1
        [1, 2, 1],  # Robotic arm 2
        [4, 4, 3],  # Human
    ],
    dtype=np.float32,
)

# Human-readable labels aligned with RAW_CAPS rows (index i corresponds to RAW_CAPS[i])
AGENT_NAMES = [
    "MobileRobot-0",
    "MobileManipulator-1",
    "LeggedRobot-2",
    "RoboticArm1-3",
    "RoboticArm2-4",
    "Human-5",
]

def eligible(cap, req_1to4):
    """
    Return True if capability vector `cap` satisfies requirement vector `req_1to4`
    in all dimensions (elementwise cap >= req)
    """
    return bool(np.all(cap >= req_1to4))

def dataset_eligibility(tasks):
    """
    Compute per-agent eligibility counts/fractions directly from a list of dataset tasks

    Each task is expected to be dict-like and contain:
      - t["requirements"]: length-3 vector in raw scale 1..4
    """
    # Stack requirements into a [N, 3] array (raw scale 1..4)
    reqs = np.array([t["requirements"] for t in tasks], dtype=np.float32)
    elig_counts = np.zeros((len(AGENT_NAMES),), dtype=np.int64)

    # For each task requirement vector, check each agent's capabilities
    for r in reqs:
        for i in range(len(AGENT_NAMES)):
            if eligible(RAW_CAPS[i], r):
                elig_counts[i] += 1

    # Convert counts to fractions of the split size.
    frac = elig_counts / max(1, len(tasks))
    return elig_counts, frac

def sampled_eligibility(env_config, total_tasks=20000, seed=0):
    """
    Empirically audit eligibility over tasks sampled by the environment.

    The environment is reset repeatedly and stepped with trivial actions to advance through
    tasks; on each step, the currently sampled task is recorded and eligibility is checked.

    Returns:
      sampled        : total number of tasks sampled
      elig_counts    : per-agent eligibility count over sampled tasks
      frac           : per-agent eligibility fraction over sampled tasks
      cat_counts     : Counter of sampled task categories
    """
    env = HeterogeneousTeamEnv(env_config)

    cat_counts = Counter()
    elig_counts = np.zeros((len(AGENT_NAMES),), dtype=np.int64)

    sampled = 0   # number of tasks sampled so far
    ep = 0        # episode counter (used to vary reset seed)

    while sampled < total_tasks:
        # Different seed per episode to reduce correlation in "permute" mode.
        _obs, _info = env.reset(seed=seed + ep)

        # Step through the episode until max steps or until we reach total_tasks.
        while sampled < total_tasks and env.current_step < env.max_steps:
            # The task being acted on at this step.
            task = env.current_task

            # Track category distribution (falls back to "unknown" if missing).
            cat_counts[task.get("category", "unknown")] += 1

            # Task requirements are stored normalized in env; convert back to raw 1..4 for comparison.
            req_1to4 = env._unnormalize(task["requirements"]).astype(np.float32)

            # Count eligibility for each agent for this sampled task.
            for i in range(len(AGENT_NAMES)):
                if eligible(RAW_CAPS[i], req_1to4):
                    elig_counts[i] += 1

            # Step with trivial actions (the audit goal is sampling behavior, not policy performance).
            actions = {aid: 0 for aid in env.possible_agents}
            _obs, _rew, _term, _trunc, _info = env.step(actions)

            sampled += 1

        ep += 1

    # Convert counts to fractions over the sampled tasks.
    frac = elig_counts / float(max(1, sampled))
    return sampled, elig_counts, frac, cat_counts

def pretty_print(title, counts, frac):
    """Helper to print per-agent eligibility counts and percentages with consistent formatting."""
    print("\n" + "=" * 90)
    print(title)
    for i, name in enumerate(AGENT_NAMES):
        print(f"{name:20s} eligible: {int(counts[i]):8d} ({frac[i]*100:6.2f}%)")

def main():
    # Resolve dataset path relative to current working directory
    ds_path = os.path.abspath("./datasets/tasks_v1.npz")

    # Load tasks for both splits from the dataset file
    train_tasks, eval_tasks = load_dataset_npz(ds_path)

    # 1) Dataset baseline eligibility (what proportion of tasks each agent can do in the raw dataset)
    tr_counts, tr_frac = dataset_eligibility(train_tasks)
    ev_counts, ev_frac = dataset_eligibility(eval_tasks)

    pretty_print(f"DATASET BASELINE (TRAIN) tasks={len(train_tasks)}", tr_counts, tr_frac)
    pretty_print(f"DATASET BASELINE (EVAL)  tasks={len(eval_tasks)}", ev_counts, ev_frac)

    # 2) Empirical env sampling audit (choose the sampling mode you want to verify)
    env_config = {
        "max_steps": 50,
        "train_mode": True,            # sampling cache used only in train_mode for category_weighted
        "dataset_mode": True,
        "dataset_path": ds_path,
        # Try: "permute", "random", "sequential", "category_weighted"
        "dataset_sampling": "category_weighted",
        # Match your training config if using category_weighted.
        "category_mix_uniform": 0.30,
        "category_weight_nonspecialist": 0.20,
        "category_weight_floor": 0.05,
        "print_category_sampling_cache": True,
        # Masking/other knobs don't affect which task is sampled next (as noted in code comment)
        "use_action_mask": False,
        "force_eligible_to_bid": False,
    }

    # Sample many tasks from the env to estimate the actual sampling distribution
    sampled_n, sm_counts, sm_frac, cat_counts = sampled_eligibility(env_config, total_tasks=20000, seed=0)

    pretty_print(
        f"EMPIRICAL ENV SAMPLING tasks_sampled={sampled_n} sampling={env_config['dataset_sampling']}",
        sm_counts,
        sm_frac,
    )

    # 3) Lift vs dataset baseline (train split)
    #    >1 means the env sampling produced a higher eligibility fraction than the dataset baseline
    print("\n" + "-" * 90)
    print("ELIGIBILITY LIFT vs TRAIN DATASET ( >1 means sampled more often than dataset proportion )")
    for i, name in enumerate(AGENT_NAMES):
        base = float(tr_frac[i])
        lift = (float(sm_frac[i]) / base) if base > 1e-12 else float("inf")
        print(
            f"{name:20s} lift={lift:7.3f}  "
            f"(sampled={sm_frac[i]*100:6.2f}%  base={tr_frac[i]*100:6.2f}%)"
        )

    # 4) Category distribution of sampled tasks (top 15)
    print("\n" + "-" * 90)
    print("TOP SAMPLED CATEGORIES:")
    for cat, c in cat_counts.most_common(15):
        print(f"{cat:20s} count={c:8d} ({100.0*c/max(1,sampled_n):6.2f}%)")

if __name__ == "__main__":
    main()