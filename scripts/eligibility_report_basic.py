"""
Script: eligibility_report_basic.py

What it does
------------
Loads a task dataset from `./datasets/tasks_v1.npz` and prints, for each split
(train/eval), how many tasks each agent is *eligible* to perform.

Each task has a 3D requirement vector (shape [3]) with values in the range 1..4.
Each agent has a corresponding 3D capability vector (also 1..4). An agent is
considered eligible for a task if, for every dimension, `capability >= requirement`.

How to run
----------
1) Ensure you have the dataset at: ./datasets/tasks_v1.npz
2) Ensure `task_generator.load_dataset_npz` is importable in your environment.
3) Run:
   python scripts/eligibility_report_basic.py

It will print an eligibility count and percentage per agent for both splits.
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
from task_generator import load_dataset_npz

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

def eligible(cap, req):
    """
    Return True if capability vector `cap` satisfies requirement vector `req`
    in all dimensions (elementwise cap >= req).
    """
    return bool(np.all(cap >= req))

def main():
    # Resolve dataset path relative to current working directory
    ds_path = os.path.abspath("./datasets/tasks_v1.npz")

    # Load tasks for both splits from the .npz dataset file
    # Expected: iterable of dict-like tasks with key "requirements" -> length-3 vector
    train_tasks, eval_tasks = load_dataset_npz(ds_path)

    # Process and print stats for each split.
    for split_name, tasks in [("train", train_tasks), ("eval", eval_tasks)]:
        # Stack requirements into a [N, 3] array (values are expected to be in 1..4)
        reqs = np.array([t["requirements"] for t in tasks], dtype=np.float32)

        # Count, for each agent, how many tasks it can satisfy
        elig_counts = np.zeros((len(AGENT_NAMES),), dtype=np.int64)
        for r in reqs:
            for i in range(len(AGENT_NAMES)):
                if eligible(RAW_CAPS[i], r):
                    elig_counts[i] += 1

        # Print the results
        print("\n" + "=" * 80)
        print(f"SPLIT: {split_name} | tasks={len(tasks)}")
        for i, name in enumerate(AGENT_NAMES):
            # Fraction of tasks in this split for which the agent is eligible
            frac = elig_counts[i] / max(1, len(tasks))
            print(f"{name:18s} eligible: {elig_counts[i]:6d} ({frac*100:5.1f}%)")

if __name__ == "__main__":
    main()