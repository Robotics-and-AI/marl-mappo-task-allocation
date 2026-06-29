"""
make_dataset.py

Purpose:
- Generates a dataset of tasks using `task_generator.build_dataset(...)`
- Saves the resulting train/eval splits to a compressed NumPy `.npz` file

What it does, step-by-step:
1) Builds an absolute output path: ./datasets/tasks_v1.npz
2) Creates the output directory if it doesn't already exist
3) Generates training and evaluation tasks deterministically (seed=42)
4) Saves both splits to disk in NPZ format
5) Prints where the dataset was saved and how many tasks are in each split

Usage:
    python make_dataset.py
"""

import os
from task_generator import build_dataset, save_dataset_npz

def main():
    # Resolve the output file path and ensure its directory exists
    out_path = os.path.abspath("./datasets/tasks_v1.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Create the dataset splits (train/eval). Seed fixes randomness for reproducibility
    train_tasks, eval_tasks = build_dataset(seed=42)

    # Persist the dataset to a single .npz file.
    save_dataset_npz(out_path, train_tasks, eval_tasks)

    # Report results.
    print(f"✅ Saved dataset to: {out_path}")
    print(f"Train tasks: {len(train_tasks)} | Eval tasks: {len(eval_tasks)}")

if __name__ == "__main__":
    main()