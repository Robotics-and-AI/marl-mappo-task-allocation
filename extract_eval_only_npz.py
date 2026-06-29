"""
extract_eval_only_npz.py

Goal
----
Given a dataset .npz produced by your task_generator.py (or save_dataset_npz),
create a new .npz that contains ONLY the evaluation split:

    - eval_instructions
    - eval_categories
    - eval_requirements

Why
---
This is useful when you want:
  - a clean eval-only dataset to ship/run in evaluation scripts
  - to avoid any train data being loaded by mistake
  - repeatable benchmarking using a fixed set of eval tasks

Input
-----
An .npz file that contains at least:
  - eval_instructions (dtype=object, shape=(N,))
  - eval_categories   (dtype=object, shape=(N,))
  - eval_requirements (dtype=float32, shape=(N,3))  in RAW [1..4]

Output
------
A new compressed .npz containing ONLY those 3 eval keys, same shapes.

Usage
-----
python extract_eval_only_npz.py --in datasets/tasks_v1.npz --out datasets/tasks_v1_eval_only.npz
"""

import argparse
import numpy as np


def extract_eval_only(in_path: str, out_path: str) -> None:
    # Load NPZ. allow_pickle=True is needed because instructions/categories are stored as dtype=object arrays.
    data = np.load(in_path, allow_pickle=True)

    # Check that required keys exist.
    required = ["eval_instructions", "eval_categories", "eval_requirements"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"Missing keys in {in_path}: {missing}. Found keys: {data.files}")

    # Extract eval arrays.
    eval_instr = data["eval_instructions"]
    eval_cat = data["eval_categories"]
    eval_req = data["eval_requirements"].astype(np.float32, copy=False)

    # Basic alignment check: all eval arrays must have same length.
    if not (len(eval_instr) == len(eval_cat) == len(eval_req)):
        raise ValueError(
            f"Eval arrays misaligned: len(instr)={len(eval_instr)}, "
            f"len(cat)={len(eval_cat)}, len(req)={len(eval_req)}"
        )

    # Save ONLY eval keys.
    np.savez_compressed(
        out_path,
        eval_instructions=eval_instr,
        eval_categories=eval_cat,
        eval_requirements=eval_req,
    )

    print(f"Saved eval-only dataset: {out_path}")
    print(f"Eval samples: {len(eval_instr)}")
    print("Keys written: eval_instructions, eval_categories, eval_requirements")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input dataset .npz (train+eval)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output eval-only .npz")
    args = ap.parse_args()

    extract_eval_only(args.in_path, args.out_path)


if __name__ == "__main__":
    main()