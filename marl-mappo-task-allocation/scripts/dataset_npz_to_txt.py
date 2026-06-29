"""
dataset_npz_to_txt.py

Purpose
-------
Convert a task dataset stored in .npz into a human-readable .txt report.

This script supports:
  1) Full dataset NPZ files containing BOTH splits:
       - train_instructions / train_categories / train_requirements
       - eval_instructions  / eval_categories  / eval_requirements

  2) Eval-only dataset NPZ files containing ONLY:
       - eval_instructions / eval_categories / eval_requirements

  3) Train-only dataset NPZ files containing ONLY:
       - train_instructions / train_categories / train_requirements

It groups tasks by their scenario/category and writes a text report in a natural numeric
ordering (scenario2 before scenario10).

Output format (per category)
----------------------------
================================================================================
<category>
================================================================================

[<category>] TRAIN
--------------------------------------------------------------------------------
train    <j>    <category>    <requirements-as-list>
<instruction-text>

[<category>] EVAL
--------------------------------------------------------------------------------
eval     <j>    <category>    <requirements-as-list>
<instruction-text>

Usage
-----
# Auto-detect available splits and export them:
python scripts/dataset_npz_to_txt.py --npz datasets/tasks_v1.npz --out datasets/tasks_v1_grouped.txt

# Export only eval split (works on eval-only NPZ too):
python scripts/dataset_npz_to_txt.py --npz datasets/tasks_v1_eval_only.npz --out datasets/eval_only.txt --split eval
"""

import argparse
import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


def scenario_sort_key(cat: str):
    """
    Sort categories in a "natural" way when they end with a number.
    Example: scenario2 comes before scenario10.
    """
    m = re.search(r"(\d+)$", str(cat))
    return int(m.group(1)) if m else str(cat)


def detect_splits(d: np.lib.npyio.NpzFile) -> Dict[str, bool]:
    """
    Return which splits exist in the loaded NPZ based on key presence.
    """
    has_train = all(k in d.files for k in ("train_instructions", "train_categories", "train_requirements"))
    has_eval = all(k in d.files for k in ("eval_instructions", "eval_categories", "eval_requirements"))
    return {"train": has_train, "eval": has_eval}


def group_indices(d: np.lib.npyio.NpzFile, split: str) -> Dict[str, List[int]]:
    """
    Build a mapping: category -> list of indices for a given split ("train" or "eval").
    """
    cats = d[f"{split}_categories"]
    groups = defaultdict(list)
    for i, c in enumerate(cats):
        groups[str(c)].append(i)
    return groups


def write_split_block(
    f,
    d: np.lib.npyio.NpzFile,
    split: str,
    cat: str,
    indices: List[int],
):
    """
    Write one category block for a given split.
    """
    f.write(f"[{cat}] {split.upper()}\n")
    f.write("-" * 120 + "\n")

    instr_arr = d[f"{split}_instructions"]
    req_arr = d[f"{split}_requirements"]

    for j, i in enumerate(indices):
        instr = instr_arr[i]
        req = np.asarray(req_arr[i], dtype=np.float32)

        f.write(f"{split}\t{j}\t{cat}\t{req.tolist()}\n")
        f.write(str(instr) + "\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to dataset .npz (full, eval-only, or train-only)")
    ap.add_argument("--out", required=True, help="Path to output .txt report")
    ap.add_argument(
        "--split",
        default="auto",
        choices=["auto", "train", "eval", "both"],
        help="Which split(s) to export. 'auto' exports whatever exists in the NPZ.",
    )
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    available = detect_splits(d)

    if args.split == "train" and not available["train"]:
        raise KeyError(f"Requested split=train but train_* keys not found in {args.npz}. Keys: {d.files}")
    if args.split == "eval" and not available["eval"]:
        raise KeyError(f"Requested split=eval but eval_* keys not found in {args.npz}. Keys: {d.files}")
    if args.split == "both" and (not available["train"] or not available["eval"]):
        raise KeyError(
            f"Requested split=both but missing one split in {args.npz}. "
            f"has_train={available['train']} has_eval={available['eval']} Keys: {d.files}"
        )

    # Determine which splits to write
    if args.split == "auto":
        splits_to_write = [s for s in ("train", "eval") if available[s]]
        if not splits_to_write:
            raise KeyError(f"No train_* or eval_* keys found in {args.npz}. Keys: {d.files}")
    elif args.split == "both":
        splits_to_write = ["train", "eval"]
    else:
        splits_to_write = [args.split]

    # Group indices per split
    groups = {}
    for sp in splits_to_write:
        groups[sp] = group_indices(d, sp)

    # Union of categories across selected splits
    all_cats = set()
    for sp in splits_to_write:
        all_cats |= set(groups[sp].keys())
    all_cats = sorted(all_cats, key=scenario_sort_key)

    # Write report
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"NPZ: {args.npz}\n")
        f.write(f"Splits exported: {', '.join(splits_to_write)}\n")
        f.write("=" * 120 + "\n\n")

        for cat in all_cats:
            # Print counts to console for convenience
            counts_msg = " | ".join(f"{sp}:{len(groups[sp].get(cat, []))}" for sp in splits_to_write)
            print(f"{cat} -> {counts_msg}")

            # Category header
            f.write("=" * 120 + "\n")
            f.write(f"{cat}\n")
            f.write("=" * 120 + "\n\n")

            # Write each split block in order (train then eval if both)
            for sp in ("train", "eval"):
                if sp in splits_to_write:
                    idxs = groups[sp].get(cat, [])
                    write_split_block(f, d, sp, cat, idxs)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()