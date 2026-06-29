# Datasets

This folder is used for generated task datasets and human-readable dataset reports.

Generated files are intentionally ignored by the default `.gitignore`:

- `tasks_v1.npz`
- `tasks_v1_eval_only.npz`
- `tasks_v1_grouped.txt`
- `eval_only.txt`

Regenerate them from the repository root with:

```bash
python make_dataset.py
python extract_eval_only_npz.py --in datasets/tasks_v1.npz --out datasets/tasks_v1_eval_only.npz
python scripts/dataset_npz_to_txt.py --npz datasets/tasks_v1.npz --out datasets/tasks_v1_grouped.txt
python scripts/dataset_npz_to_txt.py --npz datasets/tasks_v1_eval_only.npz --out datasets/eval_only.txt --split eval
```

The full dataset file contains both train and eval splits. The eval-only file contains only the eval arrays and should be used only by scripts that explicitly support eval-only datasets.
