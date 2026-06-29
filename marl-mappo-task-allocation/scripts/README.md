# Scripts

Supporting scripts for dataset inspection, eligibility diagnostics, and checkpoint evaluation.

Run them from the repository root:

```bash
python scripts/dataset_npz_to_txt.py --npz datasets/tasks_v1.npz --out datasets/tasks_v1_grouped.txt
python scripts/eligibility_report_basic.py
python scripts/eligibility_report_env_sampling.py
python scripts/evaluate_checkpoint_on_evalset.py \
  --checkpoint exported_checkpoints/best_checkpoint \
  --eval_npz datasets/tasks_v1_eval_only.npz \
  --out_csv results/eval_rows.csv \
  --out_json results/eval_summary.json \
  --episode_len 50
```
