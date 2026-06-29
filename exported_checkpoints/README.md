# Exported checkpoints

Training exports the best checkpoint to this folder by default:

```text
exported_checkpoints/best_checkpoint
```

Checkpoints can be large, so the default `.gitignore` excludes checkpoint contents. To share a trained model publicly, consider one of these options:

1. Upload the checkpoint as a GitHub Release asset.
2. Use Git LFS for checkpoint files.
3. Document how users can train their own checkpoint with `python train.py`.

The GUI and evaluation scripts expect an RLlib checkpoint directory or a folder containing `checkpoint_*` subfolders.
