# Pretrained weights

Two checkpoints from the final 140-epoch training run are retained:

- `best_model.pt`: the primary publication and inference checkpoint, saved at
  epoch 136 with the best recorded validation metric (`0.9408226953224162`);
- `last_model.pt`: the final training-state checkpoint saved at epoch 140,
  retained for training audit and possible resumption.

Use `best_model.pt` for all paper inference and reproduction commands:

```bash
python scripts/run_paper_inference.py \
  --line-dir path/to/seismic_npz \
  --checkpoint checkpoints/best_model.pt \
  --output-dir outputs/paper_inference
```

SHA-256 values are recorded in `best_model.sha256` and `last_model.sha256`.
Both files use the paper architecture and training configuration: inputs
`raw`, `env`, and `rms`; base width 32; group normalization; attention gates;
deep supervision; auxiliary edge and signed-distance heads; dropout 0.3; and
140 configured epochs.

The previous 60-epoch checkpoint is retained locally under `archive/` for
provenance only and must not be used to reproduce the paper results.

Because each final checkpoint is close to GitHub's ordinary-file size limit,
the submission version distributes `best_model.pt` as an asset of the public
`v1.0-submission` GitHub Release rather than through ordinary Git history. The
accepted release will additionally be archived in Zenodo and assigned a DOI.
