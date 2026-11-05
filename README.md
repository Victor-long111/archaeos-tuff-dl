# Archaeos Tuff DL

Attention residual U-Net for ML-assisted seismic interpretation, with deep
supervision and auxiliary edge/signed-distance heads.

## Setup

Python 3.11 recommended; a GPU is recommended for substantial training.

```bash
python -m pip install -r requirements.txt
python code/run_train.py --config configs/final_training.yaml --dry-run
```

## Workflow

Inputs are processed NPZ files containing co-registered `raw`, `env` and `rms`
arrays of shape `(time_samples, traces)`. Training/evaluation additionally
requires aligned reference masks; class 1 is foreground and 255 is ignored.
Real data, test profiles, labels, patches, trained weights and stored predictions
are **not included**. Supply authorized external materials and use new output paths.

### Attribute calculation and patch extraction

```bash
python preprocessing/Cal_attribu_NPZ.py --input-dir /path/to/line_npz --output-dir /path/to/attributed_npz
python preprocessing/Crop_Pos_patches.py --input-dir /path/to/attributed_npz --output-dir /path/to/new_patches
python preprocessing/Crop_Neg_patches.py --line-dir /path/to/attributed_npz --positive-dir /path/to/new_patches/positive --output-dir /path/to/new_patches
```

These are restored historical source scripts. The attribute script calculates
Env/Rms/IFreq/Coh and phase channels; it does not implement Swe/Var/AveEn.
Its Coh smoothing defaults also need reconciliation with Supplementary Table S1.
Prepare the authorized development-line inputs before cropping: these scripts
do not enforce test-line exclusion. Use trusted NPZ inputs and fresh output
directories; object metadata is supported. Existing crop outputs are not deleted,
and attribute outputs are not overwritten by default. No generated files belong
in this source-only upload. Only output-safety behavior was changed in these copies.

### Training and evaluation

1. Set external input/split/output paths in a working copy of `configs/final_training.yaml`.
2. Inspect the training command with `--dry-run`, then remove that flag to train.
3. Supply a compatible checkpoint for inference, then evaluate the actual arrays.

```bash
python code/run_train.py --config /path/to/training.yaml --dry-run
python scripts/run_paper_inference.py --line-dir /path/to/profiles --checkpoint /path/to/model.pt --output-dir /path/to/new_predictions --profile common
python scripts/evaluate_full_lines.py --reference-dir /path/to/references --prediction-dir /path/to/new_predictions --prediction-key pred_bin_pp --sample-interval-ms 1.0 --output /path/to/new_metrics.csv
```

Replace `1.0` with the documented sample interval. Entry points provide `--help`.
`preprocessing/` retains split-policy utilities; their output does not establish
the historical training membership. `code/sweep_thr_val.py` retains threshold
selection on validation data, not test-outcome optimization.

## Configuration and result status

Model, sampler, inference and evaluator implementations are unchanged.
`configs/final_training.yaml` is the single executable training configuration.

## Files and license

- `code/`: model, losses, data loading, augmentation, sampling, training and inference.
- `configs/`: the model configs
- `preprocessing/`: attribute calculation, positive/negative patch extraction and split utilities.
- `scripts/`: full-line inference and evaluation entry points.

MIT software license: see `LICENSE`. It does not grant access to external data.
Do not add real research artifacts or old Git history to this source-only folder.

## Review status

This is a source-only coauthor-review snapshot, not a verified paper-reproduction
release. A small synthetic attribute-to-evaluation workflow passed; this does
not validate manuscript performance. Historical train/validation/test membership
and the correspondence between checkpoints, inference settings and reported
metrics remain unresolved. Some settings and model parameter counts differ from
the manuscript. Full-line ignore masking and sliding-window end coverage have
known implementation issues. Data access restrictions do not resolve these
method and result discrepancies.
