# Archaeos Tuff DL

Official code accompanying the manuscript **"Boundary-aware multitask
segmentation for sparse-label seismic interpretation: Full-line evaluation and
regional reconstruction of the Archaeos Tuff"**.

## Overview

This repository provides a boundary-aware multitask workflow for identifying
the Archaeos Tuff in heterogeneous 2-D marine seismic profiles. The selected
model is a residual U-Net with attention-gated skip connections, deep
supervision, and auxiliary edge and signed-distance heads. The paper model uses
three co-registered inputs: raw amplitude, envelope amplitude, and RMS
amplitude.

The code covers seismic-attribute preparation, patch extraction, model training,
sliding-window full-line inference, geological time-window filtering, and the
complete-profile evaluation metrics used in the manuscript.

For submission-release validation, see `REPRODUCIBILITY.md`. Data access,
restrictions, and the GitHub-to-Zenodo archival plan are described in
`DATA_AVAILABILITY.md`.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate archaeos-tuff-dl
```

## Computational requirements

The lightweight checks and synthetic example were verified on Windows 11 with
Python 3.13.5, NumPy 2.2.6, PyTorch 2.9.1 with CUDA 12.8, and an NVIDIA GeForce
RTX 5060 Laptop GPU with 8 GB VRAM. The reproducible environment files target
Python 3.10 or 3.11 and PyTorch 2.1 or newer.

- CPU execution is supported for data inspection, metrics, tests, and small
  model smoke checks.
- A CUDA-capable GPU is recommended for full-line inference and training.
- Eight GB of GPU memory is sufficient for batch-one patch inference; training
  with the paper batch size may require more memory or a smaller batch size.
- Disk requirements depend on the selected release assets. The minimum model,
  test-input, stored-prediction, and manifest set is approximately 2.3 GB.
- Runtime varies with profile size, overlap, device, and post-processing. Users
  should record their hardware and elapsed time with reproduced results.

## Quick check

```bash
python scripts/create_synthetic_example.py
python scripts/inspect_npz.py data/example/synthetic_line.npz
python scripts/run_synthetic_demo.py
python scripts/smoke_test_model.py
python -m pytest -q
```

The synthetic example verifies installation and array compatibility. It is not
a substitute for the geological test profiles used in the paper.

## Training

Set the dataset paths in `configs/final_training.yaml`, then run:

```bash
python code/run_train.py --config configs/final_training.yaml
```

The public configuration records the paper architecture and main training
parameters, including 140 epochs, batch size 64, AdamW learning rate `1e-4`,
weight decay `1e-4`, and random seed 42.

## Full-line inference

Input NPZ files must contain co-registered 2-D arrays named `raw`, `env`, and
`rms`, stored as `(time_samples, traces)`. The publication inference checkpoint
is `checkpoints/best_model.pt`; `checkpoints/last_model.pt` records the final
training epoch and is retained for resuming or auditing training. Run:

```bash
python scripts/run_paper_inference.py \
  --line-dir path/to/seismic_npz \
  --checkpoint checkpoints/best_model.pt \
  --output-dir outputs/paper_inference
```

The wrapper uses the paper inference settings: `256 x 256` windows, stride 128,
Hann-weighted merging, score threshold 0.5, fixed candidate filtering, invalid
data suppression, and geological time-window gating.

## Full-line evaluation

Reference NPZ files may store the segmentation as `mask`, `gt`, `label`, `y`,
or `target`. Recompute top/base MAE and RMSE, Coverage, and Connectivity with:

```bash
python scripts/evaluate_full_lines.py \
  --reference-dir path/to/reference_npz \
  --prediction-dir path/to/prediction_npz \
  --output outputs/full_line_metrics.csv
```

## Reproducing the reported test results

The public `v1.0-submission` GitHub Release must provide the trained checkpoint,
seven processed test inputs, their stored predictions, and the release
manifests. Follow `REPRODUCIBILITY.md` to verify hashes, rerun full-line
inference, and recalculate the machine-readable metrics in
`results/geological_usability_metrics_test7.csv`.

## Repository structure

```text
code/             Model, losses, data loader, training, and inference
preprocessing/    Attribute calculation, patch extraction, and blind splitting
configs/          Paper training and inference settings
scripts/          Example, inference, inspection, and evaluation entry points
tests/            Lightweight model and metric tests
docs/             User guide for inputs, outputs, and typical workflows
data/             Data format and availability notes
results/          Machine-readable metrics reported in the manuscript
checkpoints/      Instructions for pretrained model weights
```

## Data and pretrained weights

Large processed profiles, stored predictions, and `best_model.pt` are excluded
from ordinary Git history. During submission and peer review they must be
publicly downloadable as assets of the `v1.0-submission` GitHub Release. After
acceptance, the immutable accepted release will additionally be archived in
Zenodo and assigned a version-specific DOI. The GitHub repository will remain
public.

See `DATA_AVAILABILITY.md` for the access boundary, `data/README.md` for the NPZ
schema, and `zenodo/FILES.csv` for the canonical release inventory and hashes.

## License and citation

The source code is released under the MIT License. Dataset licenses are stated
separately by their data providers. Citation metadata are provided in
`CITATION.cff`.

## Contact

Zelong Ju  
Institute of Geophysics, Universität Hamburg  
zelong.ju@uni-hamburg.de
