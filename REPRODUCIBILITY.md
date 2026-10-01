# Reproducibility guide

This guide defines the minimum workflows used to evaluate the public submission
release. Commands are run from the repository root.

## 1. Install the environment

Using `venv`:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate archaeos-tuff-dl
```

## 2. Validate the installation

```bash
python scripts/inspect_npz.py data/example/synthetic_line.npz
python scripts/run_synthetic_demo.py
python scripts/smoke_test_model.py
python -m pytest -q
python scripts/check_release.py --content-only
```

Expected behaviour:

- the synthetic input contains `raw`, `env`, `rms`, and `mask` arrays with a
  common shape;
- the synthetic demo writes `outputs/synthetic_demo_prediction.npz`;
- the model smoke test prints `MODEL_FORWARD_OK`;
- the automated tests pass; and
- the content audit reports `CONTENT READY`.

The synthetic demo uses randomly initialized weights and verifies software
execution only. It does not reproduce geological performance.

## 3. Recreate the paper split when the complete patch corpus is available

```bash
python preprocessing/make_paper_splits.py \
  --pos-root path/to/patches/positive \
  --neg-root path/to/patches/negative \
  --outdir outputs/paper_splits \
  --verify-summary data/splits/split_summary.json
```

This operation removes every patch from the seven independent test profiles
before stratified training-validation splitting and verifies all archived
aggregate class counts. The original file-level membership list is not part of
the release, so this reproduces the documented policy rather than claiming
bit-for-bit recovery of that historical list. The public representative patch
subset is too small to recreate the full split and is provided only for format
inspection.

## 4. Obtain submission-release assets

Download the assets attached to the public `v1.0-submission` GitHub Release.
Keep test inputs and stored predictions in separate directories, for example:

```text
release_assets/
├── best_model.pt
├── final_inference.yaml
├── final_profile_overrides.yaml
├── test_inputs/
│   └── seven processed test NPZ files
├── stored_test_predictions/
│   └── seven prediction NPZ files
└── stored_application_predictions/
    └── six prediction NPZ files
```

The canonical file list, sizes, and hashes are in `zenodo/FILES.csv`. Verify
downloads against `zenodo/CHECKSUMS.sha256` before use.

On Linux or macOS, if the files and checksum manifest are in one directory:

```bash
sha256sum --check CHECKSUMS.sha256
```

On Windows PowerShell, calculate an individual checksum with:

```powershell
Get-FileHash -Algorithm SHA256 best_model.pt
```

## 5. Run full-line inference

The common publication settings use full-line threshold 0.35 and are recorded
in `configs/final_inference.yaml`. Run the six common test profiles with:

```bash
python scripts/run_paper_inference.py \
  --line-dir release_assets/test_inputs \
  --checkpoint release_assets/best_model.pt \
  --output-dir outputs/paper_inference \
  --profile common \
  --lines HH06-10 HH06-14 HH06-15 HH06-22 MSM135-P1004 MSM135-P1018
```

Run the frozen HH06-17 profile-specific configuration with:

```bash
python scripts/run_paper_inference.py \
  --line-dir release_assets/test_inputs \
  --checkpoint release_assets/best_model.pt \
  --output-dir outputs/paper_inference \
  --profile hh06-17 \
  --lines HH06-17
```

The corresponding HH06-03/HH06-45 application mode is selected with
`--profile hh06-03-45`. The complete mapping is archived in
`configs/final_profile_overrides.yaml`. When `--lines` is omitted, every NPZ
below `--line-dir` is processed recursively, so a mixed directory should use
the explicit profile groups above.

Expected outputs include segmentation scores, thresholded masks,
post-processed masks, stored parameters, and visual quality-control PNG files.

## 6. Recalculate full-line metrics

```bash
python scripts/evaluate_full_lines.py \
  --reference-dir release_assets/test_inputs \
  --prediction-dir outputs/paper_inference \
  --output outputs/reproduced_test7_metrics.csv
```

Compare the generated table with
`results/geological_usability_metrics_test7.csv`. Metric definitions are
implemented in `code/full_line_metrics.py` and described in
`docs/USER_GUIDE.md`.

## 7. Audit the public GitHub release

After committing all files, configuring the public `origin`, and creating the
submission tag, run:

```bash
python scripts/check_release.py
```

The audit checks required files, the checkpoint checksum, the synthetic data
schema, Git commit state, the `origin` remote, a clean working tree, and the
presence of an exact version tag. Public visibility and release-asset downloads
must still be confirmed in a browser.

## Reproducibility boundary

The release supports software validation, full-line inference, metric
recalculation, and inspection of stored application predictions. Original
SEG-Y files and unreleased application inputs are outside the public release.
Regional geological screening and final horizon digitization include expert
interpretation and are not represented as fully automated steps.
