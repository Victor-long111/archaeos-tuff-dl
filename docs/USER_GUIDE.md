# User guide

## Purpose

Archaeos Tuff DL segments candidate Archaeos Tuff intervals in 2-D marine
seismic profiles. The software supports three workflows: model training,
full-line inference, and complete-profile evaluation.

## Inputs

Full-line inputs are compressed NumPy NPZ files. Arrays must have shape
`(time_samples, traces)` and share the same sampling grid.

- `raw`: seismic amplitude;
- `env`: envelope amplitude;
- `rms`: RMS amplitude;
- `mask`: optional binary reference segmentation for evaluation.

Reference arrays may alternatively be named `gt`, `label`, `y`, or `target`.

## Outputs

Full-line inference creates one output directory per input profile. The main NPZ
contains:

- `seg_prob`: merged AT likelihood score;
- `pred_bin`: mask after score thresholding;
- `pred_bin_pp`: mask after candidate filtering;
- inference and post-processing parameters;
- optional edge, signed-distance, validity, and time-window arrays.

Three PNG files provide visual quality control of the likelihood scores and
candidate masks. Likelihood scores guide interpretation and are not calibrated
probabilities of geological correctness.

## Typical workflow

1. Prepare line-level NPZ files with `preprocessing/Cal_attribu_NPZ.py`.
2. Create positive and negative patches with the two crop scripts.
3. Create profile-aware development splits with `make_splits_blind.py`.
4. Set local paths in `configs/final_training.yaml` and run training.
5. Run fixed-configuration full-line inference with
   `scripts/run_paper_inference.py`.
6. Review model responses together with seismic facies, stratigraphic position,
   structural context, and cross-line continuity.
7. Calculate full-line metrics with `scripts/evaluate_full_lines.py` when a
   reference mask is available.

## Important options

The paper configuration uses `256 x 256` windows, stride 128, a full-line
threshold of 0.35, group normalization, attention gates, deep supervision, and
auxiliary heads. The 0.5 value in `configs/final_training.yaml` is used for
patch-level validation. The inference implementation also exposes
candidate-component size, trace-span, morphological closing, invalid-data
suppression, and geological time-window options. Profile-specific final
settings for HH06-17 and HH06-03/HH06-45 are recorded in
`configs/final_profile_overrides.yaml` and selected through the wrapper's
`--profile` option. Run either entry point with `--help` to view all options.

## Expected behavior

The synthetic demo should create an NPZ prediction and exit without error. Its
model is randomly initialized and therefore demonstrates software execution
only. Scientifically meaningful predictions require trained weights and seismic
data prepared using the same attribute and normalization conventions as the
paper.

## Limitations

Regional geological screening and final horizon digitization are expert
interpretation steps and are not fully automated. Full cruise profiles may be
distributed separately where data permissions allow.
