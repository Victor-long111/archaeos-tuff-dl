# Application-line predictions

This directory records prediction products for six application profiles used
to evaluate regional applicability of the Archaeos Tuff segmentation workflow.

## Legacy predictions

The NPZ files under `legacy_predictions/` were copied from the original local
results folder on 2026-09-30. They were generated in March-April 2026, before
the final 140-epoch checkpoints were completed in May 2026. All six files use
a score threshold of 0.35 and none records the checkpoint filename or SHA-256.
They are therefore retained for provenance and visual comparison only; they
must not be described as outputs of `checkpoints/best_model.pt`.

Four profiles use the main post-processing configuration. HH06-03 and HH06-45
instead disable time-window gating and component filtering. HH06-08 and HH06-40
have empty post-processed masks. These differences are recorded in
`application_lines.csv`.

Large NPZ files are intentionally ignored by Git. A public paper release should
store final application predictions in the DOI-bearing data archive and retain
only their manifest, checksums, download instructions, and reproduction scripts
in Git.

## Required final regeneration

Before publication, rerun all six profiles with:

- `checkpoints/best_model.pt`;
- the single frozen publication inference configuration;
- a checkpoint SHA-256 embedded in every output file;
- identical threshold and post-processing settings unless a documented
  profile-specific geological rule is scientifically justified.

The regenerated archive should also include the processed input profiles, or a
clear data-access statement if redistribution is restricted.
