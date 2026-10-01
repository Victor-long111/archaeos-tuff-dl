# Application-line predictions

This directory records prediction products for six application profiles used
to evaluate regional applicability of the Archaeos Tuff segmentation workflow.

## Final publication predictions

The six NPZ files listed in `application_lines.csv` are the frozen application
outputs used in the manuscript. Their identities are fixed by filename, file
size, modification time, and SHA-256 checksum. The manifest records the input
channels, normalization, threshold, invalid-data suppression, geological
time-window gating, component filtering, and post-processing parameters for
each profile.

All six files use a score threshold of 0.35. Four profiles use the common
post-processing configuration, while HH06-03 and HH06-45 use the documented
profile-specific setting without time-window gating or component filtering.
HH06-08 and HH06-40 contain no pixels after final post-processing. These are
the final recorded outcomes for those profiles rather than missing products.

Large NPZ files are intentionally excluded from ordinary Git history. The
`v1.0-submission` release distributes the final application predictions as
individual assets together with this manifest and the checksum file. Original
application-profile inputs are outside the minimum public release; their
availability boundary is explained in `DATA_AVAILABILITY.md`.

The application outputs support inspection of the manuscript interpretation.
They are not included in the seven-line quantitative test set and should not be
combined with `geological_usability_metrics_test7.csv`.
