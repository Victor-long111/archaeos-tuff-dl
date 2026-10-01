# Reported results

`geological_usability_metrics_test7.csv` contains the complete-profile metrics
reported for the seven independent test lines in the manuscript: top and base
MAE/RMSE, Coverage, and Connectivity.

Use `scripts/evaluate_full_lines.py` to calculate the same metric definitions
for compatible reference and prediction NPZ files.

`application/` contains the inventory and checksums for six application-line
prediction products used in the manuscript. These are the frozen final
publication outputs; their per-profile inference and post-processing settings
are recorded in `application/application_lines.csv`. See
`application/README.md` for interpretation and distribution details.
