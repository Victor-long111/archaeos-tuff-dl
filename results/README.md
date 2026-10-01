# Reported results

`geological_usability_metrics_test7.csv` contains the complete-profile metrics
reported for the seven independent test lines in the manuscript: top and base
MAE/RMSE, Coverage, and Connectivity.

Use `scripts/evaluate_full_lines.py` to calculate the same metric definitions
for compatible reference and prediction NPZ files.

`application/` contains the inventory and checksums for six application-line
prediction products. The currently archived NPZ files predate the final
140-epoch model and are explicitly classified as legacy provenance artifacts;
see `application/README.md` before using them.
