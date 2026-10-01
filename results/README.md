# Reported results

`model_selection_metrics.csv` contains the architecture-comparison and module-
ablation values reported in Supplementary Table S4, Section I. All benchmark
architectures use raw amplitude (`Amp`) as the common input.

`input_representation_selection_metrics.csv` contains the input-representation
comparison reported in Supplementary Table S4, Section II. The selected
attention-based residual U-Net architecture is held fixed across these rows.
The `selected` column identifies the configuration carried forward in the
manuscript.

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
