# Archaeos Tuff DL: model, processed test profiles, and prediction products

## Release scope

This Zenodo record is the minimum data-and-model companion release for the
manuscript *Boundary-aware multitask segmentation for sparse-label seismic
interpretation: Full-line evaluation and regional reconstruction of the
Archaeos Tuff*.

Release version: `1.0.0`  
Release date: `2026-09-30`

The public source-code repository, the immutable software release, and this
data record should be linked to one another through the Zenodo **Related
identifiers** field and the manuscript's Code and Data Availability statement.

## Included material

The release contains:

- `best_model.pt`, the primary trained checkpoint supplied for inference;
- `best_model.sha256`, the checkpoint checksum supplied with the software;
- `final_inference.yaml`, the frozen inference configuration;
- seven processed independent test profiles with reference masks;
- seven stored prediction products for those test profiles;
- `geological_usability_metrics_test7.csv`, the per-profile metrics reported
  for the seven-profile test set;
- `test_lines.csv`, the test-profile inventory;
- six stored prediction products for manuscript application profiles; and
- `application_lines.csv`, the application-product inventory.

`FILES.csv` describes every payload file. `CHECKSUMS.sha256` provides SHA-256
checksums for integrity verification. The four release-metadata files are kept
under `zenodo/` in the source repository but should be uploaded alongside the
payload files in the Zenodo record.

The `Positive/` and `Negative/` patch collections are not required for this
minimum release and are intentionally excluded. They may be deposited in a
later, versioned training-example release.

## Test-input format

Each test-profile NPZ contains co-registered two-dimensional arrays with shape
`(time_samples, traces)`:

- `raw`: seismic amplitude;
- `env`: envelope amplitude;
- `rms`: RMS amplitude; and
- `mask`: binary reference mask used for evaluation.

The paper model uses `raw`, `env`, and `rms` as input channels. Some files also
retain additional seismic attributes and a `meta_attr` record; these are not
required by the three-channel paper model. The sample interval is 1 ms for the
seven released test profiles.

## Prediction format

Prediction NPZ files contain, where applicable:

- `seg_prob`: merged segmentation score;
- `pred_bin`: thresholded binary mask;
- `pred_bin_pp`: post-processed binary mask; and
- stored inference and post-processing parameters.

The test and application prediction files are released as the numerical
products used in the manuscript workflow. File identity and integrity are
defined by `FILES.csv` and `CHECKSUMS.sha256`.

## Metrics

`geological_usability_metrics_test7.csv` reports, for each independent test
profile:

- top MAE and RMSE in milliseconds;
- base MAE and RMSE in milliseconds;
- reference coverage in percent; and
- connectivity.

Metric definitions and the evaluation implementation are provided in the
public source-code repository. The CSV is the machine-readable reference table
for comparison with reproduced results.

## Minimum use workflow

1. Download all files in this Zenodo record without renaming them.
2. Verify file integrity against `CHECKSUMS.sha256` or the hashes in
   `FILES.csv`.
3. Install the public code release using its `requirements.txt` or
   `environment.yml`.
4. Place the seven test-input NPZ files in one input directory and use
   `best_model.pt` together with `final_inference.yaml` for inference.
5. Evaluate the generated masks with the repository's full-line evaluation
   script and compare the output with
   `geological_usability_metrics_test7.csv`.

The repository README and user guide are authoritative for command-line
options and expected outputs.

## Data availability and limitations

Original SEG-Y files are not included because of cruise-data size and
redistribution conditions. The released test inputs are processed,
model-ready NPZ products rather than original acquisition files. Reference
masks, predictions, metrics, and manifests are included to permit evaluation
of the published workflow.

The six application prediction products are included, but their original
seismic inputs are not part of this minimum record. Regional geological
screening and final horizon digitization remain expert-interpretation steps
and are not represented as fully automated outputs.

See `LICENSE_DATA.txt` for the separate terms governing software, author-created
derived products, and processed arrays derived from third-party seismic data.

## Citation

Please cite both the associated article and the version-specific DOI displayed
on the Zenodo record. The version-specific DOI should be used when exact file
identity is important.

## Contact

Zelong Ju  
Institute of Geophysics, Universität Hamburg  
zelong.ju@uni-hamburg.de
