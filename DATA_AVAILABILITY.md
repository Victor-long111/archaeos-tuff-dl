# Data availability

## Public material at submission

The public GitHub repository contains the source code, fixed training and
inference configurations, a synthetic seismic example, tests, documentation,
machine-readable reported metrics, and manifests for the independent test and
application profiles.

The following large binary files are intentionally excluded from ordinary Git
history and must be attached to the public `v1.0-submission` GitHub Release
before the manuscript is submitted:

- `best_model.pt`;
- seven processed independent test-profile NPZ files;
- seven stored test-prediction NPZ files;
- six stored application-prediction NPZ files; and
- 231 representative processed training patches (144 positive and 87
  negative) drawn only from source profiles outside the independent test and
  application sets.

Their checksums, fixed configurations, machine-readable metrics, profile
inventories, and release manifests are tracked directly in Git and are also
included alongside the binary assets for convenient verification.

`zenodo/FILES.csv` gives the expected filenames, sizes, categories, and SHA-256
values. `zenodo/CHECKSUMS.sha256` is the integrity manifest for the submission
assets. The release must remain publicly downloadable throughout peer review.

The representative patches are supplied to document the model input and label
format and to provide inspectable training examples. They are not the complete
training/validation corpus and do not carry an assigned train/validation split.
Their deterministic source-line exclusion rule and file-level metadata are
documented in `data/patches/README_PATCHES.md` and
`data/patches/patch_manifest.csv`.

## Original seismic data

Original SEG-Y survey files are not distributed through this repository. They
are excluded because of their size and the redistribution conditions attached
to the source cruise data. The seven released test profiles are processed,
model-ready NPZ derivatives containing co-registered `raw`, `env`, `rms`, and
reference `mask` arrays rather than original acquisition files.

The original application-profile inputs are not part of the minimum public
release. Their derived prediction products and inventory are supplied to
document the manuscript application workflow.

## Synthetic example

`data/example/synthetic_line.npz` is stored directly in Git and is available
without external downloads. It permits installation checks, array-format
validation, and an end-to-end software demonstration. It is not a substitute
for the geological test profiles when reproducing the reported metrics.

## Archival plan

During submission and peer review, the public GitHub repository and its
`v1.0-submission` Release are the access points for code and supporting files.
After acceptance, the immutable accepted GitHub Release will be archived in
Zenodo and assigned a version-specific DOI. The GitHub repository will remain
public; the Zenodo DOI will supplement rather than replace it.

## Rights and contact

Software is released under the repository MIT License. Rights and restrictions
for model weights, derived products, and processed seismic arrays are described
in `zenodo/LICENSE_DATA.txt`. Redistribution of processed seismic inputs is
conditional on the applicable source-data terms.

Questions about access and file identity should be directed to:

Zelong Ju  
Institute of Geophysics, Universität Hamburg  
zelong.ju@uni-hamburg.de
