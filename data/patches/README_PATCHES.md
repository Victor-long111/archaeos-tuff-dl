# Representative training patches

## Scope

This release includes 231 processed, model-ready NPZ patches selected from the
available patch collection:

- 144 positive patches containing labelled Archaeos Tuff pixels;
- 87 negative patches with empty reference masks; and
- a total payload size of 413,440,079 bytes (approximately 394.3 MiB).

These files are representative training examples. They are not the complete
training/validation dataset and are not presented as the exact corpus used for
every training run reported in the manuscript. No train/validation split is
assigned to this subset.

## Leakage control

The released patches were filtered by source-profile identifier. Patches from
all seven independent test profiles and all six manuscript application
profiles were excluded. The excluded identifiers are:

`HH06-03`, `HH06-08`, `HH06-10`, `HH06-14`, `HH06-15`, `HH06-17`,
`HH06-22`, `HH06-40`, `HH06-41`, `HH06-45`, `MSM135-P1004`,
`MSM135-P1018`, and `MSM135-P1312`.

The deterministic selection and validation procedure is implemented in
`scripts/build_patch_release_manifest.py`. File-level provenance, class,
source profile, dimensions, label statistics, byte size, and SHA-256 digest
are recorded in `patch_manifest.csv`.

## NPZ format

Every released patch is 256 x 256 samples and contains co-registered arrays:

- `raw`: seismic amplitude;
- `env`: envelope amplitude;
- `rms`: RMS amplitude; and
- `mask`: binary reference mask.

All four arrays share the same two-dimensional shape and were checked for
non-finite values. Positive/negative class labels were verified against the
number of non-zero mask pixels.

## Distribution

The NPZ files are distributed as individual assets of the GitHub submission
Release rather than committed to ordinary Git history. Verify downloaded
files with `CHECKSUMS_PATCHES.sha256` or the digests in `patch_manifest.csv`.

Original SEG-Y acquisition files are not included. See
`DATA_AVAILABILITY.md` and `zenodo/LICENSE_DATA.txt` for the access boundary
and applicable terms.
