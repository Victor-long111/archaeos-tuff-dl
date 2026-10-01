# Data format

Each seismic profile is stored as a compressed NumPy NPZ file. Model inputs are
co-registered 2-D arrays with shape `(time_samples, traces)`.

| Key | Description | Required |
|---|---|---:|
| `raw` | Seismic amplitude | Yes |
| `env` | Envelope amplitude | Yes |
| `rms` | RMS amplitude | Yes |
| `mask` | Reference AT segmentation | Evaluation only |

Equivalent reference keys `gt`, `label`, `y`, and `target` are also accepted.
Missing and muted samples must use the same representation across all input
channels.

The full profiles used in the manuscript are distributed separately because of
their size and cruise-data permissions. A synthetic installation example can be
created with:

```bash
python scripts/create_synthetic_example.py
```

The seven processed independent test profiles are documented under `test7/`.
They contain model-ready arrays and reference masks, not the original SEG-Y
files. Large NPZ files are intended for a DOI-bearing data archive rather than
ordinary Git history.

The submission Release also supplies 231 representative 256 x 256 training
patches from source profiles that do not overlap the independent test or
application profiles. See `patches/README_PATCHES.md` and
`patches/patch_manifest.csv`. This subset documents the training-array format;
it is not the complete training/validation corpus.
