# Paper data split

The paper uses a line-disjoint evaluation design. All patches originating from
the seven profiles in `heldout_test_lines.txt` were excluded before the
remaining positive and negative patch pools were divided into training and
validation subsets with random seed 42 and an 80/20 split.

`split_summary.json` is the sanitized machine-readable record archived from the
paper run. It contains aggregate counts and no workstation or cluster paths.
The complete training and validation patch corpus is not redistributed, so the
individual `train.txt` and `val.txt` lists are generated locally rather than
stored as links to unavailable files.

Given access to the complete positive and negative patch directories, recreate
the archived policy and verify its aggregate counts with:

```bash
python preprocessing/make_paper_splits.py \
  --pos-root path/to/patches/positive \
  --neg-root path/to/patches/negative \
  --outdir outputs/paper_splits \
  --verify-summary data/splits/split_summary.json
```

The generated `heldout_source_patches.txt` is an exclusion audit. It is not the
paper test dataset. Independent evaluation was conducted on the seven complete
processed seismic profiles documented under `data/test7/`.

The original file-level training/validation membership list was not retained
in the release. Consequently, this package supports deterministic regeneration
of the documented policy and verification of all reported class counts, but it
does not claim bit-for-bit recovery of the historical membership list.
