# Seven independent test profiles

The seven NPZ files listed in `test_lines.csv` are processed, model-ready test
profiles. They are not original SEG-Y files. Each archive contains co-registered
`raw`, `env`, `rms`, and binary `mask` arrays. All required arrays are finite,
have matching `(time_samples, traces)` shapes, and use a 1 ms sample interval.

The archives additionally contain `ifreq`, `swe`, `coh`, `var`, and `aveen`
attributes plus a `meta_attr` record describing attribute generation. The paper
model uses only `raw`, `env`, and `rms`; `mask` is used for evaluation.

The local source copies on the Desktop and the E: research directory were
verified as byte-identical on 2026-09-30. Their hashes are recorded in
`CHECKSUMS.sha256`.

## Distribution

The seven full archives total approximately 1.85 GiB and are intentionally not
stored in ordinary Git history. During submission and peer review they must be
attached to the public `v1.0-submission` GitHub Release. After acceptance, the
accepted release will additionally be archived in a DOI-bearing repository.
If redistribution of derived cruise data is not permitted,
`DATA_AVAILABILITY.md` must state the restriction, data owner, and access route.

A reduced public package containing only `raw`, `env`, `rms`, and `mask` would
be approximately 692 MiB, 63.5% smaller than the full attribute archives. Any
repacked files must receive new SHA-256 values and a new manifest version.

## Reproduction status

The input data, reference masks, stored predictions, and Table 4 metrics form
the final publication test set. The files listed in the release manifests are
the frozen products used for the manuscript results. Their filenames, sizes,
and SHA-256 values provide the version boundary for independent verification.

Users can either inspect the stored predictions and recalculate the reported
metrics, or rerun inference with the publication checkpoint and configuration
distributed in the `v1.0-submission` release. Reproduced values should be
compared with `results/geological_usability_metrics_test7.csv` using the metric
definitions in `code/full_line_metrics.py`.
