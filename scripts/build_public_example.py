"""Crop a legally releasable line NPZ into a compact public example."""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a compact excerpt only from data approved for redistribution."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--col-start", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    row_slice = slice(args.row_start, args.row_start + args.height)
    col_slice = slice(args.col_start, args.col_start + args.width)
    keep = ("raw", "env", "rms", "mask", "gt", "label", "y", "target")

    with np.load(args.input, allow_pickle=False) as source:
        arrays = {}
        for key in keep:
            if key in source:
                value = np.asarray(source[key])
                if value.ndim != 2:
                    raise ValueError(f"Expected 2-D array for {key}, got {value.shape}")
                arrays[key] = value[row_slice, col_slice]

    required = {"raw", "env", "rms"}
    missing = required.difference(arrays)
    if missing:
        raise KeyError(f"Missing required arrays: {sorted(missing)}")
    if not arrays["raw"].size:
        raise ValueError("Requested crop is outside the source array.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved public excerpt to {args.output}")
    print("Confirm redistribution permission and document source-line provenance before release.")


if __name__ == "__main__":
    main()

