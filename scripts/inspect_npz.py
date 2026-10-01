"""Print NPZ keys, shapes, dtypes, and value ranges without modifying data."""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    args = parser.parse_args()

    with np.load(args.npz, allow_pickle=True) as data:
        print(f"File: {args.npz}")
        for key in data.files:
            value = np.asarray(data[key])
            if value.dtype == object:
                limits = "object metadata (values not printed)"
            elif np.issubdtype(value.dtype, np.number) and value.size:
                finite = value[np.isfinite(value)]
                limits = (
                    f"min={finite.min():.6g}, max={finite.max():.6g}"
                    if finite.size
                    else "no finite values"
                )
            else:
                limits = "non-numeric"
            print(f"{key:20s} shape={str(value.shape):16s} dtype={value.dtype} {limits}")


if __name__ == "__main__":
    main()
