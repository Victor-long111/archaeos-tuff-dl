"""Create a small synthetic NPZ file for installation and I/O smoke tests."""

from pathlib import Path

import numpy as np


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "data" / "example" / "synthetic_line.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    height, width = 512, 512
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    reflector = np.sin(2 * np.pi * (18 * y + 2 * x))
    raw = (reflector + 0.25 * rng.standard_normal((height, width))).astype(np.float32)
    env = np.abs(raw).astype(np.float32)

    kernel = np.ones(15, dtype=np.float32) / 15.0
    rms = np.apply_along_axis(
        lambda column: np.sqrt(np.convolve(column**2, kernel, mode="same")),
        axis=0,
        arr=raw,
    ).astype(np.float32)

    top = (210 + 22 * np.sin(2 * np.pi * x)).astype(np.int32)
    base = top + 45
    rows = np.arange(height)[:, None]
    mask = ((rows >= top) & (rows <= base)).astype(np.uint8)

    np.savez_compressed(output, raw=raw, env=env, rms=rms, mask=mask)
    print(f"Created functional test file: {output}")
    print("This synthetic file is not scientific validation data.")


if __name__ == "__main__":
    main()

