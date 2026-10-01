#!/usr/bin/env python3
"""Build the public representative-patch inventory without evaluation overlap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import numpy as np


EXCLUDED_LINES = {
    "HH06-03",
    "HH06-08",
    "HH06-10",
    "HH06-14",
    "HH06-15",
    "HH06-17",
    "HH06-22",
    "HH06-40",
    "HH06-41",
    "HH06-45",
    "MSM135-P1004",
    "MSM135-P1018",
    "MSM135-P1312",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_line(filename: str) -> str:
    match = re.match(r"^(HH06-\d+|MSM135[_-]P?\d+)", filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unable to identify source line from {filename}")
    return match.group(1).upper().replace("_", "-")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-list", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, str | int | float]] = []
    asset_paths: list[Path] = []
    for class_name in ("Positive", "Negative"):
        directory = args.patch_root / class_name
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        for path in sorted(directory.glob("*.npz"), key=lambda item: item.name.lower()):
            line = source_line(path.name)
            if line in EXCLUDED_LINES:
                continue
            with np.load(path, allow_pickle=False) as archive:
                required = {"raw", "env", "rms", "mask"}
                missing = sorted(required - set(archive.files))
                if missing:
                    raise ValueError(f"{path.name}: missing {', '.join(missing)}")
                arrays = [archive[key] for key in ("raw", "env", "rms", "mask")]
                shapes = {array.shape for array in arrays}
                if len(shapes) != 1 or len(arrays[0].shape) != 2:
                    raise ValueError(f"{path.name}: arrays do not share one 2-D shape")
                if any(not np.isfinite(array).all() for array in arrays):
                    raise ValueError(f"{path.name}: required arrays contain non-finite values")
                mask = archive["mask"]
                positive_pixels = int(np.count_nonzero(mask > 0))
                height, width = mask.shape

            expected_positive = class_name == "Positive"
            if expected_positive != (positive_pixels > 0):
                raise ValueError(f"{path.name}: directory class disagrees with mask")
            size = path.stat().st_size
            rows.append(
                {
                    "filename": path.name,
                    "class": class_name.lower(),
                    "source_line": line,
                    "height": height,
                    "width": width,
                    "size_bytes": size,
                    "mask_positive_pixels": positive_pixels,
                    "mask_positive_percent": f"{100.0 * positive_pixels / mask.size:.6f}",
                    "sha256": sha256(path),
                    "release_role": "representative_training_patch",
                }
            )
            asset_paths.append(path.resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "patch_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checksums = args.output_dir / "CHECKSUMS_PATCHES.sha256"
    checksums.write_text(
        "".join(f"{row['sha256']}  {row['filename']}\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )

    if args.asset_list:
        args.asset_list.parent.mkdir(parents=True, exist_ok=True)
        args.asset_list.write_text(
            "".join(f"{path}\n" for path in asset_paths),
            encoding="utf-8",
            newline="\n",
        )

    positive = sum(row["class"] == "positive" for row in rows)
    negative = sum(row["class"] == "negative" for row in rows)
    total_bytes = sum(int(row["size_bytes"]) for row in rows)
    print(
        f"PATCH_RELEASE_READY files={len(rows)} positive={positive} "
        f"negative={negative} bytes={total_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
