#!/usr/bin/env python3
"""Recreate the archived line-disjoint patch-splitting policy.

The complete patch corpus is arranged in separate ``positive`` and ``negative``
directories. Patches from the seven independent test profiles are excluded
before the remaining positive and negative pools are split independently into
training and validation subsets. The script verifies the archived aggregate
counts; the original file-level membership list is not distributed. The
independent evaluation itself uses the seven complete processed profiles, not
the held-out patch files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_TEST_LINES = (
    "HH06_10",
    "HH06_14",
    "HH06_15",
    "HH06_17",
    "HH06_22",
    "MSM135_P1004",
    "MSM135_P1018",
)


def canonical_line_id(filename: str) -> str | None:
    """Return the normalized source-line identifier encoded in a patch name."""
    hh06 = re.search(r"HH06[-_](\d{2})([A-Za-z]?)", filename, flags=re.IGNORECASE)
    if hh06:
        suffix = hh06.group(2).lower()
        return f"HH06_{hh06.group(1)}{suffix}"

    msm135 = re.search(r"MSM135[-_]?P(\d{4})", filename, flags=re.IGNORECASE)
    if msm135:
        return f"MSM135_P{msm135.group(1)}"
    return None


def discover_npz(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.rglob("*.npz"))


def split_class_pool(
    files: Iterable[Path],
    test_lines: set[str],
    val_ratio: float,
    rng: np.random.Generator,
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    development: list[Path] = []
    heldout: list[Path] = []
    unparsed: list[Path] = []

    for path in files:
        line_id = canonical_line_id(path.name)
        if line_id is None:
            unparsed.append(path)
        elif line_id in test_lines:
            heldout.append(path)
        else:
            development.append(path)

    shuffled = np.asarray(development, dtype=object)
    rng.shuffle(shuffled)
    validation_count = int(round(len(shuffled) * val_ratio))
    validation = shuffled[:validation_count].tolist()
    training = shuffled[validation_count:].tolist()
    return training, validation, heldout, unparsed


def write_list(path: Path, files: Iterable[Path]) -> None:
    values = [str(file) for file in files]
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def build_split(
    pos_root: Path,
    neg_root: Path,
    outdir: Path,
    test_lines: Iterable[str] = DEFAULT_TEST_LINES,
    seed: int = 42,
    val_ratio: float = 0.20,
) -> dict:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if not pos_root.is_dir() or not neg_root.is_dir():
        raise FileNotFoundError("Both positive and negative patch directories are required.")

    normalized_test_lines = {line.replace("-", "_") for line in test_lines}
    pos_files = discover_npz(pos_root)
    neg_files = discover_npz(neg_root)
    if not pos_files or not neg_files:
        raise RuntimeError("Both class directories must contain NPZ patch files.")

    duplicate_names = {path.name for path in pos_files} & {path.name for path in neg_files}
    if duplicate_names:
        example = sorted(duplicate_names)[0]
        raise ValueError(f"A patch filename occurs in both classes: {example}")

    rng = np.random.default_rng(seed)
    train_pos, val_pos, test_pos, bad_pos = split_class_pool(
        pos_files, normalized_test_lines, val_ratio, rng
    )
    train_neg, val_neg, test_neg, bad_neg = split_class_pool(
        neg_files, normalized_test_lines, val_ratio, rng
    )

    observed_test_lines = {
        canonical_line_id(path.name) for path in [*test_pos, *test_neg]
    }
    missing_test_lines = sorted(normalized_test_lines - observed_test_lines)
    if bad_pos or bad_neg:
        raise ValueError(
            f"Unable to parse source line for {len(bad_pos)} positive and "
            f"{len(bad_neg)} negative patches."
        )
    if missing_test_lines:
        raise ValueError(
            "No held-out patches were found for: " + ", ".join(missing_test_lines)
        )

    train = [*train_pos, *train_neg]
    validation = [*val_pos, *val_neg]
    rng.shuffle(train)
    rng.shuffle(validation)
    heldout = sorted([*test_pos, *test_neg])

    outdir.mkdir(parents=True, exist_ok=True)
    write_list(outdir / "train.txt", train)
    write_list(outdir / "val.txt", validation)
    write_list(outdir / "train_pos.txt", train_pos)
    write_list(outdir / "train_neg.txt", train_neg)
    write_list(outdir / "val_pos.txt", val_pos)
    write_list(outdir / "val_neg.txt", val_neg)
    write_list(outdir / "heldout_source_patches.txt", heldout)

    summary = {
        "seed": seed,
        "val_ratio": val_ratio,
        "source_layout": {
            "positive": "<patch_root>/positive",
            "negative": "<patch_root>/negative",
        },
        "test_lines": sorted(normalized_test_lines),
        "missing_test_lines": missing_test_lines,
        "split_policy": "PATCH_LEVEL_TRAINVAL_ON_NONTEST__FULL_LINE_TEST_HOLDOUT",
        "test_evaluation_unit": "seven complete processed seismic profiles",
        "counts": {
            "pos_files_total": len(pos_files),
            "neg_files_total": len(neg_files),
            "bad_pos_unparsed": len(bad_pos),
            "bad_neg_unparsed": len(bad_neg),
            "nontest_pos_total": len(train_pos) + len(val_pos),
            "nontest_neg_total": len(train_neg) + len(val_neg),
            "train_pos": len(train_pos),
            "train_neg": len(train_neg),
            "val_pos": len(val_pos),
            "val_neg": len(val_neg),
            "test_pos": len(test_pos),
            "test_neg": len(test_neg),
        },
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def verify_expected(actual: dict, expected_path: Path) -> None:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if actual["test_lines"] != expected["test_lines"]:
        raise RuntimeError("Generated test-line list differs from the archived paper split.")
    for key, expected_value in expected["counts"].items():
        if actual["counts"].get(key) != expected_value:
            raise RuntimeError(
                f"Split count mismatch for {key}: "
                f"generated={actual['counts'].get(key)}, expected={expected_value}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos-root", required=True, type=Path)
    parser.add_argument("--neg-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-lines", nargs="+", default=list(DEFAULT_TEST_LINES))
    parser.add_argument(
        "--verify-summary",
        type=Path,
        help="Fail if generated counts differ from the archived paper summary.",
    )
    args = parser.parse_args()

    summary = build_split(
        pos_root=args.pos_root,
        neg_root=args.neg_root,
        outdir=args.outdir,
        test_lines=args.test_lines,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    if args.verify_summary:
        verify_expected(summary, args.verify_summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
