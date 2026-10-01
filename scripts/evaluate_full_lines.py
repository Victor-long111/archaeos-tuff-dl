#!/usr/bin/env python3
"""Recompute the six complete-profile metrics reported in the manuscript."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from full_line_metrics import calculate_full_line_metrics  # noqa: E402


REFERENCE_KEYS = ("mask", "gt", "label", "y", "target")
CSV_COLUMNS = (
    "Test_line",
    "Top_MAE_ms",
    "Top_RMSE_ms",
    "Base_MAE_ms",
    "Base_RMSE_ms",
    "Coverage_gt_percent",
    "Connectivity",
)


def canonical_line_id(stem: str) -> str:
    for suffix in ("_pred", "_reproc_Jonas", "_Reproc_MH"):
        stem = stem.removesuffix(suffix)
    return stem.replace("-", "_")


def load_reference(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as archive:
        for key in REFERENCE_KEYS:
            if key in archive.files:
                return np.asarray(archive[key])
    raise KeyError(f"No reference key {REFERENCE_KEYS} found in {path}")


def load_prediction(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f"Prediction key {key!r} not found in {path}")
        return np.asarray(archive[key])


def evaluate(
    reference_dir: Path,
    prediction_dir: Path,
    prediction_key: str,
    sample_interval_ms: float,
) -> list[dict[str, float | str]]:
    prediction_by_id = {
        canonical_line_id(path.stem): path for path in prediction_dir.rglob("*.npz")
    }
    rows: list[dict[str, float | str]] = []
    for reference_path in sorted(reference_dir.glob("*.npz")):
        line_id = canonical_line_id(reference_path.stem)
        prediction_path = prediction_by_id.get(line_id)
        if prediction_path is None:
            raise FileNotFoundError(f"No prediction NPZ found for {reference_path.name}")
        values = calculate_full_line_metrics(
            load_reference(reference_path),
            load_prediction(prediction_path, prediction_key),
            sample_interval_ms=sample_interval_ms,
        )
        rows.append(
            {
                "Test_line": line_id,
                "Top_MAE_ms": values.top_mae_ms,
                "Top_RMSE_ms": values.top_rmse_ms,
                "Base_MAE_ms": values.base_mae_ms,
                "Base_RMSE_ms": values.base_rmse_ms,
                "Coverage_gt_percent": values.coverage_gt_percent,
                "Connectivity": values.connectivity,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No reference NPZ files found in {reference_dir}")
    return rows


def write_csv(rows: list[dict[str, float | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def compare_csv(
    rows: list[dict[str, float | str]], expected_path: Path, tolerance: float
) -> list[str]:
    with expected_path.open(newline="", encoding="utf-8-sig") as stream:
        expected = {
            canonical_line_id(row["Test_line"]): row for row in csv.DictReader(stream)
        }
    differences: list[str] = []
    for row in rows:
        line_id = str(row["Test_line"])
        if line_id not in expected:
            differences.append(f"{line_id}: missing from expected table")
            continue
        for column in CSV_COLUMNS[1:]:
            actual = float(row[column])
            target = float(expected[line_id][column])
            if abs(actual - target) > tolerance:
                differences.append(
                    f"{line_id} {column}: recomputed={actual:.9g}, expected={target:.9g}"
                )
    return differences


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-key", default="pred_bin_pp")
    parser.add_argument("--sample-interval-ms", type=float, default=1.0)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    rows = evaluate(
        args.reference_dir,
        args.prediction_dir,
        args.prediction_key,
        args.sample_interval_ms,
    )
    write_csv(rows, args.output)
    print(f"WROTE {len(rows)} line(s) to {args.output}")

    if args.compare:
        differences = compare_csv(rows, args.compare, args.tolerance)
        if differences:
            print("METRIC_MISMATCH")
            for difference in differences:
                print(f"- {difference}")
            return 1
        print("METRICS_MATCH_EXPECTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
