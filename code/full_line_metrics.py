"""Geological-usability metrics for complete seismic profiles.

Arrays are stored as ``(time_samples, traces)`` and positive samples are 1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FullLineMetrics:
    top_mae_ms: float
    top_rmse_ms: float
    base_mae_ms: float
    base_rmse_ms: float
    coverage_gt_percent: float
    connectivity: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _longest_contiguous_run(indices: np.ndarray) -> int:
    if indices.size == 0:
        return 0
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    return max(len(run) for run in np.split(indices, breaks))


def calculate_full_line_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    sample_interval_ms: float = 1.0,
) -> FullLineMetrics:
    """Calculate the six metrics defined in Supplementary Methods S4."""

    reference = np.asarray(reference) == 1
    prediction = np.asarray(prediction).astype(bool)
    if reference.ndim != 2 or prediction.ndim != 2:
        raise ValueError("Reference and prediction must be two-dimensional arrays.")
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Shape mismatch: reference={reference.shape}, prediction={prediction.shape}"
        )
    if sample_interval_ms <= 0:
        raise ValueError("sample_interval_ms must be positive.")

    reference_traces = reference.any(axis=0)
    predicted_traces = prediction.any(axis=0)
    shared_traces = reference_traces & predicted_traces
    if not shared_traces.any():
        raise ValueError("No trace contains both a reference and predicted interval.")

    reference_top = np.argmax(reference, axis=0)
    predicted_top = np.argmax(prediction, axis=0)
    reference_base = reference.shape[0] - 1 - np.argmax(reference[::-1], axis=0)
    predicted_base = prediction.shape[0] - 1 - np.argmax(prediction[::-1], axis=0)

    top_error = (
        np.abs(predicted_top[shared_traces] - reference_top[shared_traces])
        * sample_interval_ms
    )
    base_error = (
        np.abs(predicted_base[shared_traces] - reference_base[shared_traces])
        * sample_interval_ms
    )

    predicted_indices = np.flatnonzero(predicted_traces)
    longest_run = _longest_contiguous_run(predicted_indices)

    return FullLineMetrics(
        top_mae_ms=float(top_error.mean()),
        top_rmse_ms=float(np.sqrt(np.mean(np.square(top_error)))),
        base_mae_ms=float(base_error.mean()),
        base_rmse_ms=float(np.sqrt(np.mean(np.square(base_error)))),
        coverage_gt_percent=float(
            100.0 * shared_traces.sum() / reference_traces.sum()
        ),
        connectivity=float(longest_run / predicted_indices.size),
    )
