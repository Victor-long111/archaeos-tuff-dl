import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from full_line_metrics import calculate_full_line_metrics  # noqa: E402


def test_full_line_metrics_follow_supplementary_methods_s4():
    reference = np.zeros((8, 6), dtype=np.uint8)
    prediction = np.zeros_like(reference)
    reference[2:5, 1:5] = 1
    prediction[3:6, 1:3] = 1
    prediction[2:5, 4] = 1

    metrics = calculate_full_line_metrics(reference, prediction)

    assert metrics.top_mae_ms == 2 / 3
    assert metrics.top_rmse_ms == np.sqrt(2 / 3)
    assert metrics.base_mae_ms == 2 / 3
    assert metrics.base_rmse_ms == np.sqrt(2 / 3)
    assert metrics.coverage_gt_percent == 75.0
    assert metrics.connectivity == 2 / 3
