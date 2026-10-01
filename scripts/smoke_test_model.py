"""Run a minimal CPU forward pass through the paper model."""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from model import AttentionResUNet  # noqa: E402


def main() -> None:
    model = AttentionResUNet(
        in_channels=3,
        base=32,
        norm="gn",
        use_att=True,
        deep_supervision=True,
        aux_heads=True,
        dropout_p=0.3,
    ).eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))

    print(f"Model output type: {type(output).__name__}")
    print("MODEL_FORWARD_OK")


if __name__ == "__main__":
    main()
