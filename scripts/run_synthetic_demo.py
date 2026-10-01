"""Run an end-to-end model demonstration on the synthetic seismic example."""

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from model import AttentionResUNet  # noqa: E402


def main() -> None:
    source = ROOT / "data" / "example" / "synthetic_line.npz"
    output = ROOT / "outputs" / "synthetic_demo_prediction.npz"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing {source}. Run scripts/create_synthetic_example.py first."
        )

    with np.load(source) as archive:
        channels = [np.asarray(archive[key], dtype=np.float32) for key in ("raw", "env", "rms")]
    array = np.stack(channels, axis=0)[:, :256, :256]
    mean = array.mean(axis=(1, 2), keepdims=True)
    std = array.std(axis=(1, 2), keepdims=True).clip(min=1e-6)
    tensor = torch.from_numpy(((array - mean) / std)[None])

    torch.manual_seed(42)
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
        model_output = model(tensor)
        logits = model_output["seg"][-1]
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        seg_prob=probability.astype(np.float32),
        pred_bin=(probability >= 0.5).astype(np.uint8),
        threshold=np.float32(0.5),
        note=np.array(["random-weight software demonstration"], dtype="U40"),
    )
    print(f"Saved demonstration output: {output}")
    print("The random-weight output is a software test, not a scientific result.")


if __name__ == "__main__":
    main()
