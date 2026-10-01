"""Run inference with the parameters archived in the paper prediction files."""

import argparse
import subprocess
import sys
from pathlib import Path


def build_command(line_dir: Path, checkpoint: Path, output_dir: Path, amp: bool) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "code" / "inference.py"),
        "--line-dir", str(line_dir),
        "--ckpt", str(checkpoint),
        "--outdir", str(output_dir),
        "--input-keys", "raw", "env", "rms",
        "--patch", "256",
        "--stride", "128",
        "--thr", "0.5",
        "--base", "32",
        "--norm", "gn",
        "--deep-supervision",
        "--aux-heads",
        "--auto-arch",
        "--norm-mode", "raw",
        "--raw-valid-pct", "10.0",
        "--save-aux",
        "--sdf-tau", "0.12",
        "--post-min-area", "160",
        "--post-min-trace-span", "70",
        "--post-min-time-span", "6",
        "--pp-close",
        "--pp-close-y", "9",
        "--pp-close-x", "3",
        "--suppress-invalid",
        "--invalid-local-win", "31",
        "--invalid-std-eps", "0.02",
        "--tw-gate",
        "--tw-thr-low", "0.15",
        "--tw-close-y", "3",
        "--tw-close-x", "7",
        "--tw-min-trace-span", "80",
        "--tw-max-thickness", "160",
        "--tw-base-halfwin", "30",
        "--tw-halfwin-scale", "0.6",
        "--tw-smooth-win", "51",
        "--tw-score-mode", "span_qprob",
        "--tw-score-q", "75.0",
    ]
    if amp:
        command.append("--amp")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = build_command(args.line_dir, args.checkpoint, args.output_dir, not args.no_amp)
    print(" ".join(f'"{item}"' if " " in item else item for item in command))
    if args.dry_run:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Final checkpoint not found: {args.checkpoint}")
    if not args.line_dir.is_dir():
        raise NotADirectoryError(f"Test-line directory not found: {args.line_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
