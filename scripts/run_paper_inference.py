"""Run inference with the parameters archived in the paper prediction files."""

import argparse
import subprocess
import sys
from pathlib import Path


def build_command(
    line_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    amp: bool,
    profile: str = "common",
    lines: list[str] | None = None,
) -> list[str]:
    profiles = {
        "common": {
            "post_min_area": "160",
            "post_min_trace_span": "70",
            "post_min_time_span": "6",
            "pp_close": True,
            "pp_close_y": "9",
            "pp_close_x": "3",
            "tw_gate": True,
            "tw_base_halfwin": "30",
            "tw_score_mode": "span_qprob",
            "tw_score_q": "75.0",
        },
        "hh06-17": {
            "post_min_area": "160",
            "post_min_trace_span": "70",
            "post_min_time_span": "6",
            "pp_close": True,
            "pp_close_y": "3",
            "pp_close_x": "11",
            "tw_gate": False,
            "tw_base_halfwin": "25",
            "tw_score_mode": "span_meanprob",
            "tw_score_q": "70.0",
        },
        "hh06-03-45": {
            "post_min_area": "0",
            "post_min_trace_span": "0",
            "post_min_time_span": "0",
            "pp_close": False,
            "pp_close_y": "3",
            "pp_close_x": "11",
            "tw_gate": False,
            "tw_base_halfwin": "25",
            "tw_score_mode": "span_meanprob",
            "tw_score_q": "70.0",
        },
    }
    settings = profiles[profile]
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
        "--thr", "0.35",
        "--base", "32",
        "--norm", "gn",
        "--deep-supervision",
        "--aux-heads",
        "--auto-arch",
        "--norm-mode", "raw",
        "--raw-valid-pct", "10.0",
        "--save-aux",
        "--sdf-tau", "0.12",
        "--post-min-area", settings["post_min_area"],
        "--post-min-trace-span", settings["post_min_trace_span"],
        "--post-min-time-span", settings["post_min_time_span"],
        "--pp-close-y", settings["pp_close_y"],
        "--pp-close-x", settings["pp_close_x"],
        "--suppress-invalid",
        "--invalid-local-win", "31",
        "--invalid-std-eps", "0.02",
        "--tw-thr-low", "0.15",
        "--tw-close-y", "3",
        "--tw-close-x", "7",
        "--tw-min-trace-span", "80",
        "--tw-max-thickness", "160",
        "--tw-base-halfwin", settings["tw_base_halfwin"],
        "--tw-halfwin-scale", "0.6",
        "--tw-smooth-win", "51",
        "--tw-score-mode", settings["tw_score_mode"],
        "--tw-score-q", settings["tw_score_q"],
    ]
    if settings["pp_close"]:
        command.append("--pp-close")
    if settings["tw_gate"]:
        command.append("--tw-gate")
    if lines:
        command.extend(["--lines", *lines])
    if amp:
        command.append("--amp")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("common", "hh06-17", "hh06-03-45"),
        default="common",
        help="Frozen final inference profile to apply.",
    )
    parser.add_argument(
        "--lines",
        nargs="*",
        default=None,
        help="Optional line-name tokens used to select NPZ files below --line-dir.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = build_command(
        args.line_dir,
        args.checkpoint,
        args.output_dir,
        not args.no_amp,
        profile=args.profile,
        lines=args.lines,
    )
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
