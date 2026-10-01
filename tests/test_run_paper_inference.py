from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_paper_inference import build_command  # noqa: E402


def _value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_common_profile_matches_frozen_full_line_settings():
    command = build_command(
        Path("inputs"), Path("best_model.pt"), Path("outputs"), False
    )

    assert _value(command, "--thr") == "0.35"
    assert _value(command, "--post-min-area") == "160"
    assert _value(command, "--pp-close-y") == "9"
    assert "--pp-close" in command
    assert "--tw-gate" in command


def test_special_profiles_match_archived_prediction_parameters():
    hh06_17 = build_command(
        Path("inputs"),
        Path("best_model.pt"),
        Path("outputs"),
        False,
        profile="hh06-17",
        lines=["HH06-17"],
    )
    assert _value(hh06_17, "--thr") == "0.35"
    assert _value(hh06_17, "--pp-close-y") == "3"
    assert _value(hh06_17, "--pp-close-x") == "11"
    assert _value(hh06_17, "--tw-base-halfwin") == "25"
    assert _value(hh06_17, "--tw-score-mode") == "span_meanprob"
    assert "--tw-gate" not in hh06_17
    assert "--lines" in hh06_17

    hh06_03_45 = build_command(
        Path("inputs"),
        Path("best_model.pt"),
        Path("outputs"),
        False,
        profile="hh06-03-45",
    )
    assert _value(hh06_03_45, "--post-min-area") == "0"
    assert "--pp-close" not in hh06_03_45
    assert "--tw-gate" not in hh06_03_45
