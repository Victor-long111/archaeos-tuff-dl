from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from inference import resolve_line_files  # noqa: E402


def test_line_directory_without_tokens_discovers_all_npz(tmp_path):
    first = tmp_path / "HH06-10_reproc_Jonas.npz"
    second = tmp_path / "nested" / "MSM135_P1004_Reproc_MH.npz"
    ignored = tmp_path / "notes.txt"
    second.parent.mkdir()
    first.touch()
    second.touch()
    ignored.touch()

    resolved = resolve_line_files(None, str(tmp_path), None)

    assert [Path(path).name for path in resolved] == [first.name, second.name]


def test_line_tokens_filter_directory_and_accept_hyphen_underscore(tmp_path):
    wanted = tmp_path / "HH06-10_reproc_Jonas.npz"
    other = tmp_path / "HH06-14_reproc_Jonas.npz"
    wanted.touch()
    other.touch()

    resolved = resolve_line_files(None, str(tmp_path), ["HH06_10"])

    assert [Path(path).name for path in resolved] == [wanted.name]
