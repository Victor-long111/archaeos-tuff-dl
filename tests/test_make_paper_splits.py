import tempfile
import unittest
from pathlib import Path

from preprocessing.make_paper_splits import build_split, canonical_line_id


def _touch(root: Path, names: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).touch()


class PaperSplitTests(unittest.TestCase):
    def test_canonical_line_id_supports_paper_names(self) -> None:
        self.assertEqual(
            canonical_line_id("HH06-17_reproc_patch_01.npz"), "HH06_17"
        )
        self.assertEqual(canonical_line_id("HH06_22_patch_02.npz"), "HH06_22")
        self.assertEqual(
            canonical_line_id("MSM135_P1004_patch_03.npz"), "MSM135_P1004"
        )
        self.assertIsNone(canonical_line_id("unknown_patch.npz"))

    def test_line_holdout_precedes_stratified_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            positive = root / "positive"
            negative = root / "negative"
            output = root / "output"

            _touch(
                positive,
                [
                    *[f"HH06_01_positive_{index:02d}.npz" for index in range(8)],
                    "HH06_10_positive_heldout.npz",
                ],
            )
            _touch(
                negative,
                [
                    *[
                        f"MSM135_P1001_negative_{index:02d}.npz"
                        for index in range(12)
                    ],
                    "HH06_10_negative_heldout.npz",
                ],
            )

            summary = build_split(
                positive,
                negative,
                output,
                test_lines=["HH06_10"],
                seed=42,
                val_ratio=0.25,
            )

            self.assertEqual(summary["counts"]["train_pos"], 6)
            self.assertEqual(summary["counts"]["val_pos"], 2)
            self.assertEqual(summary["counts"]["train_neg"], 9)
            self.assertEqual(summary["counts"]["val_neg"], 3)
            self.assertEqual(summary["counts"]["test_pos"], 1)
            self.assertEqual(summary["counts"]["test_neg"], 1)
            self.assertEqual(len((output / "train.txt").read_text().splitlines()), 15)
            self.assertEqual(len((output / "val.txt").read_text().splitlines()), 5)
            self.assertEqual(
                len((output / "heldout_source_patches.txt").read_text().splitlines()),
                2,
            )


if __name__ == "__main__":
    unittest.main()
