#!/usr/bin/env python3
"""Audit the minimum artifacts for a public C&G GitHub submission release."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "DATA_AVAILABILITY.md",
    "REPRODUCIBILITY.md",
    "requirements.txt",
    "environment.yml",
    "configs/final_training.yaml",
    "configs/final_inference.yaml",
    "code/model.py",
    "code/train.py",
    "code/inference.py",
    "code/full_line_metrics.py",
    "scripts/run_paper_inference.py",
    "scripts/evaluate_full_lines.py",
    "scripts/run_synthetic_demo.py",
    "docs/USER_GUIDE.md",
    "data/example/synthetic_line.npz",
    "results/geological_usability_metrics_test7.csv",
    "zenodo/README_ZENODO.md",
    "zenodo/FILES.csv",
    "zenodo/CHECKSUMS.sha256",
    "zenodo/LICENSE_DATA.txt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def check_publication_state(failures: list[str], warnings: list[str]) -> None:
    if not (ROOT / ".git").exists():
        failures.append("The release directory is not a Git repository.")
        return

    if git("rev-parse", "--verify", "HEAD").returncode != 0:
        failures.append("The Git repository has no commit.")

    remote = git("remote", "get-url", "origin")
    if remote.returncode != 0 or not remote.stdout.strip():
        failures.append("No 'origin' remote is configured.")
    else:
        remote_url = remote.stdout.strip()
        print(f"GIT_REMOTE {remote_url}")
        if "github.com" not in remote_url.lower():
            warnings.append("The origin remote is not hosted on github.com.")
        warnings.append(
            "Verify in a browser that the remote repository and release assets are public."
        )

    status = git("status", "--porcelain")
    if status.returncode != 0:
        failures.append("Unable to inspect the Git working tree.")
    elif status.stdout.strip():
        failures.append("The Git working tree contains uncommitted changes.")

    if git("describe", "--exact-match", "--tags", "HEAD").returncode != 0:
        warnings.append("HEAD has no exact version tag; create v1.0-submission for review.")


def check_checkpoint(failures: list[str], warnings: list[str]) -> None:
    checkpoint = ROOT / "checkpoints" / "best_model.pt"
    checksum_file = ROOT / "checkpoints" / "best_model.sha256"
    if not checkpoint.is_file():
        warnings.append(
            "best_model.pt is not in the working tree; attach it to the public GitHub "
            "submission release and link that release from README.md."
        )
        return

    actual = sha256(checkpoint)
    print(f"CHECKPOINT checkpoints/best_model.pt sha256={actual}")
    if not checksum_file.is_file():
        failures.append("Missing checkpoints/best_model.sha256.")
        return
    expected = checksum_file.read_text(encoding="utf-8").split()[0].lower()
    if actual != expected:
        failures.append("best_model.pt does not match best_model.sha256.")


def check_synthetic_example(failures: list[str]) -> None:
    path = ROOT / "data" / "example" / "synthetic_line.npz"
    if not path.is_file():
        return
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted({"raw", "env", "rms", "mask"} - set(archive.files))
            if missing:
                failures.append(
                    "Synthetic example is missing arrays: " + ", ".join(missing)
                )
            else:
                shapes = {archive[key].shape for key in ("raw", "env", "rms", "mask")}
                if len(shapes) != 1:
                    failures.append("Synthetic example arrays do not share one shape.")
                else:
                    print(f"SYNTHETIC_EXAMPLE shape={next(iter(shapes))}")
    except Exception as exc:  # pragma: no cover - defensive release audit
        failures.append(f"Unable to read the synthetic example: {exc}")


def check_documentation(failures: list[str], warnings: list[str]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required_terms = [
        "## Installation",
        "## Computational requirements",
        "## Full-line inference",
        "DATA_AVAILABILITY.md",
        "REPRODUCIBILITY.md",
    ]
    for term in required_terms:
        if term not in readme:
            failures.append(f"README.md is missing required content: {term}")

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if "REPLACE" in citation or "TODO" in citation:
        failures.append("CITATION.cff contains an unresolved placeholder.")
    if "repository-code:" not in citation:
        warnings.append(
            "Add repository-code to CITATION.cff after the public GitHub URL exists."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--content-only",
        action="store_true",
        help="Check repository contents without requiring a commit, remote, or clean tree.",
    )
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    for relative_path in REQUIRED_FILES:
        if not (ROOT / relative_path).is_file():
            failures.append(f"Missing required file: {relative_path}")

    check_checkpoint(failures, warnings)
    check_synthetic_example(failures)
    if (ROOT / "README.md").is_file() and (ROOT / "CITATION.cff").is_file():
        check_documentation(failures, warnings)
    if not args.content_only:
        check_publication_state(failures, warnings)

    for warning in warnings:
        print(f"WARNING: {warning}")

    if failures:
        print("\nNOT READY FOR PUBLIC GITHUB RELEASE")
        for failure in failures:
            print(f"- {failure}")
        return 1

    if args.content_only:
        print("\nCONTENT READY; GIT PUBLICATION STATE NOT CHECKED")
    else:
        print("\nREADY FOR PUBLIC GITHUB RELEASE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
