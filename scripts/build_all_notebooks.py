"""Rebuild and execute all notebooks at once.

Usage from project root:

    .venv/bin/python scripts/build_all_notebooks.py            # build + execute
    .venv/bin/python scripts/build_all_notebooks.py --no-run   # build only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BUILDERS = [
    ("scripts/build_eda_notebook.py", "notebooks/01_eda.ipynb"),
    ("scripts/build_business_score_notebook.py", "notebooks/02_business_score.ipynb"),
    ("scripts/build_training_notebook.py", "notebooks/03_training.ipynb"),
    ("scripts/build_drift_notebook.py", "notebooks/04_drift.ipynb"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-run", action="store_true", help="Skip nbconvert execution")
    args = parser.parse_args()

    python = sys.executable

    for builder, notebook in BUILDERS:
        print(f"\n=== {builder} ===")
        subprocess.run([python, str(ROOT / builder)], check=True)

        if not args.no_run:
            print(f"  → executing {notebook}")
            subprocess.run(
                [
                    python,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    str(ROOT / notebook),
                    "--inplace",
                    "--ExecutePreprocessor.timeout=300",
                ],
                check=True,
            )

    print("\n✓ All notebooks built" + ("" if args.no_run else " and executed."))


if __name__ == "__main__":
    main()
