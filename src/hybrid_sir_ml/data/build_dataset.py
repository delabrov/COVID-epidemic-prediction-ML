"""Build a first unified analysis dataset from raw sources.

Phase 1 goal: define target schema and materialize an empty/stub dataset artifact
that downstream notebooks and models can rely on.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TARGET_COLUMNS = [
    "date",
    "location",
    "new_cases",
    "new_cases_smoothed_7d",
    "mobility_retail_and_recreation",
    "mobility_workplaces",
    "people_vaccinated_per_hundred",
    "policy_stringency_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a phase-1 analysis dataset artifact")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Project root path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to data/processed/analysis_dataset.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root

    output_path = args.output or project_root / "data" / "processed" / "analysis_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(TARGET_COLUMNS)

    print(f"Dataset scaffold written: {output_path}")
    print(f"Columns: {', '.join(TARGET_COLUMNS)}")


if __name__ == "__main__":
    main()
