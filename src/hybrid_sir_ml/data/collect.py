"""Download and stage raw epidemiology datasets.

This script is intentionally minimal for phase 1 scaffolding.
It creates source-specific folders in data/raw and writes download metadata.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class DataSource:
    name: str
    description: str
    url: str
    filename: str


SOURCES = [
    DataSource(
        name="owid",
        description="Our World In Data COVID-19 dataset",
        url="https://covid.ourworldindata.org/data/owid-covid-data.csv",
        filename="owid-covid-data.csv",
    ),
    DataSource(
        name="google_mobility",
        description="Google Community Mobility Reports",
        url="https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv",
        filename="Global_Mobility_Report.csv",
    ),
    DataSource(
        name="oxcgrt",
        description="Oxford COVID-19 Government Response Tracker",
        url="https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_compact_national_v1.csv",
        filename="OxCGRT_compact_national_v1.csv",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data source registry and raw data folders")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Project root path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root

    raw_root = project_root / "data" / "raw"
    metadata_root = project_root / "data" / "metadata"
    raw_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    for source in SOURCES:
        (raw_root / source.name).mkdir(parents=True, exist_ok=True)

    registry = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [asdict(source) for source in SOURCES],
        "note": "Download logic will be added in the next implementation step.",
    }

    registry_path = metadata_root / "sources_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"Source registry written: {registry_path}")
    print("Raw directories prepared:")
    for source in SOURCES:
        print(f"- {raw_root / source.name}")


if __name__ == "__main__":
    main()
