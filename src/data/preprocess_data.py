"""Preprocess OWID COVID-19 data for epidemiological analysis."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.diagnostics.validate_timeseries import run_timeseries_diagnostics

LOGGER = logging.getLogger(__name__)

RELEVANT_COLUMNS: list[str] = [
    "date",
    "location",
    "population",
    "new_cases",
    "total_cases",
    "new_deaths",
    "total_deaths",
    "people_vaccinated",
    "people_fully_vaccinated",
    "total_boosters",
    "new_vaccinations",
    "stringency_index",
    "reproduction_rate",
    "icu_patients",
    "hosp_patients",
]

NON_NEGATIVE_COLUMNS: list[str] = [
    "population",
    "new_cases",
    "total_cases",
    "new_deaths",
    "total_deaths",
    "people_vaccinated",
    "people_fully_vaccinated",
    "total_boosters",
    "new_vaccinations",
    "stringency_index",
    "reproduction_rate",
    "icu_patients",
    "hosp_patients",
]

SLOW_CHANGING_FFILL_COLUMNS: list[str] = [
    "population",
    "people_vaccinated",
    "people_fully_vaccinated",
    "total_boosters",
]


class MissingColumnError(ValueError):
    """Raised when required columns are not available in input data."""


def load_raw_data(raw_csv_path: Path) -> pd.DataFrame:
    """Load raw OWID CSV data."""
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_csv_path}")

    dataframe = pd.read_csv(raw_csv_path, low_memory=False)
    LOGGER.info("Loaded raw data from %s (%s rows)", raw_csv_path, len(dataframe))
    return dataframe


def select_available_columns(df: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    """Return available and missing columns from a desired set."""
    available_columns = [column for column in columns if column in df.columns]
    missing_columns = [column for column in columns if column not in df.columns]
    return available_columns, missing_columns


def extract_country_data(raw_df: pd.DataFrame, country: str = "France") -> pd.DataFrame:
    """Extract and standardize one-country raw subset before cleaning rules.

    Steps:
    - filter by country
    - parse date
    - sort by date
    - drop duplicate dates (keep last)
    - set date index
    - keep only relevant available columns
    """
    required_columns = {"date", "location"}
    missing_required = [column for column in required_columns if column not in raw_df.columns]
    if missing_required:
        raise MissingColumnError(f"Missing required columns: {missing_required}")

    country_df = raw_df.loc[raw_df["location"] == country].copy()
    if country_df.empty:
        raise ValueError(f"No rows found for country='{country}'")

    country_df["date"] = pd.to_datetime(country_df["date"], errors="coerce")
    invalid_date_rows = int(country_df["date"].isna().sum())
    if invalid_date_rows:
        LOGGER.warning("Dropping %s rows with invalid dates", invalid_date_rows)
    country_df = country_df.dropna(subset=["date"])

    available_columns, missing_columns = select_available_columns(country_df, RELEVANT_COLUMNS)
    if missing_columns:
        LOGGER.warning("Missing optional columns in source data: %s", missing_columns)
    country_df = country_df[available_columns]

    country_df = country_df.sort_values("date")
    before_dedup = len(country_df)
    country_df = country_df.drop_duplicates(subset=["date"], keep="last")
    removed_duplicates = before_dedup - len(country_df)
    if removed_duplicates:
        LOGGER.warning("Removed %s duplicated date rows", removed_duplicates)

    country_df = country_df.set_index("date", drop=True).sort_index()
    country_df.index.name = "date"

    return country_df


def preprocess_country_data(
    raw_df: pd.DataFrame,
    country: str = "France",
    *,
    raw_country_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Preprocess OWID data for one country.

    Assumptions:
    - Only non-negative epidemiological quantities are clipped to NaN when negative.
    - Missing values are preserved for fast-changing indicators.
    - Forward-fill is only applied to slow-changing cumulative vaccination variables
      and population.
    """
    country_df = raw_country_df.copy() if raw_country_df is not None else extract_country_data(raw_df, country)

    for column in [c for c in NON_NEGATIVE_COLUMNS if c in country_df.columns]:
        negative_mask = country_df[column] < 0
        negative_count = int(negative_mask.sum())
        if negative_count:
            LOGGER.warning("Replacing %s negative values in '%s' with NaN", negative_count, column)
            country_df.loc[negative_mask, column] = np.nan

    for column in [c for c in SLOW_CHANGING_FFILL_COLUMNS if c in country_df.columns]:
        country_df[column] = country_df[column].ffill()

    if "new_cases" in country_df.columns:
        country_df["new_cases_7d_avg"] = country_df["new_cases"].rolling(window=7, min_periods=1).mean()
    else:
        country_df["new_cases_7d_avg"] = np.nan

    if "new_deaths" in country_df.columns:
        country_df["new_deaths_7d_avg"] = country_df["new_deaths"].rolling(window=7, min_periods=1).mean()
    else:
        country_df["new_deaths_7d_avg"] = np.nan

    country_df = country_df.sort_index()
    country_df.index.name = "date"

    return country_df


def save_processed_data(
    processed_df: pd.DataFrame,
    output_path: Path,
    *,
    save_csv: bool = False,
) -> None:
    """Save processed data as parquet and optionally CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_df = processed_df.sort_index()
    processed_df.to_parquet(output_path)
    LOGGER.info("Processed parquet saved: %s", output_path)

    if save_csv:
        csv_path = output_path.with_suffix(".csv")
        processed_df.to_csv(csv_path, index=True)
        LOGGER.info("Processed CSV saved: %s", csv_path)


def build_data_quality_report_text(processed_df: pd.DataFrame) -> str:
    """Build a concise data quality report as text."""
    if processed_df.empty:
        return "=== Data Quality Report ===\nProcessed dataframe is empty. No quality report available."

    date_start = processed_df.index.min()
    date_end = processed_df.index.max()

    lines: list[str] = []
    lines.append("=== Data Quality Report ===")
    lines.append(f"Date range: {date_start.date()} -> {date_end.date()}")
    lines.append(f"Number of rows: {len(processed_df)}")

    missing_series = processed_df.isna().sum().sort_values(ascending=False)
    lines.append("")
    lines.append("Missing values per column:")
    for column, missing_count in missing_series.items():
        lines.append(f"- {column}: {int(missing_count)}")

    numeric_df = processed_df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        lines.append("")
        lines.append("Basic statistics (numeric columns):")
        lines.append(numeric_df.describe().transpose().to_string())

    return "\n".join(lines)


def print_data_quality_report(processed_df: pd.DataFrame) -> str:
    """Print and return data quality report text."""
    report_text = build_data_quality_report_text(processed_df)
    print(f"\n{report_text}")
    return report_text


def build_output_filename(country: str) -> str:
    """Build a filesystem-safe output filename from country name."""
    slug = re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")
    return f"covid_{slug}.parquet"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess OWID COVID-19 data")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/raw/owid_covid_data.csv"),
        help="Input raw OWID CSV path",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="France",
        help="Country name as present in OWID 'location' column",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed files are written",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/processed/reports"),
        help="Directory where diagnostics reports are written",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save processed data as CSV",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    raw_df = load_raw_data(args.input_path)
    raw_country_df = extract_country_data(raw_df, country=args.country)
    processed_df = preprocess_country_data(raw_df, country=args.country, raw_country_df=raw_country_df)

    run_timeseries_diagnostics(
        country=args.country,
        raw_country_df=raw_country_df,
        processed_df=processed_df,
        reports_dir=args.reports_dir,
    )

    output_path = args.output_dir / build_output_filename(args.country)
    save_processed_data(processed_df, output_path, save_csv=args.save_csv)
    print_data_quality_report(processed_df)


if __name__ == "__main__":
    main()
