"""Temporal validation and diagnostics for processed epidemiological time series."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_KEY_COLUMNS: list[str] = [
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
    "new_cases_7d_avg",
    "new_deaths_7d_avg",
]

DEFAULT_CUMULATIVE_COLUMNS: list[str] = [
    "total_cases",
    "total_deaths",
    "people_vaccinated",
    "people_fully_vaccinated",
    "total_boosters",
]

DEFAULT_INVALID_NEGATIVE_COLUMNS: list[str] = [
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


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with DatetimeIndex sorted in ascending order."""
    output = df.copy()

    if isinstance(output.index, pd.DatetimeIndex):
        output.index = pd.to_datetime(output.index, errors="coerce")
    elif "date" in output.columns:
        output["date"] = pd.to_datetime(output["date"], errors="coerce")
        output = output.dropna(subset=["date"]).set_index("date")
    else:
        raise ValueError("Dataframe must have DatetimeIndex or a 'date' column.")

    output = output[~output.index.isna()]
    output = output.sort_index()
    output.index.name = "date"
    return output


def check_index_properties(df: pd.DataFrame) -> dict[str, Any]:
    """Check index sort, uniqueness, inferred frequency, and expected daily length."""
    data = _ensure_datetime_index(df)
    index = data.index

    if len(index) == 0:
        return {
            "is_monotonic": True,
            "is_unique": True,
            "inferred_frequency": None,
            "expected_daily_length": 0,
            "observed_length": 0,
            "min_date": None,
            "max_date": None,
        }

    expected_daily_length = len(pd.date_range(index.min(), index.max(), freq="D"))
    inferred_frequency = None
    if index.is_monotonic_increasing and index.is_unique and len(index) >= 3:
        inferred_frequency = pd.infer_freq(index)

    return {
        "is_monotonic": bool(index.is_monotonic_increasing),
        "is_unique": bool(index.is_unique),
        "inferred_frequency": inferred_frequency,
        "expected_daily_length": int(expected_daily_length),
        "observed_length": int(len(index)),
        "min_date": index.min(),
        "max_date": index.max(),
    }


def find_missing_dates(df: pd.DataFrame) -> dict[str, Any]:
    """Find missing calendar dates between min and max date."""
    data = _ensure_datetime_index(df)
    index = data.index

    if len(index) == 0:
        empty_range = pd.DatetimeIndex([], name="date")
        return {
            "expected_range": empty_range,
            "missing_dates": empty_range,
            "missing_count": 0,
        }

    expected_range = pd.date_range(index.min(), index.max(), freq="D", name="date")
    observed_unique = pd.DatetimeIndex(sorted(index.unique()), name="date")
    missing_dates = expected_range.difference(observed_unique)

    return {
        "expected_range": expected_range,
        "missing_dates": missing_dates,
        "missing_count": int(len(missing_dates)),
    }


def compute_variable_coverage(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute non-null coverage metrics for selected variables."""
    data = _ensure_datetime_index(df)
    row_count = len(data)
    rows: list[dict[str, Any]] = []

    for column in columns:
        if column not in data.columns:
            rows.append(
                {
                    "variable": column,
                    "present_in_dataset": False,
                    "non_null_count": 0,
                    "missing_count": int(row_count),
                    "missing_pct": float(100.0 if row_count else 0.0),
                    "first_non_null_date": pd.NaT,
                    "last_non_null_date": pd.NaT,
                }
            )
            continue

        series = data[column]
        non_null_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        missing_pct = float((missing_count / row_count * 100.0) if row_count else 0.0)

        rows.append(
            {
                "variable": column,
                "present_in_dataset": True,
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 2),
                "first_non_null_date": series.first_valid_index(),
                "last_non_null_date": series.last_valid_index(),
            }
        )

    coverage_df = pd.DataFrame(rows)
    return coverage_df


def detect_trailing_missing_stretches(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Count consecutive missing values at the end of each selected series."""
    data = _ensure_datetime_index(df)
    total_rows = len(data)
    rows: list[dict[str, Any]] = []

    for column in columns:
        if column not in data.columns:
            trailing_missing_count = int(total_rows)
            rows.append(
                {
                    "variable": column,
                    "present_in_dataset": False,
                    "trailing_missing_count": trailing_missing_count,
                    "trailing_missing_pct": round(100.0 if total_rows else 0.0, 2),
                }
            )
            continue

        series = data[column]
        trailing_missing_count = 0
        for value in reversed(series.tolist()):
            if pd.isna(value):
                trailing_missing_count += 1
            else:
                break

        trailing_missing_pct = (trailing_missing_count / total_rows * 100.0) if total_rows else 0.0
        rows.append(
            {
                "variable": column,
                "present_in_dataset": True,
                "trailing_missing_count": int(trailing_missing_count),
                "trailing_missing_pct": round(float(trailing_missing_pct), 2),
            }
        )

    return pd.DataFrame(rows)


def detect_cumulative_plateaus(
    df: pd.DataFrame,
    cumulative_columns: list[str],
    trailing_window: int = 30,
) -> pd.DataFrame:
    """Detect potentially suspicious cumulative plateaus near series end."""
    data = _ensure_datetime_index(df)
    rows: list[dict[str, Any]] = []

    for column in cumulative_columns:
        if column not in data.columns:
            rows.append(
                {
                    "variable": column,
                    "present_in_dataset": False,
                    "trailing_window": trailing_window,
                    "tail_non_null_count": 0,
                    "tail_distinct_values": 0,
                    "is_tail_constant": False,
                    "end_plateau_length": 0,
                    "suspicious_plateau": False,
                    "last_non_null_date": pd.NaT,
                }
            )
            continue

        series = data[column]
        non_null_series = series.dropna()
        if non_null_series.empty:
            rows.append(
                {
                    "variable": column,
                    "present_in_dataset": True,
                    "trailing_window": trailing_window,
                    "tail_non_null_count": 0,
                    "tail_distinct_values": 0,
                    "is_tail_constant": False,
                    "end_plateau_length": 0,
                    "suspicious_plateau": False,
                    "last_non_null_date": pd.NaT,
                }
            )
            continue

        tail = non_null_series.tail(trailing_window)
        tail_distinct_values = int(tail.nunique(dropna=True))
        is_tail_constant = bool(len(tail) > 0 and tail_distinct_values == 1)

        end_plateau_length = 0
        last_value = tail.iloc[-1]
        for value in reversed(tail.tolist()):
            if value == last_value:
                end_plateau_length += 1
            else:
                break

        suspicious_plateau = bool(end_plateau_length >= min(14, max(7, trailing_window // 2)))

        rows.append(
            {
                "variable": column,
                "present_in_dataset": True,
                "trailing_window": trailing_window,
                "tail_non_null_count": int(len(tail)),
                "tail_distinct_values": tail_distinct_values,
                "is_tail_constant": is_tail_constant,
                "end_plateau_length": int(end_plateau_length),
                "suspicious_plateau": suspicious_plateau,
                "last_non_null_date": non_null_series.index.max(),
            }
        )

    return pd.DataFrame(rows)


def summarize_cleaning_impact(
    raw_country_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    columns: list[str],
    invalid_negative_columns: list[str],
) -> pd.DataFrame:
    """Compare missing and negative values in raw vs processed country data."""
    raw_data = _ensure_datetime_index(raw_country_df)
    processed_data = _ensure_datetime_index(processed_df)

    rows: list[dict[str, Any]] = []
    for column in columns:
        if column in raw_data.columns:
            raw_series = raw_data[column]
            raw_missing_count = int(raw_series.isna().sum())
            raw_negative_count = (
                int((raw_series < 0).sum())
                if column in invalid_negative_columns and pd.api.types.is_numeric_dtype(raw_series)
                else 0
            )
        else:
            raw_missing_count = int(len(raw_data))
            raw_negative_count = 0

        if column in processed_data.columns:
            final_missing_count = int(processed_data[column].isna().sum())
        else:
            final_missing_count = int(len(processed_data))

        rows.append(
            {
                "variable": column,
                "raw_missing_count": raw_missing_count,
                "raw_negative_count": raw_negative_count,
                "final_missing_count": final_missing_count,
            }
        )

    return pd.DataFrame(rows)


def _build_report_text(
    country: str,
    processed_df: pd.DataFrame,
    index_properties: dict[str, Any],
    missing_dates_info: dict[str, Any],
    coverage_df: pd.DataFrame,
    cleaning_impact_df: pd.DataFrame,
    trailing_missing_df: pd.DataFrame,
    plateau_df: pd.DataFrame,
) -> str:
    """Build a human-readable diagnostics report."""
    lines: list[str] = []
    lines.append("COVID Time Series Validation Report")
    lines.append("=" * 40)
    lines.append("")

    lines.append("1. Dataset overview")
    lines.append(f"- Country: {country}")
    min_date = index_properties.get("min_date")
    max_date = index_properties.get("max_date")
    lines.append(f"- Min date: {min_date.date() if pd.notna(min_date) else 'N/A'}")
    lines.append(f"- Max date: {max_date.date() if pd.notna(max_date) else 'N/A'}")
    lines.append(f"- Number of rows: {len(processed_df)}")
    lines.append("")

    lines.append("2. Index validation")
    lines.append(f"- Sorted (monotonic): {index_properties['is_monotonic']}")
    lines.append(f"- Unique: {index_properties['is_unique']}")
    lines.append(f"- Inferred frequency: {index_properties['inferred_frequency']}")
    lines.append(f"- Expected daily rows: {index_properties['expected_daily_length']}")
    lines.append(f"- Actual rows: {index_properties['observed_length']}")
    lines.append(f"- Missing calendar dates: {missing_dates_info['missing_count']}")
    lines.append("")

    lines.append("3. Missing calendar dates")
    if missing_dates_info["missing_count"] == 0:
        lines.append("- No missing calendar dates detected.")
    else:
        preview = [d.strftime("%Y-%m-%d") for d in missing_dates_info["missing_dates"][:20]]
        lines.append(f"- First missing dates (up to 20): {preview}")
    lines.append("")

    lines.append("4. Variable coverage summary")
    lines.append(coverage_df.to_string(index=False))
    lines.append("")

    lines.append("5. Cleaning impact summary")
    lines.append(cleaning_impact_df.to_string(index=False))
    lines.append("")

    lines.append("6. Trailing missing diagnostics")
    lines.append(trailing_missing_df.to_string(index=False))
    lines.append("")

    lines.append("7. Tail plateau diagnostics")
    lines.append(plateau_df.to_string(index=False))
    lines.append("")

    return "\n".join(lines)


def _slugify_country(country: str) -> str:
    """Build filesystem-safe country slug."""
    return re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")


def run_timeseries_diagnostics(
    country: str,
    raw_country_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    reports_dir: Path,
    *,
    key_columns: list[str] | None = None,
    cumulative_columns: list[str] | None = None,
    invalid_negative_columns: list[str] | None = None,
    trailing_window: int = 30,
) -> dict[str, Any]:
    """Run full temporal diagnostics, print summary, and save report artifacts."""
    key_columns = key_columns or DEFAULT_KEY_COLUMNS
    cumulative_columns = cumulative_columns or DEFAULT_CUMULATIVE_COLUMNS
    invalid_negative_columns = invalid_negative_columns or DEFAULT_INVALID_NEGATIVE_COLUMNS

    processed_data = _ensure_datetime_index(processed_df)
    raw_country_data = _ensure_datetime_index(raw_country_df)

    index_properties = check_index_properties(processed_data)
    missing_dates_info = find_missing_dates(processed_data)
    coverage_df = compute_variable_coverage(processed_data, key_columns)
    cleaning_impact_df = summarize_cleaning_impact(
        raw_country_data,
        processed_data,
        columns=key_columns,
        invalid_negative_columns=invalid_negative_columns,
    )
    trailing_missing_df = detect_trailing_missing_stretches(processed_data, key_columns)
    plateau_df = detect_cumulative_plateaus(
        processed_data,
        cumulative_columns,
        trailing_window=trailing_window,
    )

    report_text = _build_report_text(
        country=country,
        processed_df=processed_data,
        index_properties=index_properties,
        missing_dates_info=missing_dates_info,
        coverage_df=coverage_df,
        cleaning_impact_df=cleaning_impact_df,
        trailing_missing_df=trailing_missing_df,
        plateau_df=plateau_df,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    country_slug = _slugify_country(country)

    report_path = reports_dir / f"covid_{country_slug}_validation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    coverage_path = reports_dir / f"covid_{country_slug}_coverage_summary.csv"
    cleaning_path = reports_dir / f"covid_{country_slug}_cleaning_impact_summary.csv"
    missing_dates_path = reports_dir / f"covid_{country_slug}_missing_dates.csv"

    coverage_df.to_csv(coverage_path, index=False)
    cleaning_impact_df.to_csv(cleaning_path, index=False)
    pd.DataFrame({"missing_date": missing_dates_info["missing_dates"]}).to_csv(missing_dates_path, index=False)

    print("\n=== Temporal Validation Report ===")
    print(report_text)

    LOGGER.info("Validation report saved: %s", report_path)
    LOGGER.info("Coverage summary saved: %s", coverage_path)
    LOGGER.info("Cleaning impact summary saved: %s", cleaning_path)
    LOGGER.info("Missing dates saved: %s", missing_dates_path)

    return {
        "index_properties": index_properties,
        "missing_dates_info": missing_dates_info,
        "coverage_summary": coverage_df,
        "cleaning_impact_summary": cleaning_impact_df,
        "trailing_missing_summary": trailing_missing_df,
        "plateau_summary": plateau_df,
        "report_text": report_text,
        "report_path": report_path,
        "coverage_path": coverage_path,
        "cleaning_path": cleaning_path,
        "missing_dates_path": missing_dates_path,
    }
