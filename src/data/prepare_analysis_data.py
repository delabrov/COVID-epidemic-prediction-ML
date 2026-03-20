"""Prepare a study-ready COVID analysis dataset from processed country data.

This module performs three key steps:
1. Determine a robust study window.
2. Stabilize temporal granularity (daily + weekly outputs).
3. Add population-normalized epidemiological variables.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_STUDY_KEY_COLUMNS: list[str] = [
    "new_cases",
    "new_deaths",
    "new_cases_7d_avg",
    "new_deaths_7d_avg",
    "new_vaccinations",
    "stringency_index",
    "reproduction_rate",
    "icu_patients",
    "hosp_patients",
]

DEFAULT_FLOW_COLUMNS: list[str] = ["new_cases", "new_deaths"]

PER_100K_COLUMNS: list[str] = [
    "new_cases",
    "new_deaths",
    "total_cases",
    "total_deaths",
    "new_vaccinations",
    "icu_patients",
    "hosp_patients",
]

PCT_POP_COLUMNS: list[str] = [
    "total_cases",
    "total_deaths",
    "people_vaccinated",
    "people_fully_vaccinated",
    "total_boosters",
]


@dataclass
class StudyWindowResult:
    """Container for selected study window diagnostics."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    selected_days: int
    base_start_date: pd.Timestamp
    base_end_date: pd.Timestamp
    selection_reason: str
    min_row_coverage: float
    min_window_days: int
    row_coverage: pd.Series
    segments_df: pd.DataFrame
    key_columns_used: list[str]


def _slugify_country(country: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe has sorted DatetimeIndex named 'date'."""
    output = df.copy()

    if isinstance(output.index, pd.DatetimeIndex):
        output.index = pd.to_datetime(output.index, errors="coerce")
    elif "date" in output.columns:
        output["date"] = pd.to_datetime(output["date"], errors="coerce")
        output = output.dropna(subset=["date"]).set_index("date")
    else:
        raise ValueError("Dataframe must have DatetimeIndex or a 'date' column")

    output = output[~output.index.isna()].sort_index()
    output.index.name = "date"
    return output


def _segments_from_mask(mask: pd.Series) -> pd.DataFrame:
    """Convert boolean mask to contiguous true segments."""
    mask = mask.astype(bool)
    rows: list[dict[str, Any]] = []
    start = None

    for date, is_true in mask.items():
        if is_true and start is None:
            start = date
        elif not is_true and start is not None:
            end = prev_date
            length_days = int((end - start).days + 1)
            rows.append({"start_date": start, "end_date": end, "length_days": length_days})
            start = None
        prev_date = date

    if start is not None:
        end = mask.index[-1]
        length_days = int((end - start).days + 1)
        rows.append({"start_date": start, "end_date": end, "length_days": length_days})

    return pd.DataFrame(rows)


def _get_common_non_null_bounds(df: pd.DataFrame, columns: list[str]) -> tuple[pd.Timestamp, pd.Timestamp, list[str]]:
    """Return [max first-valid, min last-valid] bounds for available columns."""
    available = [column for column in columns if column in df.columns]
    if not available:
        return df.index.min(), df.index.max(), []

    first_dates: list[pd.Timestamp] = []
    last_dates: list[pd.Timestamp] = []

    for column in available:
        series = df[column]
        first_date = series.first_valid_index()
        last_date = series.last_valid_index()
        if first_date is not None and last_date is not None:
            first_dates.append(first_date)
            last_dates.append(last_date)

    if not first_dates or not last_dates:
        return df.index.min(), df.index.max(), available

    return max(first_dates), min(last_dates), available


def compute_row_coverage(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Compute per-day non-null ratio over selected columns."""
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series(0.0, index=df.index, name="row_coverage")

    coverage = df[available].notna().mean(axis=1)
    coverage.name = "row_coverage"
    return coverage


def select_study_window(
    df: pd.DataFrame,
    *,
    key_columns: list[str],
    flow_columns: list[str] | None = None,
    min_row_coverage: float = 0.5,
    min_window_days: int = 180,
) -> StudyWindowResult:
    """Select the most reliable contiguous study window.

    Strategy:
    - Restrict candidate range to common non-null bounds of flow columns.
    - Compute per-day row coverage on key columns.
    - Keep contiguous segments where coverage >= threshold.
    - Select longest segment (prefer those >= min_window_days).
    """
    data = _ensure_datetime_index(df)
    flow_columns = flow_columns or DEFAULT_FLOW_COLUMNS

    base_start, base_end, _ = _get_common_non_null_bounds(data, flow_columns)
    base_mask = (data.index >= base_start) & (data.index <= base_end)

    key_columns_used = [column for column in key_columns if column in data.columns]
    row_coverage = compute_row_coverage(data, key_columns_used)

    candidate_mask = pd.Series(base_mask, index=data.index) & (row_coverage >= min_row_coverage)
    segments_df = _segments_from_mask(candidate_mask)

    if segments_df.empty:
        selected_start = base_start
        selected_end = base_end
        reason = "No high-coverage contiguous segment found; fallback to flow-bounded range"
    else:
        eligible = segments_df[segments_df["length_days"] >= min_window_days]
        selected_pool = eligible if not eligible.empty else segments_df
        selected = selected_pool.sort_values("length_days", ascending=False).iloc[0]
        selected_start = pd.Timestamp(selected["start_date"])
        selected_end = pd.Timestamp(selected["end_date"])
        if eligible.empty:
            reason = "Selected longest available segment below requested min_window_days"
        else:
            reason = "Selected longest contiguous segment meeting coverage threshold"

    selected_days = int((selected_end - selected_start).days + 1)

    return StudyWindowResult(
        start_date=selected_start,
        end_date=selected_end,
        selected_days=selected_days,
        base_start_date=pd.Timestamp(base_start),
        base_end_date=pd.Timestamp(base_end),
        selection_reason=reason,
        min_row_coverage=float(min_row_coverage),
        min_window_days=int(min_window_days),
        row_coverage=row_coverage,
        segments_df=segments_df,
        key_columns_used=key_columns_used,
    )


def add_population_normalized_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-100k and percent-population epidemiological variables."""
    data = df.copy()

    if "population" not in data.columns:
        LOGGER.warning("Population column missing: normalization skipped")
        return data

    safe_pop = data["population"].where(data["population"] > 0)

    for column in PER_100K_COLUMNS:
        if column in data.columns:
            data[f"{column}_per_100k"] = (data[column] / safe_pop) * 100_000.0

    for column in PCT_POP_COLUMNS:
        if column in data.columns:
            data[f"{column}_pct_population"] = (data[column] / safe_pop) * 100.0

    return data


def build_weekly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily study dataset to weekly granularity."""
    data = _ensure_datetime_index(df)

    sum_candidates = ["new_cases", "new_deaths", "new_vaccinations"]
    mean_candidates = [
        "stringency_index",
        "reproduction_rate",
        "icu_patients",
        "hosp_patients",
        "new_cases_7d_avg",
        "new_deaths_7d_avg",
        "row_coverage",
    ]
    last_candidates = [
        "population",
        "total_cases",
        "total_deaths",
        "people_vaccinated",
        "people_fully_vaccinated",
        "total_boosters",
    ]

    sum_columns = [column for column in sum_candidates if column in data.columns]
    mean_columns = [column for column in mean_candidates if column in data.columns]
    last_columns = [column for column in last_candidates if column in data.columns]

    weekly_parts: list[pd.DataFrame] = []

    if sum_columns:
        weekly_parts.append(data[sum_columns].resample("W-SUN").sum(min_count=1))
    if mean_columns:
        weekly_parts.append(data[mean_columns].resample("W-SUN").mean())
    if last_columns:
        weekly_parts.append(data[last_columns].resample("W-SUN").last())

    normalized_columns = [
        column
        for column in data.columns
        if column.endswith("_per_100k") or column.endswith("_pct_population")
    ]
    if normalized_columns:
        weekly_parts.append(data[normalized_columns].resample("W-SUN").mean())

    weekly_df = pd.concat(weekly_parts, axis=1) if weekly_parts else pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

    if "location" in data.columns:
        weekly_df["location"] = data["location"].resample("W-SUN").last()

    weekly_df = weekly_df.dropna(how="all")
    weekly_df.index.name = "date"
    return weekly_df


def _plot_study_window_selection(
    daily_df: pd.DataFrame,
    window: StudyWindowResult,
    country: str,
    output_path: Path,
) -> None:
    """Plot row coverage and selected window overlay."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(window.row_coverage.index, window.row_coverage.values, color="tab:blue", label="Row coverage")
    axes[0].axhline(window.min_row_coverage, color="tab:red", linestyle="--", label="Coverage threshold")
    axes[0].axvspan(window.start_date, window.end_date, alpha=0.15, color="tab:green", label="Selected window")
    axes[0].set_title(f"Study Window Selection - {country}")
    axes[0].set_ylabel("Coverage ratio")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    plotted = False
    if "new_cases_7d_avg" in daily_df.columns:
        axes[1].plot(daily_df.index, daily_df["new_cases_7d_avg"], color="tab:blue", label="new_cases_7d_avg")
        plotted = True
    elif "new_cases" in daily_df.columns:
        axes[1].plot(daily_df.index, daily_df["new_cases"], color="tab:blue", alpha=0.4, label="new_cases")
        plotted = True

    if "new_deaths_7d_avg" in daily_df.columns:
        axes[1].plot(daily_df.index, daily_df["new_deaths_7d_avg"], color="tab:red", label="new_deaths_7d_avg")
        plotted = True
    elif "new_deaths" in daily_df.columns:
        axes[1].plot(daily_df.index, daily_df["new_deaths"], color="tab:red", alpha=0.4, label="new_deaths")
        plotted = True

    axes[1].axvspan(window.start_date, window.end_date, alpha=0.15, color="tab:green", label="Selected window")
    axes[1].set_title("Signal overview with selected window")
    axes[1].set_ylabel("Daily values")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.2)
    if plotted:
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    LOGGER.info("Saved figure: %s", output_path)


def _plot_missingness_heatmap(
    daily_df: pd.DataFrame,
    columns: list[str],
    country: str,
    output_path: Path,
) -> None:
    """Plot missingness heatmap for key columns."""
    available = [column for column in columns if column in daily_df.columns]
    if not available:
        LOGGER.warning("Skipping missingness heatmap: no key columns available")
        return

    matrix = daily_df[available].isna().transpose().astype(int).values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(13, 4 + 0.25 * len(available)))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=1)

    axis.set_title(f"Missingness Heatmap (1 = missing) - {country}")
    axis.set_ylabel("Variable")
    axis.set_xlabel("Date")
    axis.set_yticks(np.arange(len(available)))
    axis.set_yticklabels(available)

    if len(daily_df.index) > 1:
        tick_count = min(8, len(daily_df.index))
        tick_positions = np.linspace(0, len(daily_df.index) - 1, tick_count, dtype=int)
        tick_labels = [daily_df.index[position].strftime("%Y-%m-%d") for position in tick_positions]
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(tick_labels, rotation=45, ha="right")

    cbar = fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    cbar.set_label("Missing indicator")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    LOGGER.info("Saved figure: %s", output_path)


def _plot_daily_vs_weekly_cases(
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    country: str,
    output_path: Path,
) -> None:
    """Plot daily and weekly case dynamics for granularity diagnostics."""
    if "new_cases" not in daily_df.columns and "new_cases_7d_avg" not in daily_df.columns:
        LOGGER.warning("Skipping daily vs weekly cases plot: cases columns unavailable")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(13, 5))

    if "new_cases_7d_avg" in daily_df.columns:
        axis.plot(daily_df.index, daily_df["new_cases_7d_avg"], color="tab:blue", label="Daily 7d avg cases")
    elif "new_cases" in daily_df.columns:
        axis.plot(daily_df.index, daily_df["new_cases"], color="tab:blue", alpha=0.5, label="Daily cases")

    if "new_cases" in weekly_df.columns:
        axis.bar(weekly_df.index, weekly_df["new_cases"], width=5, alpha=0.25, color="tab:orange", label="Weekly sum cases")

    axis.set_title(f"Daily vs Weekly Cases - {country}")
    axis.set_ylabel("Cases")
    axis.set_xlabel("Date")
    axis.grid(alpha=0.2)
    axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    LOGGER.info("Saved figure: %s", output_path)


def _plot_normalized_trends(daily_df: pd.DataFrame, country: str, output_path: Path) -> None:
    """Plot normalized cases/deaths trends when available."""
    required = ["new_cases_per_100k", "new_deaths_per_100k"]
    available = [column for column in required if column in daily_df.columns]
    if not available:
        LOGGER.warning("Skipping normalized trends plot: per-100k columns unavailable")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(13, 5))

    if "new_cases_per_100k" in daily_df.columns:
        axis.plot(daily_df.index, daily_df["new_cases_per_100k"], color="tab:blue", label="new_cases_per_100k")
    if "new_deaths_per_100k" in daily_df.columns:
        axis.plot(daily_df.index, daily_df["new_deaths_per_100k"], color="tab:red", label="new_deaths_per_100k")

    axis.set_title(f"Normalized Daily Trends (per 100k) - {country}")
    axis.set_ylabel("Per 100k people")
    axis.set_xlabel("Date")
    axis.grid(alpha=0.2)
    axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    LOGGER.info("Saved figure: %s", output_path)


def _build_study_window_report_text(
    country: str,
    window: StudyWindowResult,
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
) -> str:
    """Build text report for study window and analysis-data preparation."""
    lines: list[str] = []
    lines.append("COVID Study Window and Analysis Preparation Report")
    lines.append("=" * 54)
    lines.append("")
    lines.append("1. Selected study window")
    lines.append(f"- Country: {country}")
    lines.append(f"- Base flow-bounded range: {window.base_start_date.date()} -> {window.base_end_date.date()}")
    lines.append(f"- Selected window: {window.start_date.date()} -> {window.end_date.date()}")
    lines.append(f"- Selected days: {window.selected_days}")
    lines.append(f"- Selection reason: {window.selection_reason}")
    lines.append(f"- Min row coverage threshold: {window.min_row_coverage}")
    lines.append(f"- Min window days target: {window.min_window_days}")
    lines.append("")

    lines.append("2. Coverage summary (selected window)")
    selected = daily_df.loc[window.start_date : window.end_date]
    if not selected.empty and "row_coverage" in selected.columns:
        lines.append(f"- Mean row coverage: {selected['row_coverage'].mean():.3f}")
        lines.append(f"- Median row coverage: {selected['row_coverage'].median():.3f}")
        lines.append(f"- Min row coverage: {selected['row_coverage'].min():.3f}")
        lines.append(f"- Max row coverage: {selected['row_coverage'].max():.3f}")
    else:
        lines.append("- Row coverage unavailable")
    lines.append("")

    lines.append("3. Candidate contiguous segments")
    if window.segments_df.empty:
        lines.append("- No candidate segment above coverage threshold")
    else:
        lines.append(window.segments_df.sort_values("length_days", ascending=False).head(10).to_string(index=False))
    lines.append("")

    lines.append("4. Output datasets")
    lines.append(f"- Daily rows (selected window): {len(selected)}")
    lines.append(f"- Weekly rows: {len(weekly_df)}")
    lines.append(f"- Added normalized columns: {len([c for c in daily_df.columns if c.endswith('_per_100k') or c.endswith('_pct_population')])}")
    lines.append("")

    return "\n".join(lines)


def run_analysis_data_preparation(
    *,
    country: str,
    processed_df: pd.DataFrame,
    output_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
    min_row_coverage: float = 0.5,
    min_window_days: int = 180,
    key_columns: list[str] | None = None,
    save_csv: bool = False,
) -> dict[str, Any]:
    """Run study window selection, normalization, weekly aggregation, and diagnostics outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    country_slug = _slugify_country(country)
    data = _ensure_datetime_index(processed_df)

    key_columns = key_columns or DEFAULT_STUDY_KEY_COLUMNS
    window = select_study_window(
        data,
        key_columns=key_columns,
        flow_columns=DEFAULT_FLOW_COLUMNS,
        min_row_coverage=min_row_coverage,
        min_window_days=min_window_days,
    )

    analysis_daily = data.loc[window.start_date : window.end_date].copy()
    analysis_daily["row_coverage"] = window.row_coverage.loc[analysis_daily.index]
    analysis_daily = add_population_normalized_variables(analysis_daily)
    analysis_weekly = build_weekly_dataset(analysis_daily)

    daily_output_path = output_dir / f"covid_{country_slug}_analysis_daily.parquet"
    weekly_output_path = output_dir / f"covid_{country_slug}_analysis_weekly.parquet"
    analysis_daily.to_parquet(daily_output_path)
    analysis_weekly.to_parquet(weekly_output_path)

    daily_csv_path: Path | None = None
    weekly_csv_path: Path | None = None
    if save_csv:
        daily_csv_path = daily_output_path.with_suffix(".csv")
        weekly_csv_path = weekly_output_path.with_suffix(".csv")
        analysis_daily.to_csv(daily_csv_path, index=True)
        analysis_weekly.to_csv(weekly_csv_path, index=True)

    segments_path = reports_dir / f"covid_{country_slug}_study_window_segments.csv"
    coverage_path = reports_dir / f"covid_{country_slug}_study_window_daily_coverage.csv"
    report_path = reports_dir / f"covid_{country_slug}_study_window_report.txt"

    window.segments_df.to_csv(segments_path, index=False)
    window.row_coverage.to_frame().to_csv(coverage_path, index=True)

    report_text = _build_study_window_report_text(
        country=country,
        window=window,
        daily_df=analysis_daily,
        weekly_df=analysis_weekly,
    )
    report_path.write_text(report_text, encoding="utf-8")
    print("\n=== Study Window Report ===")
    print(report_text)

    window_plot_path = figures_dir / f"covid_{country_slug}_study_window_selection.png"
    missingness_plot_path = figures_dir / f"covid_{country_slug}_missingness_heatmap.png"
    daily_weekly_plot_path = figures_dir / f"covid_{country_slug}_daily_vs_weekly_cases.png"
    normalized_plot_path = figures_dir / f"covid_{country_slug}_normalized_trends.png"

    _plot_study_window_selection(data, window, country, window_plot_path)
    _plot_missingness_heatmap(data.loc[window.base_start_date : window.base_end_date], key_columns, country, missingness_plot_path)
    _plot_daily_vs_weekly_cases(analysis_daily, analysis_weekly, country, daily_weekly_plot_path)
    _plot_normalized_trends(analysis_daily, country, normalized_plot_path)

    LOGGER.info("Study-window daily dataset saved: %s", daily_output_path)
    LOGGER.info("Study-window weekly dataset saved: %s", weekly_output_path)
    LOGGER.info("Study-window report saved: %s", report_path)

    return {
        "window": window,
        "analysis_daily": analysis_daily,
        "analysis_weekly": analysis_weekly,
        "daily_output_path": daily_output_path,
        "weekly_output_path": weekly_output_path,
        "daily_csv_path": daily_csv_path,
        "weekly_csv_path": weekly_csv_path,
        "report_path": report_path,
        "report_text": report_text,
        "segments_path": segments_path,
        "coverage_path": coverage_path,
        "window_plot_path": window_plot_path,
        "missingness_plot_path": missingness_plot_path,
        "daily_weekly_plot_path": daily_weekly_plot_path,
        "normalized_plot_path": normalized_plot_path,
    }


def _parse_key_columns(raw_value: str | None) -> list[str] | None:
    if raw_value is None or not raw_value.strip():
        return None
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for independent analysis-data preparation run."""
    parser = argparse.ArgumentParser(description="Prepare analysis-ready COVID dataset")
    parser.add_argument("--country", type=str, default="France", help="Country label for reports")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/covid_france.parquet"),
        help="Input processed dataset path (parquet or CSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/analysis"),
        help="Directory for analysis datasets",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/processed/reports"),
        help="Directory for study-window reports",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures/analysis"),
        help="Directory for study-window diagnostic figures",
    )
    parser.add_argument(
        "--min-row-coverage",
        type=float,
        default=0.5,
        help="Minimum daily row coverage ratio required for candidate segments",
    )
    parser.add_argument(
        "--min-window-days",
        type=int,
        default=180,
        help="Minimum target length for selected study window",
    )
    parser.add_argument(
        "--key-columns",
        type=str,
        default=None,
        help="Optional comma-separated key columns override",
    )
    parser.add_argument("--save-csv", action="store_true", help="Also save analysis datasets as CSV")
    return parser.parse_args()


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    input_df = _load_dataset(args.input_path)
    run_analysis_data_preparation(
        country=args.country,
        processed_df=input_df,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        min_row_coverage=args.min_row_coverage,
        min_window_days=args.min_window_days,
        key_columns=_parse_key_columns(args.key_columns),
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
