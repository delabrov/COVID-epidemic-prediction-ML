"""Generate exploratory epidemiological plots from processed data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_processed_data(input_path: Path) -> pd.DataFrame:
    """Load processed dataset (parquet or CSV fallback)."""
    if not input_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        dataframe = pd.read_parquet(input_path)
    elif input_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    if "date" in dataframe.columns:
        dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
        dataframe = dataframe.dropna(subset=["date"]).set_index("date")
    else:
        dataframe.index = pd.to_datetime(dataframe.index, errors="coerce")
        dataframe = dataframe[~dataframe.index.isna()]

    dataframe = dataframe.sort_index()
    return dataframe


def _finalize_plot(title: str, ylabel: str) -> None:
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()


def _save_current_figure(output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=140)
    plt.close()
    LOGGER.info("Saved figure: %s", output_path)


def plot_time_series(df: pd.DataFrame, output_dir: Path, country: str) -> None:
    """Plot daily and smoothed time series for cases and deaths."""
    if {"new_cases", "new_cases_7d_avg"}.issubset(df.columns):
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df["new_cases"], label="New Cases (daily)", alpha=0.4)
        plt.plot(df.index, df["new_cases_7d_avg"], label="New Cases (7d avg)", linewidth=2)
        _finalize_plot(f"COVID-19 Cases - {country}", "Cases")
        _save_current_figure(output_dir, "new_cases_timeseries.png")
    else:
        LOGGER.warning("Skipping cases plot: required columns not found")

    if {"new_deaths", "new_deaths_7d_avg"}.issubset(df.columns):
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df["new_deaths"], label="New Deaths (daily)", alpha=0.4)
        plt.plot(df.index, df["new_deaths_7d_avg"], label="New Deaths (7d avg)", linewidth=2)
        _finalize_plot(f"COVID-19 Deaths - {country}", "Deaths")
        _save_current_figure(output_dir, "new_deaths_timeseries.png")
    else:
        LOGGER.warning("Skipping deaths plot: required columns not found")


def plot_cumulative(df: pd.DataFrame, output_dir: Path, country: str) -> None:
    """Plot cumulative and daily (with rolling mean) cases/deaths in one combined figure."""
    has_cases = "total_cases" in df.columns
    has_deaths = "total_deaths" in df.columns
    has_new_cases = "new_cases" in df.columns
    has_new_deaths = "new_deaths" in df.columns
    has_new_cases_avg = "new_cases_7d_avg" in df.columns
    has_new_deaths_avg = "new_deaths_7d_avg" in df.columns

    if not any([has_cases, has_deaths, has_new_cases, has_new_deaths, has_new_cases_avg, has_new_deaths_avg]):
        LOGGER.warning("Skipping combined metrics plot: required columns not found")
        return

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(14, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.0]},
    )

    if has_cases:
        axes[0].plot(df.index, df["total_cases"], color="tab:blue", label="Total Cases")
        axes[0].set_title(f"Cumulative Cases - {country}")
        axes[0].set_ylabel("Cases")
        axes[0].grid(alpha=0.2)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "total_cases not available", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title(f"Cumulative Cases - {country}")
        axes[0].grid(alpha=0.2)

    if has_new_cases:
        axes[1].bar(df.index, df["new_cases"], color="tab:blue", alpha=0.3, label="New Cases (daily)")
    if has_new_cases_avg:
        axes[1].plot(df.index, df["new_cases_7d_avg"], color="tab:blue", linewidth=2, label="New Cases (7d avg)")
    if has_new_cases or has_new_cases_avg:
        axes[1].set_title(f"Daily Cases - {country}")
        axes[1].set_ylabel("Cases / day")
        axes[1].grid(alpha=0.2)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "new_cases/new_cases_7d_avg not available", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title(f"Daily Cases - {country}")
        axes[1].grid(alpha=0.2)

    if has_deaths:
        axes[2].plot(df.index, df["total_deaths"], color="tab:red", label="Total Deaths")
        axes[2].set_title(f"Cumulative Deaths - {country}")
        axes[2].set_ylabel("Deaths")
        axes[2].grid(alpha=0.2)
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "total_deaths not available", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title(f"Cumulative Deaths - {country}")
        axes[2].grid(alpha=0.2)

    if has_new_deaths:
        axes[3].bar(df.index, df["new_deaths"], color="tab:red", alpha=0.3, label="New Deaths (daily)")
    if has_new_deaths_avg:
        axes[3].plot(df.index, df["new_deaths_7d_avg"], color="tab:red", linewidth=2, label="New Deaths (7d avg)")
    if has_new_deaths or has_new_deaths_avg:
        axes[3].set_title(f"Daily Deaths - {country}")
        axes[3].set_ylabel("Deaths / day")
        axes[3].grid(alpha=0.2)
        axes[3].legend()
    else:
        axes[3].text(0.5, 0.5, "new_deaths/new_deaths_7d_avg not available", ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_title(f"Daily Deaths - {country}")
        axes[3].grid(alpha=0.2)

    axes[3].set_xlabel("Date")
    figure.tight_layout()
    _save_current_figure(output_dir, "cumulative_metrics.png")


def plot_vaccination(df: pd.DataFrame, output_dir: Path, country: str) -> None:
    """Plot vaccination progression if columns are available."""
    vaccination_columns = [
        column
        for column in ["people_vaccinated", "people_fully_vaccinated", "total_boosters"]
        if column in df.columns
    ]
    if not vaccination_columns:
        LOGGER.info("Skipping vaccination plot: no vaccination columns found")
        return

    plt.figure(figsize=(12, 5))
    for column in vaccination_columns:
        plt.plot(df.index, df[column], label=column)
    _finalize_plot(f"Vaccination Curves - {country}", "People")
    _save_current_figure(output_dir, "vaccination_curves.png")


def plot_stringency_vs_cases(df: pd.DataFrame, output_dir: Path, country: str) -> None:
    """Plot stringency index against smoothed cases with dual y-axis."""
    required = {"stringency_index", "new_cases_7d_avg"}
    if not required.issubset(df.columns):
        LOGGER.info("Skipping stringency plot: missing required columns")
        return

    fig, axis_cases = plt.subplots(figsize=(12, 5))
    axis_stringency = axis_cases.twinx()

    line_cases = axis_cases.plot(df.index, df["new_cases_7d_avg"], label="new_cases_7d_avg", color="tab:blue")
    line_stringency = axis_stringency.plot(
        df.index,
        df["stringency_index"],
        label="stringency_index",
        color="tab:orange",
    )

    axis_cases.set_title(f"Stringency Index vs Cases - {country}")
    axis_cases.set_xlabel("Date")
    axis_cases.set_ylabel("Cases (7d avg)", color="tab:blue")
    axis_stringency.set_ylabel("Stringency Index", color="tab:orange")
    axis_cases.grid(alpha=0.2)

    lines = line_cases + line_stringency
    labels = [line.get_label() for line in lines]
    axis_cases.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    _save_current_figure(output_dir, "stringency_vs_cases.png")


def generate_all_plots(df: pd.DataFrame, output_dir: Path, country: str) -> None:
    """Generate all exploratory plots."""
    plot_cumulative(df, output_dir, country)
    plot_vaccination(df, output_dir, country)
    plot_stringency_vs_cases(df, output_dir, country)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for plotting."""
    parser = argparse.ArgumentParser(description="Generate exploratory COVID-19 plots")
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to processed parquet or CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Figure output directory",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="",
        help="Country label used in plot titles",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    dataframe = load_processed_data(args.input_path)
    inferred_country = args.country
    if not inferred_country and "location" in dataframe.columns:
        non_null_locations = dataframe["location"].dropna()
        if not non_null_locations.empty:
            inferred_country = str(non_null_locations.iloc[0])

    generate_all_plots(dataframe, args.output_dir, country=inferred_country or "Selected Country")


if __name__ == "__main__":
    main()
