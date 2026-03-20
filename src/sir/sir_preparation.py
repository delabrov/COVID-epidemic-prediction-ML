"""SIR preparation module (steps 1-3 from project idea).

Covered steps:
1. Define SIR modeling framework and assumptions.
2. Prepare SIR-ready study dataset.
3. Estimate initial S, I, R states.
"""

from __future__ import annotations

import argparse
import json
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

DEFAULT_START_DATE = "2020-01-05"
DEFAULT_END_DATE = "2023-07-01"
DEFAULT_SIGNAL_COLUMN = "new_cases_7d_avg"
DEFAULT_POPULATION_COLUMN = "population"


@dataclass
class SIRInitialConditions:
    """Initial conditions for SIR model."""

    date: pd.Timestamp
    population: float
    s0: float
    i0: float
    r0: float


def _slugify_country(country: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure input dataframe has sorted DatetimeIndex named 'date'."""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).set_index("date")
    else:
        raise ValueError("Input dataframe must have DatetimeIndex or a 'date' column")

    out = out[~out.index.isna()].sort_index()
    out.index.name = "date"
    return out


def load_input_dataset(input_path: Path) -> pd.DataFrame:
    """Load processed country dataset (parquet or CSV)."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file extension: {input_path.suffix}")
    return _ensure_datetime_index(df)


def define_sir_framework(
    *,
    country: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    signal_column: str,
    infectious_period_days: int,
) -> dict[str, Any]:
    """Define baseline SIR framework metadata for reproducibility."""
    return {
        "country": country,
        "time_window": {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "time_step": "1 day",
        },
        "sir_equations": {
            "dS_dt": "-beta * S * I / N",
            "dI_dt": "beta * S * I / N - gamma * I",
            "dR_dt": "gamma * I",
        },
        "observation_mapping": {
            "incidence_signal": signal_column,
            "note": "Incidence signal used as observed epidemic dynamics for calibration",
        },
        "assumptions": [
            "Population N is approximately constant on the selected window",
            "Signal uses 7-day smoothed new cases to reduce reporting noise",
            "Daily time step is used for SIR discretization",
            "I_t is approximated from recent incidence over infectious period",
            "R_t is approximated as cumulative infections minus active infections",
        ],
        "initialization_hyperparameters": {
            "infectious_period_days": infectious_period_days,
        },
    }


def _ensure_signal_column(df: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    """Ensure chosen signal exists; compute from new_cases if possible."""
    out = df.copy()
    if signal_column in out.columns:
        return out

    if signal_column == "new_cases_7d_avg" and "new_cases" in out.columns:
        out[signal_column] = out["new_cases"].rolling(window=7, min_periods=1).mean()
        LOGGER.warning("Signal column '%s' was missing and has been computed from 'new_cases'", signal_column)
        return out

    raise ValueError(f"Required signal column not found: {signal_column}")


def _add_per_100k_columns(df: pd.DataFrame, population_column: str) -> pd.DataFrame:
    """Add /100k normalized columns for selected epidemiological variables."""
    out = df.copy()
    if population_column not in out.columns:
        raise ValueError(f"Population column not found: {population_column}")

    pop = out[population_column].astype(float)
    pop = pop.where(pop > 0)

    base_columns = [
        "new_cases",
        "new_deaths",
        "total_cases",
        "total_deaths",
        "new_vaccinations",
        "icu_patients",
        "hosp_patients",
        "new_cases_7d_avg",
        "new_deaths_7d_avg",
    ]

    for column in base_columns:
        if column in out.columns:
            out[f"{column}_per_100k"] = (out[column] / pop) * 100_000.0

    return out


def prepare_sir_study_dataset(
    df: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    signal_column: str,
    population_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare SIR-ready dataframe on fixed study window with core checks."""
    out = _ensure_datetime_index(df)
    out = _ensure_signal_column(out, signal_column)

    window_df = out.loc[start_date:end_date].copy()
    if window_df.empty:
        raise ValueError("Selected time window is empty after filtering")

    if population_column not in window_df.columns:
        raise ValueError(f"Population column not found: {population_column}")

    if window_df[population_column].isna().all():
        raise ValueError("Population column is fully missing on selected window")

    window_df[population_column] = window_df[population_column].ffill().bfill()
    window_df = _add_per_100k_columns(window_df, population_column)

    signal_missing_count = int(window_df[signal_column].isna().sum())
    if signal_missing_count > 0:
        LOGGER.warning(
            "Signal '%s' contains %s missing values inside study window; interpolating linearly",
            signal_column,
            signal_missing_count,
        )
        window_df[signal_column] = window_df[signal_column].interpolate(method="time").ffill().bfill()

    report = {
        "start_date": window_df.index.min().strftime("%Y-%m-%d"),
        "end_date": window_df.index.max().strftime("%Y-%m-%d"),
        "rows": int(len(window_df)),
        "columns": int(len(window_df.columns)),
        "signal_column": signal_column,
        "signal_missing_after_fill": int(window_df[signal_column].isna().sum()),
        "population_column": population_column,
    }
    return window_df, report


def estimate_initial_sir_states(
    df: pd.DataFrame,
    *,
    signal_column: str,
    population_column: str,
    infectious_period_days: int,
) -> tuple[pd.DataFrame, SIRInitialConditions, dict[str, Any]]:
    """Estimate S, I, R trajectories and initial conditions.

    Estimation rules:
    - I_t ~= rolling sum of incidence over infectious_period_days.
    - cumulative infections ~= cumulative sum of incidence.
    - R_t ~= cumulative infections - I_t.
    - S_t = N - I_t - R_t.
    """
    if infectious_period_days <= 0:
        raise ValueError("infectious_period_days must be > 0")

    out = df.copy()
    incidence = out[signal_column].astype(float).clip(lower=0)
    population = out[population_column].astype(float)
    out["population_for_sir"] = population

    out["incidence_for_sir"] = incidence
    out["I_estimated"] = incidence.rolling(window=infectious_period_days, min_periods=1).sum()
    out["cumulative_infections_estimated"] = incidence.cumsum()
    out["R_estimated"] = (out["cumulative_infections_estimated"] - out["I_estimated"]).clip(lower=0)

    # Keep S non-negative by capping total infected+removed at N.
    total_ir = out["I_estimated"] + out["R_estimated"]
    overflow = (total_ir - population).clip(lower=0)
    if float(overflow.max()) > 0:
        out["R_estimated"] = (out["R_estimated"] - overflow).clip(lower=0)

    out["S_estimated"] = (population - out["I_estimated"] - out["R_estimated"]).clip(lower=0)

    first_date = out.index.min()
    initial = SIRInitialConditions(
        date=first_date,
        population=float(population.loc[first_date]),
        s0=float(out.loc[first_date, "S_estimated"]),
        i0=float(out.loc[first_date, "I_estimated"]),
        r0=float(out.loc[first_date, "R_estimated"]),
    )

    summary = {
        "infectious_period_days": infectious_period_days,
        "initial_conditions": {
            "date": initial.date.strftime("%Y-%m-%d"),
            "N": initial.population,
            "S0": initial.s0,
            "I0": initial.i0,
            "R0": initial.r0,
            "S0_plus_I0_plus_R0": initial.s0 + initial.i0 + initial.r0,
        },
    }
    return out, initial, summary


def _plot_sir_signal(dataset: pd.DataFrame, signal_column: str, country: str, output_path: Path) -> None:
    """Plot SIR signal in absolute and per-100k scales."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(dataset.index, dataset[signal_column], color="tab:blue", label=signal_column)
    axes[0].set_title(f"SIR Modeling Signal - {country}")
    axes[0].set_ylabel("Cases/day (smoothed)")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    per_100k_col = f"{signal_column}_per_100k"
    if per_100k_col in dataset.columns:
        axes[1].plot(dataset.index, dataset[per_100k_col], color="tab:orange", label=per_100k_col)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, f"{per_100k_col} unavailable", ha="center", va="center", transform=axes[1].transAxes)

    axes[1].set_title("Signal normalized per 100k")
    axes[1].set_ylabel("Cases/day per 100k")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    LOGGER.info("Saved figure: %s", output_path)


def _plot_sir_states(dataset: pd.DataFrame, country: str, output_path: Path) -> None:
    """Plot estimated S, I, R trajectories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(dataset.index, dataset["S_estimated"], label="S_estimated", color="tab:green")
    axes[0].plot(dataset.index, dataset["I_estimated"], label="I_estimated", color="tab:blue")
    axes[0].plot(dataset.index, dataset["R_estimated"], label="R_estimated", color="tab:red")
    axes[0].set_title(f"Estimated SIR States - {country}")
    axes[0].set_ylabel("People")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    if "population_for_sir" in dataset.columns:
        population = dataset["population_for_sir"]
    elif DEFAULT_POPULATION_COLUMN in dataset.columns:
        population = dataset[DEFAULT_POPULATION_COLUMN]
    else:
        raise ValueError("Population column not available for normalized SIR state plot")
    axes[1].plot(dataset.index, dataset["S_estimated"] / population, label="S/N", color="tab:green")
    axes[1].plot(dataset.index, dataset["I_estimated"] / population, label="I/N", color="tab:blue")
    axes[1].plot(dataset.index, dataset["R_estimated"] / population, label="R/N", color="tab:red")
    axes[1].set_title("Estimated normalized states")
    axes[1].set_ylabel("Fraction of population")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    LOGGER.info("Saved figure: %s", output_path)


def _build_sir_preparation_report(
    framework: dict[str, Any],
    prep_summary: dict[str, Any],
    init_summary: dict[str, Any],
    signal_column: str,
) -> str:
    lines: list[str] = []
    lines.append("SIR Preparation Report")
    lines.append("=" * 30)
    lines.append("")

    lines.append("1) Framework definition")
    lines.append(json.dumps(framework, indent=2))
    lines.append("")

    lines.append("2) SIR-ready dataset preparation")
    for key, value in prep_summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("3) Initial state estimation")
    lines.append(f"- incidence signal used: {signal_column}")
    lines.append(json.dumps(init_summary, indent=2))
    lines.append("")

    return "\n".join(lines)


def run_sir_preparation_pipeline(
    *,
    country: str,
    input_path: Path,
    output_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    signal_column: str = DEFAULT_SIGNAL_COLUMN,
    population_column: str = DEFAULT_POPULATION_COLUMN,
    infectious_period_days: int = 14,
    save_csv: bool = False,
) -> dict[str, Any]:
    """Run full SIR preparation pipeline (steps 1-3)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    country_slug = _slugify_country(country)
    parsed_start = pd.Timestamp(start_date)
    parsed_end = pd.Timestamp(end_date)

    input_df = load_input_dataset(input_path)

    framework = define_sir_framework(
        country=country,
        start_date=parsed_start,
        end_date=parsed_end,
        signal_column=signal_column,
        infectious_period_days=infectious_period_days,
    )

    sir_dataset, prep_summary = prepare_sir_study_dataset(
        input_df,
        start_date=parsed_start,
        end_date=parsed_end,
        signal_column=signal_column,
        population_column=population_column,
    )

    sir_dataset, initial_conditions, init_summary = estimate_initial_sir_states(
        sir_dataset,
        signal_column=signal_column,
        population_column=population_column,
        infectious_period_days=infectious_period_days,
    )

    sir_data_path = output_dir / f"covid_{country_slug}_sir_prepared.parquet"
    sir_dataset.to_parquet(sir_data_path)

    sir_csv_path: Path | None = None
    if save_csv:
        sir_csv_path = sir_data_path.with_suffix(".csv")
        sir_dataset.to_csv(sir_csv_path, index=True)

    framework_path = reports_dir / f"covid_{country_slug}_sir_framework.json"
    initial_conditions_path = reports_dir / f"covid_{country_slug}_sir_initial_conditions.json"
    preparation_report_path = reports_dir / f"covid_{country_slug}_sir_preparation_report.txt"

    framework_path.write_text(json.dumps(framework, indent=2), encoding="utf-8")
    initial_conditions_path.write_text(json.dumps(init_summary, indent=2), encoding="utf-8")

    report_text = _build_sir_preparation_report(
        framework=framework,
        prep_summary=prep_summary,
        init_summary=init_summary,
        signal_column=signal_column,
    )
    preparation_report_path.write_text(report_text, encoding="utf-8")

    signal_plot_path = figures_dir / f"covid_{country_slug}_sir_signal.png"
    states_plot_path = figures_dir / f"covid_{country_slug}_sir_states.png"

    _plot_sir_signal(sir_dataset, signal_column, country, signal_plot_path)
    _plot_sir_states(sir_dataset, country, states_plot_path)

    print("\n=== SIR Preparation Summary ===")
    print(report_text)

    LOGGER.info("SIR prepared dataset saved: %s", sir_data_path)
    LOGGER.info("SIR framework saved: %s", framework_path)
    LOGGER.info("SIR initial conditions saved: %s", initial_conditions_path)
    LOGGER.info("SIR preparation report saved: %s", preparation_report_path)

    return {
        "framework": framework,
        "prep_summary": prep_summary,
        "initial_conditions": initial_conditions,
        "initial_summary": init_summary,
        "sir_dataset": sir_dataset,
        "sir_data_path": sir_data_path,
        "sir_csv_path": sir_csv_path,
        "framework_path": framework_path,
        "initial_conditions_path": initial_conditions_path,
        "preparation_report_path": preparation_report_path,
        "signal_plot_path": signal_plot_path,
        "states_plot_path": states_plot_path,
        "report_text": report_text,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for independent SIR preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare SIR modeling inputs (steps 1-3)")
    parser.add_argument("--country", type=str, default="France", help="Country label")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/analysis/covid_france_analysis_daily.parquet"),
        help="Input prepared dataset path (parquet or CSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sir/inputs"),
        help="Directory for SIR prepared datasets",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/sir/reports"),
        help="Directory for SIR reports",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures/sir"),
        help="Directory for SIR diagnostic figures",
    )
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE, help="Study window start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE, help="Study window end date (YYYY-MM-DD)")
    parser.add_argument(
        "--signal-column",
        type=str,
        default=DEFAULT_SIGNAL_COLUMN,
        help="Incidence signal used for SIR preparation",
    )
    parser.add_argument(
        "--population-column",
        type=str,
        default=DEFAULT_POPULATION_COLUMN,
        help="Population column name",
    )
    parser.add_argument(
        "--infectious-period-days",
        type=int,
        default=14,
        help="Infectious period (days) used to estimate active infections",
    )
    parser.add_argument("--save-csv", action="store_true", help="Also save prepared SIR dataset as CSV")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    run_sir_preparation_pipeline(
        country=args.country,
        input_path=args.input_path,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        signal_column=args.signal_column,
        population_column=args.population_column,
        infectious_period_days=args.infectious_period_days,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
