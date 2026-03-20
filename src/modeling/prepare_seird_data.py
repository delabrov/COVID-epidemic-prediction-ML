"""Prepare SEIRD-ready dataset by reconstructing SEIRD state trajectories."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_START_DATE = "2020-01-05"
DEFAULT_END_DATE = "2023-07-01"
DEFAULT_CASES_SIGNAL = "new_cases_7d_avg"
DEFAULT_DEATHS_SIGNAL = "new_deaths_7d_avg"
DEFAULT_POPULATION_COLUMN = "population"


def _slugify_country(country: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe has sorted DatetimeIndex named 'date'."""
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
    """Load processed/analysis dataset from parquet or CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input extension: {input_path.suffix}")
    return _ensure_datetime_index(df)


def _ensure_signal_columns(df: pd.DataFrame, cases_signal: str, deaths_signal: str) -> pd.DataFrame:
    """Ensure cases/deaths signals exist, with safe fallback from daily columns."""
    out = df.copy()
    if cases_signal not in out.columns:
        if cases_signal == "new_cases_7d_avg" and "new_cases" in out.columns:
            out[cases_signal] = out["new_cases"].rolling(window=7, min_periods=1).mean()
            LOGGER.warning("Cases signal '%s' was missing and computed from 'new_cases'", cases_signal)
        else:
            raise ValueError(f"Missing required cases signal column: {cases_signal}")

    if deaths_signal not in out.columns:
        if deaths_signal == "new_deaths_7d_avg" and "new_deaths" in out.columns:
            out[deaths_signal] = out["new_deaths"].rolling(window=7, min_periods=1).mean()
            LOGGER.warning("Deaths signal '%s' was missing and computed from 'new_deaths'", deaths_signal)
        else:
            raise ValueError(f"Missing required deaths signal column: {deaths_signal}")
    return out


def define_seird_framework(
    *,
    country: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cases_signal: str,
    deaths_signal: str,
    latent_period_days: int,
    infectious_period_days: int,
) -> dict[str, Any]:
    """Define SEIRD framework metadata and assumptions."""
    return {
        "country": country,
        "time_window": {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "time_step": "1 day",
        },
        "seird_equations": {
            "dS_dt": "-beta * S * I / N",
            "dE_dt": "beta * S * I / N - sigma * E",
            "dI_dt": "sigma * E - gamma * I - mu * I",
            "dR_dt": "gamma * I",
            "dD_dt": "mu * I",
        },
        "observation_mapping": {
            "incidence_signal": cases_signal,
            "mortality_signal": deaths_signal,
        },
        "assumptions": [
            "Population N is approximately constant on selected window",
            "I_estimated is proxied by rolling incidence over infectious period",
            "E_estimated is proxied by lagged rolling incidence over latent period",
            "D_estimated uses total_deaths when available, otherwise cumulative smoothed deaths",
            "R_estimated is inferred from cumulative infections minus E, I, D",
            "S_estimated is residual compartment: N - E - I - R - D",
        ],
        "reconstruction_hyperparameters": {
            "latent_period_days": latent_period_days,
            "infectious_period_days": infectious_period_days,
        },
    }


def reconstruct_seird_states(
    df: pd.DataFrame,
    *,
    population_column: str,
    cases_signal: str,
    deaths_signal: str,
    latent_period_days: int,
    infectious_period_days: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct SEIRD states from observed smoothed signals."""
    if latent_period_days <= 0 or infectious_period_days <= 0:
        raise ValueError("latent_period_days and infectious_period_days must be > 0")

    out = df.copy()

    if population_column not in out.columns:
        raise ValueError(f"Population column missing: {population_column}")
    out[population_column] = out[population_column].astype(float).ffill().bfill()
    if out[population_column].isna().all():
        raise ValueError("Population column is fully missing after fill")

    incidence = out[cases_signal].astype(float).clip(lower=0).interpolate(method="time").ffill().bfill()
    deaths_signal_series = out[deaths_signal].astype(float).clip(lower=0).interpolate(method="time").ffill().bfill()

    i_est = incidence.rolling(window=infectious_period_days, min_periods=1).sum()
    e_base = incidence.rolling(window=latent_period_days, min_periods=1).sum()
    e_est = e_base.shift(1).fillna(e_base)

    if "total_deaths" in out.columns and out["total_deaths"].notna().any():
        d_est = out["total_deaths"].astype(float).clip(lower=0).interpolate(method="time").ffill().bfill()
    else:
        d_est = deaths_signal_series.cumsum()

    cumulative_infections = incidence.cumsum()
    r_est = (cumulative_infections - i_est - e_est - d_est).clip(lower=0)

    population = out[population_column].astype(float)
    seird_sum = e_est + i_est + r_est + d_est
    overflow = (seird_sum - population).clip(lower=0)
    if float(overflow.max()) > 0:
        r_est = (r_est - overflow).clip(lower=0)

    s_est = (population - e_est - i_est - r_est - d_est).clip(lower=0)

    out["incidence_for_seird"] = incidence
    out["deaths_for_seird"] = deaths_signal_series
    out["cumulative_infections_estimated"] = cumulative_infections
    out["S_estimated"] = s_est
    out["E_estimated"] = e_est
    out["I_estimated"] = i_est
    out["R_estimated"] = r_est
    out["D_estimated"] = d_est

    first_date = out.index.min()
    init = {
        "date": first_date.strftime("%Y-%m-%d"),
        "N": float(population.loc[first_date]),
        "S0": float(s_est.loc[first_date]),
        "E0": float(e_est.loc[first_date]),
        "I0": float(i_est.loc[first_date]),
        "R0": float(r_est.loc[first_date]),
        "D0": float(d_est.loc[first_date]),
    }

    summary = {
        "rows": int(len(out)),
        "start_date": out.index.min().strftime("%Y-%m-%d"),
        "end_date": out.index.max().strftime("%Y-%m-%d"),
        "cases_signal": cases_signal,
        "deaths_signal": deaths_signal,
        "latent_period_days": latent_period_days,
        "infectious_period_days": infectious_period_days,
        "initial_states": init,
    }
    return out, summary


def _build_preparation_report_text(
    *,
    framework: dict[str, Any],
    summary: dict[str, Any],
    population_column: str,
) -> str:
    lines: list[str] = []
    lines.append("SEIRD Preparation Report")
    lines.append("=" * 28)
    lines.append("")
    lines.append("1. Framework definition")
    lines.append(json.dumps(framework, indent=2))
    lines.append("")
    lines.append("2. State reconstruction summary")
    lines.append(f"- population_column: {population_column}")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("3. Reconstruction variable definitions")
    lines.append("- I_estimated: rolling sum of incidence signal over infectious_period_days")
    lines.append("- E_estimated: lagged rolling sum of incidence signal over latent_period_days")
    lines.append("- D_estimated: total_deaths if available else cumulative smoothed deaths signal")
    lines.append("- R_estimated: cumulative infections - I_estimated - E_estimated - D_estimated")
    lines.append("- S_estimated: N - E_estimated - I_estimated - R_estimated - D_estimated")
    lines.append("")
    return "\n".join(lines)


def run_seird_preparation_pipeline(
    *,
    country: str,
    input_path: Path,
    output_dir: Path,
    reports_dir: Path,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    cases_signal: str = DEFAULT_CASES_SIGNAL,
    deaths_signal: str = DEFAULT_DEATHS_SIGNAL,
    population_column: str = DEFAULT_POPULATION_COLUMN,
    latent_period_days: int = 5,
    infectious_period_days: int = 14,
    save_csv: bool = False,
) -> dict[str, Any]:
    """Run SEIRD-ready dataset preparation pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    country_slug = _slugify_country(country)
    parsed_start = pd.Timestamp(start_date)
    parsed_end = pd.Timestamp(end_date)

    dataset = load_input_dataset(input_path)
    dataset = _ensure_signal_columns(dataset, cases_signal=cases_signal, deaths_signal=deaths_signal)
    dataset = dataset.loc[parsed_start:parsed_end].copy()
    if dataset.empty:
        raise ValueError("Selected SEIRD study window is empty after filtering")

    framework = define_seird_framework(
        country=country,
        start_date=parsed_start,
        end_date=parsed_end,
        cases_signal=cases_signal,
        deaths_signal=deaths_signal,
        latent_period_days=latent_period_days,
        infectious_period_days=infectious_period_days,
    )

    seird_ready_df, reconstruction_summary = reconstruct_seird_states(
        dataset,
        population_column=population_column,
        cases_signal=cases_signal,
        deaths_signal=deaths_signal,
        latent_period_days=latent_period_days,
        infectious_period_days=infectious_period_days,
    )

    ready_path = output_dir / f"covid_{country_slug}_seird_ready.parquet"
    seird_ready_df.to_parquet(ready_path)

    csv_path: Path | None = None
    if save_csv:
        csv_path = ready_path.with_suffix(".csv")
        seird_ready_df.to_csv(csv_path, index=True)

    report_text = _build_preparation_report_text(
        framework=framework,
        summary=reconstruction_summary,
        population_column=population_column,
    )
    report_path = reports_dir / f"covid_{country_slug}_seird_preparation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\n=== SEIRD Preparation Summary ===")
    print(report_text)

    LOGGER.info("SEIRD-ready dataset saved: %s", ready_path)
    LOGGER.info("SEIRD preparation report saved: %s", report_path)

    return {
        "dataset": seird_ready_df,
        "framework": framework,
        "reconstruction_summary": reconstruction_summary,
        "output_path": ready_path,
        "csv_output_path": csv_path,
        "report_path": report_path,
        "report_text": report_text,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for SEIRD preparation."""
    parser = argparse.ArgumentParser(description="Prepare SEIRD-ready dataset")
    parser.add_argument("--country", type=str, default="France", help="Country label")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/analysis/covid_france_analysis_daily.parquet"),
        help="Input dataset path (parquet/csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for SEIRD-ready dataset",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/processed/reports"),
        help="Directory for SEIRD preparation report",
    )
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE, help="Study window start date")
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE, help="Study window end date")
    parser.add_argument(
        "--cases-signal",
        type=str,
        default=DEFAULT_CASES_SIGNAL,
        help="Incidence signal column",
    )
    parser.add_argument(
        "--deaths-signal",
        type=str,
        default=DEFAULT_DEATHS_SIGNAL,
        help="Deaths signal column",
    )
    parser.add_argument(
        "--population-column",
        type=str,
        default=DEFAULT_POPULATION_COLUMN,
        help="Population column",
    )
    parser.add_argument("--latent-period-days", type=int, default=5, help="Latent period assumption")
    parser.add_argument("--infectious-period-days", type=int, default=14, help="Infectious period assumption")
    parser.add_argument("--save-csv", action="store_true", help="Also save SEIRD-ready dataset as CSV")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    run_seird_preparation_pipeline(
        country=args.country,
        input_path=args.input_path,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        cases_signal=args.cases_signal,
        deaths_signal=args.deaths_signal,
        population_column=args.population_column,
        latent_period_days=args.latent_period_days,
        infectious_period_days=args.infectious_period_days,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
