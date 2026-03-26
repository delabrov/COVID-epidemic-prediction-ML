"""Prepare SEIRDV-ready dataset by reconstructing SEIRDV state trajectories."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_START_DATE = "2020-01-05"
DEFAULT_END_DATE = "2023-07-01"
DEFAULT_CASES_SIGNAL = "new_cases_7d_avg"
DEFAULT_DEATHS_SIGNAL = "new_deaths_7d_avg"
DEFAULT_POPULATION_COLUMN = "population"
DEFAULT_INFECTIVITY_PROFILE = "gamma"
DEFAULT_INFECTIVITY_SHAPE = 3.0
DEFAULT_INFECTIVITY_SCALE = 2.0
DEFAULT_LATENT_PROFILE = "uniform"
DEFAULT_LATENT_SHAPE = 2.0
DEFAULT_LATENT_SCALE = 2.0
DEFAULT_VACCINE_EFFICACY = 0.6


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


def build_profile_weights(
    length: int,
    *,
    profile: str,
    shape: float,
    scale: float,
) -> np.ndarray:
    """Build nonnegative normalized discrete weights for convolution profiles."""
    if length <= 0:
        raise ValueError("Profile length must be > 0")
    profile_name = profile.lower().strip()
    lags = np.arange(length, dtype=float) + 1.0

    if profile_name == "uniform":
        weights = np.ones(length, dtype=float)
    elif profile_name == "gamma":
        if shape <= 0 or scale <= 0:
            raise ValueError("Gamma-like profile requires shape > 0 and scale > 0")
        weights = np.power(lags, float(shape) - 1.0) * np.exp(-lags / float(scale))
    else:
        raise ValueError(f"Unsupported profile: {profile}")

    weights = np.clip(weights, a_min=0.0, a_max=None)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Invalid profile weights: nonpositive sum")
    return weights / total


def _causal_convolution(signal: pd.Series, weights: np.ndarray) -> pd.Series:
    """Causal convolution aligned to the input index."""
    values = signal.astype(float).to_numpy()
    conv = np.convolve(values, weights, mode="full")[: len(values)]
    return pd.Series(conv, index=signal.index, dtype=float)


def _resolve_vaccination_series(df: pd.DataFrame, population: pd.Series) -> tuple[pd.Series, str]:
    """Resolve vaccination stock series with fallback logic and safe cleaning."""
    candidates = ["people_fully_vaccinated", "people_vaccinated"]
    selected_column: str | None = None
    for column in candidates:
        if column in df.columns and df[column].notna().any():
            selected_column = column
            break

    if selected_column is None:
        raise ValueError(
            "Missing vaccination stock columns: expected 'people_fully_vaccinated' "
            "or fallback 'people_vaccinated'"
        )

    vaccinated = df[selected_column].astype(float)
    vaccinated = vaccinated.clip(lower=0).interpolate(method="time").ffill().bfill()
    vaccinated = vaccinated.cummax().clip(upper=population)
    return vaccinated, selected_column


def define_seirdv_framework(
    *,
    country: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cases_signal: str,
    deaths_signal: str,
    latent_period_days: int,
    infectious_period_days: int,
    vaccine_efficacy: float,
    infectivity_profile: str,
    infectivity_shape: float,
    infectivity_scale: float,
    latent_profile: str,
    latent_shape: float,
    latent_scale: float,
) -> dict[str, Any]:
    """Define SEIRDV framework metadata and assumptions."""
    return {
        "country": country,
        "time_window": {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "time_step": "1 day",
        },
        "seirdv_equations": {
            "dS_dt": "-beta*S*I/N - nu*S",
            "dV_dt": "nu*S - (1-epsilon_v)*beta*V*I/N",
            "dE_dt": "beta*S*I/N + (1-epsilon_v)*beta*V*I/N - sigma*E",
            "dI_dt": "sigma*E - gamma*I - mu*I",
            "dR_dt": "gamma*I",
            "dD_dt": "mu*I",
        },
        "observation_mapping": {
            "incidence_signal": cases_signal,
            "mortality_signal": deaths_signal,
            "vaccination_stock_signal": "people_fully_vaccinated (fallback: people_vaccinated)",
        },
        "assumptions": [
            "Population N is approximately constant on selected window",
            "Vaccination compartment V is directly observation-driven from vaccination stock columns",
            "Vaccine efficacy epsilon_v is fixed as a model hyperparameter (not inferred)",
            "I_estimated is reconstructed by convolving observed cases with an infectivity profile",
            "E_estimated is reconstructed by convolving observed cases with a latent profile",
            "D_estimated uses total_deaths when available, otherwise cumulative smoothed deaths",
            "R_estimated is inferred from cumulative infections minus E, I, D",
            "S_estimated is residual compartment: N - V - E - I - R - D",
        ],
        "reconstruction_hyperparameters": {
            "latent_period_days": latent_period_days,
            "infectious_period_days": infectious_period_days,
            "vaccine_efficacy": vaccine_efficacy,
            "infectivity_profile": infectivity_profile,
            "infectivity_shape": infectivity_shape,
            "infectivity_scale": infectivity_scale,
            "latent_profile": latent_profile,
            "latent_shape": latent_shape,
            "latent_scale": latent_scale,
        },
    }


def reconstruct_seirdv_states(
    df: pd.DataFrame,
    *,
    population_column: str,
    cases_signal: str,
    deaths_signal: str,
    latent_period_days: int,
    infectious_period_days: int,
    infectivity_profile: str,
    infectivity_shape: float,
    infectivity_scale: float,
    latent_profile: str,
    latent_shape: float,
    latent_scale: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct SEIRDV states from observed signals."""
    if latent_period_days <= 0 or infectious_period_days <= 0:
        raise ValueError("latent_period_days and infectious_period_days must be > 0")

    out = df.copy()
    if population_column not in out.columns:
        raise ValueError(f"Population column missing: {population_column}")

    out[population_column] = out[population_column].astype(float).ffill().bfill()
    if out[population_column].isna().all():
        raise ValueError("Population column is fully missing after fill")

    population = out[population_column].astype(float)
    observed_cases = out[cases_signal].astype(float).clip(lower=0).interpolate(method="time").ffill().bfill()
    deaths_signal_series = out[deaths_signal].astype(float).clip(lower=0).interpolate(method="time").ffill().bfill()

    v_est, vaccination_column_used = _resolve_vaccination_series(out, population)
    nu_flow_raw = v_est.diff().fillna(0.0).clip(lower=0.0)
    nu_flow_smoothed = nu_flow_raw.rolling(window=7, min_periods=1).mean()

    infectivity_weights = build_profile_weights(
        infectious_period_days,
        profile=infectivity_profile,
        shape=infectivity_shape,
        scale=infectivity_scale,
    )
    latent_weights = build_profile_weights(
        latent_period_days,
        profile=latent_profile,
        shape=latent_shape,
        scale=latent_scale,
    )
    infectivity_weights_scaled = infectivity_weights * float(infectious_period_days)
    latent_weights_scaled = latent_weights * float(latent_period_days)

    i_est = _causal_convolution(observed_cases, infectivity_weights_scaled).clip(lower=0)
    e_est = _causal_convolution(observed_cases, latent_weights_scaled).clip(lower=0)
    i_est_old = observed_cases.rolling(window=infectious_period_days, min_periods=1).sum().clip(lower=0)
    e_base_old = observed_cases.rolling(window=latent_period_days, min_periods=1).sum()
    e_est_old = e_base_old.shift(1).fillna(e_base_old).clip(lower=0)

    if "total_deaths" in out.columns and out["total_deaths"].notna().any():
        d_est = out["total_deaths"].astype(float).clip(lower=0).interpolate(method="time").ffill().bfill()
    else:
        d_est = deaths_signal_series.cumsum()

    cumulative_infections = observed_cases.cumsum()
    r_est = (cumulative_infections - e_est - i_est - d_est).clip(lower=0)

    seirdv_sum = v_est + e_est + i_est + r_est + d_est
    overflow = (seirdv_sum - population).clip(lower=0)
    overflow_count = int((overflow > 0).sum())
    if float(overflow.max()) > 0:
        LOGGER.info(
            "SEIRDV reconstruction overflow detected on %s rows; clipping R_estimated to enforce compartment sum <= N",
            overflow_count,
        )
        r_est = (r_est - overflow).clip(lower=0)

    s_raw = population - v_est - e_est - i_est - r_est - d_est
    s_negative_count = int((s_raw < 0).sum())
    if s_negative_count > 0:
        LOGGER.info(
            "SEIRDV reconstruction produced negative S on %s rows; clipping S_estimated at 0",
            s_negative_count,
        )
    s_est = s_raw.clip(lower=0)
    nu_rate = nu_flow_smoothed / s_est.where(s_est > 0, np.nan)

    out["observed_cases_signal"] = observed_cases
    out["incidence_for_seirdv"] = observed_cases
    out["deaths_for_seirdv"] = deaths_signal_series
    out["vaccination_column_used"] = vaccination_column_used
    out["nu_flow_raw"] = nu_flow_raw
    out["nu_flow_smoothed"] = nu_flow_smoothed
    out["nu_rate_estimated"] = nu_rate
    out["cumulative_infections_estimated"] = cumulative_infections
    out["I_estimated_old"] = i_est_old
    out["E_estimated_old"] = e_est_old
    out["S_estimated"] = s_est
    out["V_estimated"] = v_est
    out["E_estimated"] = e_est
    out["I_estimated"] = i_est
    out["R_estimated"] = r_est
    out["D_estimated"] = d_est

    out["infectivity_profile"] = infectivity_profile
    out["infectivity_shape"] = float(infectivity_shape)
    out["infectivity_scale"] = float(infectivity_scale)
    out["latent_profile"] = latent_profile
    out["latent_shape"] = float(latent_shape)
    out["latent_scale"] = float(latent_scale)

    for lag, weight in enumerate(infectivity_weights):
        out[f"infectivity_weight_lag_{lag}"] = float(weight)
    for lag, weight in enumerate(latent_weights):
        out[f"latent_weight_lag_{lag}"] = float(weight)

    first_date = out.index.min()
    init = {
        "date": first_date.strftime("%Y-%m-%d"),
        "N": float(population.loc[first_date]),
        "S0": float(s_est.loc[first_date]),
        "V0": float(v_est.loc[first_date]),
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
        "vaccination_column_used": vaccination_column_used,
        "latent_period_days": latent_period_days,
        "infectious_period_days": infectious_period_days,
        "infectivity_profile": infectivity_profile,
        "latent_profile": latent_profile,
        "infectivity_weight_sum": float(np.sum(infectivity_weights)),
        "latent_weight_sum": float(np.sum(latent_weights)),
        "observed_cases_min": float(observed_cases.min()),
        "observed_cases_max": float(observed_cases.max()),
        "vaccinated_min": float(v_est.min()),
        "vaccinated_max": float(v_est.max()),
        "nu_flow_max": float(nu_flow_raw.max()),
        "overflow_clipped_count": overflow_count,
        "s_negative_clipped_count": s_negative_count,
        "i_old_peak_date": i_est_old.idxmax().strftime("%Y-%m-%d"),
        "i_old_peak_value": float(i_est_old.max()),
        "i_new_peak_date": i_est.idxmax().strftime("%Y-%m-%d"),
        "i_new_peak_value": float(i_est.max()),
        "i_old_new_correlation": float(i_est_old.corr(i_est)),
        "cases_vs_sigma_e_correlation": float(observed_cases.corr((1.0 / float(latent_period_days)) * e_est)),
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
    lines.append("SEIRDV Preparation Report")
    lines.append("=" * 29)
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
    lines.append("- observed_cases_signal: baseline observed incidence (new_cases_7d_avg or configured signal)")
    lines.append("- V_estimated: vaccination stock from people_fully_vaccinated (fallback: people_vaccinated)")
    lines.append("- nu_flow_raw / nu_flow_smoothed: daily vaccination flow estimated from V_estimated differences")
    lines.append("- I_estimated: observed_cases_signal convolved with infectivity profile weights")
    lines.append("- E_estimated: observed_cases_signal convolved with latent profile weights")
    lines.append("- D_estimated: total_deaths if available else cumulative smoothed deaths signal")
    lines.append("- R_estimated: cumulative observed infections - E_estimated - I_estimated - D_estimated")
    lines.append("- S_estimated: N - V_estimated - E_estimated - I_estimated - R_estimated - D_estimated")
    lines.append("")
    return "\n".join(lines)


def run_seirdv_preparation_pipeline(
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
    vaccine_efficacy: float = DEFAULT_VACCINE_EFFICACY,
    infectivity_profile: str = DEFAULT_INFECTIVITY_PROFILE,
    infectivity_shape: float = DEFAULT_INFECTIVITY_SHAPE,
    infectivity_scale: float = DEFAULT_INFECTIVITY_SCALE,
    latent_profile: str = DEFAULT_LATENT_PROFILE,
    latent_shape: float = DEFAULT_LATENT_SHAPE,
    latent_scale: float = DEFAULT_LATENT_SCALE,
    save_csv: bool = False,
) -> dict[str, Any]:
    """Run SEIRDV-ready dataset preparation pipeline."""
    if not (0.0 <= float(vaccine_efficacy) <= 1.0):
        raise ValueError("vaccine_efficacy must be within [0, 1]")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    country_slug = _slugify_country(country)
    parsed_start = pd.Timestamp(start_date)
    parsed_end = pd.Timestamp(end_date)

    dataset = load_input_dataset(input_path)
    dataset = _ensure_signal_columns(dataset, cases_signal=cases_signal, deaths_signal=deaths_signal)
    dataset = dataset.loc[parsed_start:parsed_end].copy()
    if dataset.empty:
        raise ValueError("Selected SEIRDV study window is empty after filtering")

    framework = define_seirdv_framework(
        country=country,
        start_date=parsed_start,
        end_date=parsed_end,
        cases_signal=cases_signal,
        deaths_signal=deaths_signal,
        latent_period_days=latent_period_days,
        infectious_period_days=infectious_period_days,
        vaccine_efficacy=vaccine_efficacy,
        infectivity_profile=infectivity_profile,
        infectivity_shape=infectivity_shape,
        infectivity_scale=infectivity_scale,
        latent_profile=latent_profile,
        latent_shape=latent_shape,
        latent_scale=latent_scale,
    )

    seirdv_ready_df, reconstruction_summary = reconstruct_seirdv_states(
        dataset,
        population_column=population_column,
        cases_signal=cases_signal,
        deaths_signal=deaths_signal,
        latent_period_days=latent_period_days,
        infectious_period_days=infectious_period_days,
        infectivity_profile=infectivity_profile,
        infectivity_shape=infectivity_shape,
        infectivity_scale=infectivity_scale,
        latent_profile=latent_profile,
        latent_shape=latent_shape,
        latent_scale=latent_scale,
    )

    ready_path = output_dir / f"covid_{country_slug}_seirdv_ready.parquet"
    seirdv_ready_df.to_parquet(ready_path)

    csv_path: Path | None = None
    if save_csv:
        csv_path = ready_path.with_suffix(".csv")
        seirdv_ready_df.to_csv(csv_path, index=True)

    report_text = _build_preparation_report_text(
        framework=framework,
        summary=reconstruction_summary,
        population_column=population_column,
    )
    report_path = reports_dir / f"covid_{country_slug}_seirdv_preparation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\n=== SEIRDV Preparation Summary ===")
    print(report_text)

    LOGGER.info("SEIRDV-ready dataset saved: %s", ready_path)
    LOGGER.info("SEIRDV preparation report saved: %s", report_path)

    return {
        "dataset": seirdv_ready_df,
        "framework": framework,
        "reconstruction_summary": reconstruction_summary,
        "output_path": ready_path,
        "csv_output_path": csv_path,
        "report_path": report_path,
        "report_text": report_text,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for SEIRDV preparation."""
    parser = argparse.ArgumentParser(description="Prepare SEIRDV-ready dataset")
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
        help="Directory for SEIRDV-ready dataset",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/processed/reports/seirdv"),
        help="Directory for SEIRDV preparation report",
    )
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE, help="Study window start date")
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE, help="Study window end date")
    parser.add_argument("--cases-signal", type=str, default=DEFAULT_CASES_SIGNAL, help="Incidence signal column")
    parser.add_argument("--deaths-signal", type=str, default=DEFAULT_DEATHS_SIGNAL, help="Deaths signal column")
    parser.add_argument("--population-column", type=str, default=DEFAULT_POPULATION_COLUMN, help="Population column")
    parser.add_argument("--latent-period-days", type=int, default=5, help="Latent period assumption")
    parser.add_argument("--infectious-period-days", type=int, default=14, help="Infectious period assumption")
    parser.add_argument(
        "--vaccine-efficacy",
        type=float,
        default=DEFAULT_VACCINE_EFFICACY,
        help="Fixed vaccine protection against infection in [0, 1]",
    )
    parser.add_argument(
        "--infectivity-profile",
        type=str,
        choices=["uniform", "gamma"],
        default=DEFAULT_INFECTIVITY_PROFILE,
        help="Infectivity convolution profile for I_estimated",
    )
    parser.add_argument("--infectivity-shape", type=float, default=DEFAULT_INFECTIVITY_SHAPE)
    parser.add_argument("--infectivity-scale", type=float, default=DEFAULT_INFECTIVITY_SCALE)
    parser.add_argument(
        "--latent-profile",
        type=str,
        choices=["uniform", "gamma"],
        default=DEFAULT_LATENT_PROFILE,
        help="Latent convolution profile for E_estimated",
    )
    parser.add_argument("--latent-shape", type=float, default=DEFAULT_LATENT_SHAPE)
    parser.add_argument("--latent-scale", type=float, default=DEFAULT_LATENT_SCALE)
    parser.add_argument("--save-csv", action="store_true", help="Also save SEIRDV-ready dataset as CSV")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    run_seirdv_preparation_pipeline(
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
        vaccine_efficacy=args.vaccine_efficacy,
        infectivity_profile=args.infectivity_profile,
        infectivity_shape=args.infectivity_shape,
        infectivity_scale=args.infectivity_scale,
        latent_profile=args.latent_profile,
        latent_shape=args.latent_shape,
        latent_scale=args.latent_scale,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
