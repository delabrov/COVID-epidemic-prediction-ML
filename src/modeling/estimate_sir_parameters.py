"""Estimate time-varying SIR parameters from reconstructed SIR states."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_DERIVATIVE_METHOD = "gradient"
DEFAULT_BETA_SMOOTHING_WINDOW = 7
DEFAULT_INFECTIOUS_PERIOD_DAYS = 14
DEFAULT_MIN_INFECTED_THRESHOLD = 10.0
DEFAULT_EPSILON = 1e-8

REQUIRED_SIR_COLUMNS: tuple[str, ...] = ("S_estimated", "I_estimated", "R_estimated")


def _slugify_country(country: str) -> str:
    """Return filesystem-safe country slug."""
    return re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure input dataframe has a sorted DatetimeIndex named 'date'."""
    output = df.copy()
    if isinstance(output.index, pd.DatetimeIndex):
        output.index = pd.to_datetime(output.index, errors="coerce")
    elif "date" in output.columns:
        output["date"] = pd.to_datetime(output["date"], errors="coerce")
        output = output.dropna(subset=["date"]).set_index("date")
    else:
        raise ValueError("Input dataframe must have a DatetimeIndex or a 'date' column")

    output = output[~output.index.isna()].sort_index()
    output.index.name = "date"
    return output


def load_sir_ready_data(input_path: Path) -> pd.DataFrame:
    """Load SIR-ready dataset from parquet or CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input extension: {input_path.suffix}")

    return _ensure_datetime_index(df)


def resolve_population_column(df: pd.DataFrame, preferred: str = "population_for_sir") -> str:
    """Resolve which population column to use for SIR parameter estimation."""
    candidates = [preferred, "population_for_sir", "population"]
    for column in dict.fromkeys(candidates):
        if column in df.columns:
            return column
    raise ValueError("Population column not found. Tried: population_for_sir, population")


def validate_sir_columns(df: pd.DataFrame, population_column: str) -> None:
    """Validate required columns for SIR parameter estimation."""
    missing = [column for column in REQUIRED_SIR_COLUMNS if column not in df.columns]
    if population_column not in df.columns:
        missing.append(population_column)

    if missing:
        raise ValueError(f"Missing required columns for SIR parameter estimation: {missing}")


def estimate_gamma(infectious_period_days: int) -> float:
    """Estimate gamma from infectious period assumption."""
    if infectious_period_days <= 0:
        raise ValueError("infectious_period_days must be > 0")
    return 1.0 / float(infectious_period_days)


def smooth_series(series: pd.Series, window: int = DEFAULT_BETA_SMOOTHING_WINDOW) -> pd.Series:
    """Apply rolling-mean smoothing while preserving index."""
    if window <= 1:
        return series.copy()
    return series.rolling(window=window, min_periods=1).mean()


def estimate_dI_dt(
    series: pd.Series,
    method: str = DEFAULT_DERIVATIVE_METHOD,
    pre_smoothing_window: int = 1,
) -> pd.Series:
    """Estimate dI/dt using stable finite differences on daily data."""
    signal = series.astype(float).copy()
    if pre_smoothing_window > 1:
        signal = smooth_series(signal, window=pre_smoothing_window)

    observed_mask = signal.notna()
    filled = signal.interpolate(limit_direction="both")

    if method == "gradient":
        derivative = pd.Series(np.gradient(filled.to_numpy(dtype=float), 1.0), index=series.index)
    elif method == "diff":
        derivative = filled.diff()
        if len(derivative) > 1:
            derivative.iloc[0] = derivative.iloc[1]
        elif len(derivative) == 1:
            derivative.iloc[0] = 0.0
    else:
        raise ValueError(f"Unsupported derivative method: {method}")

    derivative = derivative.where(observed_mask)
    derivative.name = "dI_dt"
    return derivative


def estimate_beta(
    df: pd.DataFrame,
    gamma: float,
    *,
    population_column: str,
    epsilon: float = DEFAULT_EPSILON,
    min_infected: float = DEFAULT_MIN_INFECTED_THRESHOLD,
    min_denominator: float = 1.0,
) -> pd.Series:
    """Estimate beta(t) from reconstructed SIR states and dI/dt."""
    if "dI_dt" not in df.columns:
        raise ValueError("Column 'dI_dt' is required before beta estimation")

    susceptible = df["S_estimated"].astype(float)
    infected = df["I_estimated"].astype(float)
    population = df[population_column].astype(float)
    denominator = (susceptible * infected) / population

    numerator = df["dI_dt"].astype(float) + gamma * infected
    threshold = max(float(epsilon), float(min_denominator))

    valid_mask = (
        denominator.abs() > threshold
    ) & (infected >= float(min_infected)) & np.isfinite(denominator) & np.isfinite(numerator)

    beta = numerator / denominator
    beta = beta.where(valid_mask)
    beta.name = "beta_raw"
    return beta


def compute_reproduction_numbers(
    df: pd.DataFrame,
    beta_series: pd.Series,
    gamma: float,
    *,
    population_column: str,
) -> pd.DataFrame:
    """Compute R0 proxy and effective reproduction number from beta(t)."""
    susceptible = df["S_estimated"].astype(float)
    population = df[population_column].astype(float)

    r0_proxy = beta_series / gamma
    reff = r0_proxy * (susceptible / population)

    output = pd.DataFrame(index=df.index)
    output["R0_proxy"] = r0_proxy
    output["R_eff"] = reff
    return output


def _format_stats_block(series: pd.Series) -> str:
    cleaned = series.dropna()
    if cleaned.empty:
        return "count=0 (all values are NaN)"

    desc = cleaned.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    return (
        f"count={int(desc['count'])}, mean={desc['mean']:.6f}, std={desc['std']:.6f}, "
        f"min={desc['min']:.6f}, p05={desc['5%']:.6f}, p25={desc['25%']:.6f}, "
        f"p50={desc['50%']:.6f}, p75={desc['75%']:.6f}, p95={desc['95%']:.6f}, max={desc['max']:.6f}"
    )


def summarize_parameter_estimates(df: pd.DataFrame) -> str:
    """Build textual summary stats for estimated SIR parameters."""
    lines: list[str] = []
    target_columns = [
        "beta_raw",
        "beta_smoothed",
        "R_eff_raw",
        "R_eff_smoothed",
        "R0_proxy_raw",
        "R0_proxy_smoothed",
    ]

    for column in target_columns:
        if column in df.columns:
            lines.append(f"- {column}: {_format_stats_block(df[column].astype(float))}")
        else:
            lines.append(f"- {column}: unavailable")

    return "\n".join(lines)


def _build_parameter_report(
    *,
    country: str,
    dataset: pd.DataFrame,
    infectious_period_days: int,
    gamma: float,
    derivative_method: str,
    derivative_smoothing_window: int,
    beta_smoothing_window: int,
    epsilon: float,
    min_infected_threshold: float,
    min_denominator: float,
    population_column: str,
) -> str:
    total_rows = len(dataset)
    valid_beta_points = int(dataset.get("valid_beta_mask", pd.Series(index=dataset.index, dtype=bool)).fillna(False).sum())
    invalid_beta_points = int(total_rows - valid_beta_points)

    lines: list[str] = []
    lines.append("SIR Parameter Estimation Report")
    lines.append("=" * 34)
    lines.append("")
    lines.append("1. Dataset overview")
    lines.append(f"- country: {country}")
    lines.append(f"- start_date: {dataset.index.min().date()}")
    lines.append(f"- end_date: {dataset.index.max().date()}")
    lines.append(f"- rows: {total_rows}")
    lines.append("")
    lines.append("2. Estimation configuration")
    lines.append(f"- infectious_period_days: {infectious_period_days}")
    lines.append(f"- gamma: {gamma:.8f}")
    lines.append(f"- derivative_method: {derivative_method}")
    lines.append(f"- derivative_pre_smoothing_window: {derivative_smoothing_window}")
    lines.append(f"- beta_smoothing_window: {beta_smoothing_window}")
    lines.append(f"- epsilon: {epsilon}")
    lines.append(f"- min_infected_threshold: {min_infected_threshold}")
    lines.append(f"- min_denominator_SI_over_N: {min_denominator}")
    lines.append(f"- population_column_used: {population_column}")
    lines.append("")
    lines.append("3. Validity diagnostics")
    lines.append(f"- valid beta points: {valid_beta_points}")
    lines.append(f"- invalid/NaN beta points: {invalid_beta_points}")
    lines.append("")
    lines.append("4. Summary statistics")
    lines.append(summarize_parameter_estimates(dataset))
    lines.append("")
    lines.append("5. Methodological caveats")
    lines.append("- S, I, R are reconstructed proxies, not directly observed compartments.")
    lines.append("- beta(t) is inferred from estimated states and numerical derivatives.")
    lines.append("- Estimates near low I(t) values are unstable and masked by safeguards.")

    return "\n".join(lines)


def save_parameter_estimation_report(report_text: str, output_path: Path) -> Path:
    """Persist parameter-estimation report to text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def run_sir_parameter_estimation_pipeline(
    *,
    country: str,
    input_path: Path,
    output_dir: Path,
    reports_dir: Path,
    infectious_period_days: int = DEFAULT_INFECTIOUS_PERIOD_DAYS,
    derivative_method: str = DEFAULT_DERIVATIVE_METHOD,
    derivative_smoothing_window: int = 1,
    beta_smoothing_window: int = DEFAULT_BETA_SMOOTHING_WINDOW,
    min_infected_threshold: float = DEFAULT_MIN_INFECTED_THRESHOLD,
    epsilon: float = DEFAULT_EPSILON,
    min_denominator: float = 1.0,
    population_column: str = "population_for_sir",
    save_csv: bool = False,
) -> dict[str, Any]:
    """Run SIR parameter estimation and save enriched outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    country_slug = _slugify_country(country)
    dataset = load_sir_ready_data(input_path)

    resolved_population_column = resolve_population_column(dataset, preferred=population_column)
    validate_sir_columns(dataset, population_column=resolved_population_column)

    gamma = estimate_gamma(infectious_period_days)

    enriched = dataset.copy()
    enriched["gamma"] = gamma
    enriched["dI_dt"] = estimate_dI_dt(
        enriched["I_estimated"],
        method=derivative_method,
        pre_smoothing_window=derivative_smoothing_window,
    )

    susceptible = enriched["S_estimated"].astype(float)
    infected = enriched["I_estimated"].astype(float)
    population = enriched[resolved_population_column].astype(float)
    enriched["denominator_SI_over_N"] = (susceptible * infected) / population

    threshold = max(float(epsilon), float(min_denominator))
    enriched["valid_beta_mask"] = (
        (enriched["denominator_SI_over_N"].abs() > threshold)
        & (infected >= float(min_infected_threshold))
        & np.isfinite(enriched["denominator_SI_over_N"])
        & np.isfinite(enriched["dI_dt"])
    )

    enriched["beta_raw"] = estimate_beta(
        enriched,
        gamma,
        population_column=resolved_population_column,
        epsilon=epsilon,
        min_infected=min_infected_threshold,
        min_denominator=min_denominator,
    )
    enriched["beta_smoothed"] = smooth_series(enriched["beta_raw"], window=beta_smoothing_window)
    enriched["beta_smoothed_nonnegative"] = enriched["beta_smoothed"].clip(lower=0)

    raw_reproduction = compute_reproduction_numbers(
        enriched,
        enriched["beta_raw"],
        gamma,
        population_column=resolved_population_column,
    )
    smooth_reproduction = compute_reproduction_numbers(
        enriched,
        enriched["beta_smoothed"],
        gamma,
        population_column=resolved_population_column,
    )

    enriched["R0_proxy_raw"] = raw_reproduction["R0_proxy"]
    enriched["R_eff_raw"] = raw_reproduction["R_eff"]
    enriched["R0_proxy_smoothed"] = smooth_reproduction["R0_proxy"]
    enriched["R_eff_smoothed"] = smooth_reproduction["R_eff"]

    parameters_output_path = output_dir / f"covid_{country_slug}_sir_parameters.parquet"
    enriched.to_parquet(parameters_output_path)

    csv_output_path: Path | None = None
    if save_csv:
        csv_output_path = parameters_output_path.with_suffix(".csv")
        enriched.to_csv(csv_output_path, index=True)

    report_text = _build_parameter_report(
        country=country,
        dataset=enriched,
        infectious_period_days=infectious_period_days,
        gamma=gamma,
        derivative_method=derivative_method,
        derivative_smoothing_window=derivative_smoothing_window,
        beta_smoothing_window=beta_smoothing_window,
        epsilon=epsilon,
        min_infected_threshold=min_infected_threshold,
        min_denominator=min_denominator,
        population_column=resolved_population_column,
    )

    report_path = reports_dir / f"covid_{country_slug}_sir_parameter_report.txt"
    save_parameter_estimation_report(report_text, report_path)

    print("\n=== SIR Parameter Estimation Summary ===")
    print(report_text)

    LOGGER.info("SIR parameter dataset saved: %s", parameters_output_path)
    LOGGER.info("SIR parameter report saved: %s", report_path)

    return {
        "dataset": enriched,
        "gamma": gamma,
        "output_path": parameters_output_path,
        "csv_output_path": csv_output_path,
        "report_path": report_path,
        "report_text": report_text,
        "population_column": resolved_population_column,
        "valid_beta_points": int(enriched["valid_beta_mask"].fillna(False).sum()),
        "invalid_beta_points": int(len(enriched) - enriched["valid_beta_mask"].fillna(False).sum()),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for independent SIR parameter estimation run."""
    parser = argparse.ArgumentParser(description="Estimate time-varying SIR parameters")
    parser.add_argument("--country", type=str, default="France", help="Country label")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/sir/inputs/covid_france_sir_prepared.parquet"),
        help="Path to SIR-ready dataset (parquet/csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for enriched SIR-parameter dataset",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/processed/reports/sir"),
        help="Directory for parameter-estimation reports",
    )
    parser.add_argument(
        "--infectious-period-days",
        type=int,
        default=DEFAULT_INFECTIOUS_PERIOD_DAYS,
        help="Infectious period assumption used to compute gamma",
    )
    parser.add_argument(
        "--derivative-method",
        type=str,
        choices=["gradient", "diff"],
        default=DEFAULT_DERIVATIVE_METHOD,
        help="Finite-difference method for dI/dt",
    )
    parser.add_argument(
        "--derivative-smoothing-window",
        type=int,
        default=1,
        help="Optional rolling window to smooth I_estimated before differentiation",
    )
    parser.add_argument(
        "--beta-smoothing-window",
        type=int,
        default=DEFAULT_BETA_SMOOTHING_WINDOW,
        help="Rolling window for beta and reproduction smoothing",
    )
    parser.add_argument(
        "--min-infected-threshold",
        type=float,
        default=DEFAULT_MIN_INFECTED_THRESHOLD,
        help="Minimum I(t) threshold for valid beta estimation",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Small numerical threshold for safe division",
    )
    parser.add_argument(
        "--min-denominator",
        type=float,
        default=1.0,
        help="Minimum denominator S*I/N threshold for valid beta estimation",
    )
    parser.add_argument(
        "--population-column",
        type=str,
        default="population_for_sir",
        help="Preferred population column name",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save enriched SIR-parameter dataset as CSV",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    run_sir_parameter_estimation_pipeline(
        country=args.country,
        input_path=args.input_path,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        infectious_period_days=args.infectious_period_days,
        derivative_method=args.derivative_method,
        derivative_smoothing_window=args.derivative_smoothing_window,
        beta_smoothing_window=args.beta_smoothing_window,
        min_infected_threshold=args.min_infected_threshold,
        epsilon=args.epsilon,
        min_denominator=args.min_denominator,
        population_column=args.population_column,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
