"""Estimate SEIRD parameters from reconstructed SEIRD state trajectories."""

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
DEFAULT_SMOOTHING_WINDOW = 7
DEFAULT_LATENT_PERIOD_DAYS = 5
DEFAULT_INFECTIOUS_PERIOD_DAYS = 14
DEFAULT_EPSILON = 1e-8
DEFAULT_MIN_INFECTED_THRESHOLD = 10.0
DEFAULT_MIN_EXPOSED_THRESHOLD = 10.0
DEFAULT_MIN_DENOMINATOR = 1.0

REQUIRED_COLUMNS: tuple[str, ...] = (
    "S_estimated",
    "E_estimated",
    "I_estimated",
    "R_estimated",
    "D_estimated",
)


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


def load_seird_ready_data(input_path: Path) -> pd.DataFrame:
    """Load SEIRD-ready dataset (parquet/csv)."""
    if not input_path.exists():
        raise FileNotFoundError(f"SEIRD-ready dataset not found: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input extension: {input_path.suffix}")
    return _ensure_datetime_index(df)


def validate_required_columns(df: pd.DataFrame, population_column: str) -> None:
    """Validate required columns for SEIRD parameter estimation."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if population_column not in df.columns:
        missing.append(population_column)
    if missing:
        raise ValueError(f"Missing required SEIRD columns: {missing}")


def estimate_sigma(latent_period_days: int) -> float:
    """Estimate sigma from latent period assumption."""
    if latent_period_days <= 0:
        raise ValueError("latent_period_days must be > 0")
    return 1.0 / float(latent_period_days)


def estimate_gamma(infectious_period_days: int) -> float:
    """Estimate gamma from infectious period assumption."""
    if infectious_period_days <= 0:
        raise ValueError("infectious_period_days must be > 0")
    return 1.0 / float(infectious_period_days)


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    """Smooth series with rolling mean."""
    if window <= 1:
        return series.copy()
    return series.rolling(window=window, min_periods=1).mean()


def estimate_derivative(
    series: pd.Series,
    *,
    method: str = DEFAULT_DERIVATIVE_METHOD,
    pre_smoothing_window: int = 1,
) -> pd.Series:
    """Estimate derivative using stable finite-difference method."""
    signal = series.astype(float).copy()
    if pre_smoothing_window > 1:
        signal = smooth_series(signal, pre_smoothing_window)

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

    return derivative.where(observed_mask)


def estimate_mu(
    df: pd.DataFrame,
    *,
    epsilon: float,
    min_infected_threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """Estimate mu(t) from dD/dt = mu * I."""
    infected = df["I_estimated"].astype(float)
    d_d_dt = df["dD_dt"].astype(float)
    denom_threshold = max(float(epsilon), float(min_infected_threshold))

    valid_mask = infected.abs() > denom_threshold
    mu = (d_d_dt / infected).where(valid_mask)
    mu.name = "mu_raw"
    return mu, valid_mask


def estimate_beta(
    df: pd.DataFrame,
    *,
    sigma: float,
    population_column: str,
    epsilon: float,
    min_infected_threshold: float,
    min_exposed_threshold: float,
    min_denominator: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Estimate beta(t) from dE/dt = beta*S*I/N - sigma*E."""
    susceptible = df["S_estimated"].astype(float)
    exposed = df["E_estimated"].astype(float)
    infected = df["I_estimated"].astype(float)
    population = df[population_column].astype(float)

    denominator = (susceptible * infected) / population
    numerator = df["dE_dt"].astype(float) + sigma * exposed
    denom_threshold = max(float(epsilon), float(min_denominator))

    valid_mask = (
        (denominator.abs() > denom_threshold)
        & (infected >= float(min_infected_threshold))
        & (exposed >= float(min_exposed_threshold))
        & np.isfinite(denominator)
        & np.isfinite(numerator)
    )
    beta = (numerator / denominator).where(valid_mask)
    beta.name = "beta_raw"
    return beta, denominator, valid_mask


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
    """Build summary statistics for key SEIRD parameters."""
    lines: list[str] = []
    for column in [
        "beta_raw",
        "beta_smoothed",
        "mu_raw",
        "mu_smoothed",
        "R_eff_proxy_raw",
        "R_eff_proxy_smoothed",
    ]:
        if column in df.columns:
            lines.append(f"- {column}: {_format_stats_block(df[column].astype(float))}")
        else:
            lines.append(f"- {column}: unavailable")
    return "\n".join(lines)


def _build_parameter_report_text(
    *,
    country: str,
    dataset: pd.DataFrame,
    latent_period_days: int,
    infectious_period_days: int,
    sigma: float,
    gamma: float,
    derivative_method: str,
    derivative_smoothing_window: int,
    smoothing_window: int,
    epsilon: float,
    min_infected_threshold: float,
    min_exposed_threshold: float,
    min_denominator: float,
    population_column: str,
) -> str:
    valid_beta = int(dataset["valid_beta_mask"].fillna(False).sum()) if "valid_beta_mask" in dataset.columns else 0
    valid_mu = int(dataset["valid_mu_mask"].fillna(False).sum()) if "valid_mu_mask" in dataset.columns else 0

    lines: list[str] = []
    lines.append("SEIRD Parameter Estimation Report")
    lines.append("=" * 36)
    lines.append("")
    lines.append("1. Dataset overview")
    lines.append(f"- country: {country}")
    lines.append(f"- start_date: {dataset.index.min().date()}")
    lines.append(f"- end_date: {dataset.index.max().date()}")
    lines.append(f"- rows: {len(dataset)}")
    lines.append("")
    lines.append("2. Estimation configuration")
    lines.append(f"- latent_period_days: {latent_period_days}")
    lines.append(f"- infectious_period_days: {infectious_period_days}")
    lines.append(f"- sigma: {sigma:.8f}")
    lines.append(f"- gamma: {gamma:.8f}")
    lines.append(f"- derivative_method: {derivative_method}")
    lines.append(f"- derivative_pre_smoothing_window: {derivative_smoothing_window}")
    lines.append(f"- smoothing_window: {smoothing_window}")
    lines.append(f"- epsilon: {epsilon}")
    lines.append(f"- min_infected_threshold: {min_infected_threshold}")
    lines.append(f"- min_exposed_threshold: {min_exposed_threshold}")
    lines.append(f"- min_denominator_SI_over_N: {min_denominator}")
    lines.append(f"- population_column: {population_column}")
    lines.append("")
    lines.append("3. Validity diagnostics")
    lines.append(f"- valid beta points: {valid_beta}")
    lines.append(f"- invalid beta points: {len(dataset) - valid_beta}")
    lines.append(f"- valid mu points: {valid_mu}")
    lines.append(f"- invalid mu points: {len(dataset) - valid_mu}")
    lines.append("")
    lines.append("4. Summary statistics")
    lines.append(summarize_parameter_estimates(dataset))
    lines.append("")
    lines.append("5. Methodological caveats")
    lines.append("- Reconstructed compartments are proxies, not directly observed states.")
    lines.append("- Deaths are lagging indicators and affect mu(t) stability.")
    lines.append("- Parameter estimates may be unstable when E or I are small.")
    lines.append("")
    return "\n".join(lines)


def run_seird_parameter_estimation_pipeline(
    *,
    country: str,
    input_path: Path,
    output_dir: Path,
    reports_dir: Path,
    latent_period_days: int = DEFAULT_LATENT_PERIOD_DAYS,
    infectious_period_days: int = DEFAULT_INFECTIOUS_PERIOD_DAYS,
    derivative_method: str = DEFAULT_DERIVATIVE_METHOD,
    derivative_smoothing_window: int = 1,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    epsilon: float = DEFAULT_EPSILON,
    min_infected_threshold: float = DEFAULT_MIN_INFECTED_THRESHOLD,
    min_exposed_threshold: float = DEFAULT_MIN_EXPOSED_THRESHOLD,
    min_denominator: float = DEFAULT_MIN_DENOMINATOR,
    population_column: str = "population",
    save_csv: bool = False,
) -> dict[str, Any]:
    """Run SEIRD parameter estimation and save enriched dataset/report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    country_slug = _slugify_country(country)
    dataset = load_seird_ready_data(input_path)
    validate_required_columns(dataset, population_column=population_column)

    sigma = estimate_sigma(latent_period_days)
    gamma = estimate_gamma(infectious_period_days)

    enriched = dataset.copy()
    enriched["sigma"] = sigma
    enriched["gamma"] = gamma
    enriched["dE_dt"] = estimate_derivative(
        enriched["E_estimated"],
        method=derivative_method,
        pre_smoothing_window=derivative_smoothing_window,
    )
    enriched["dD_dt"] = estimate_derivative(
        enriched["D_estimated"],
        method=derivative_method,
        pre_smoothing_window=derivative_smoothing_window,
    )

    beta_raw, denominator_s_i_over_n, beta_valid_mask = estimate_beta(
        enriched,
        sigma=sigma,
        population_column=population_column,
        epsilon=epsilon,
        min_infected_threshold=min_infected_threshold,
        min_exposed_threshold=min_exposed_threshold,
        min_denominator=min_denominator,
    )
    enriched["denominator_SI_over_N"] = denominator_s_i_over_n
    enriched["valid_beta_mask"] = beta_valid_mask
    enriched["beta_raw"] = beta_raw
    enriched["beta_smoothed"] = smooth_series(enriched["beta_raw"], smoothing_window)
    enriched["beta_smoothed_nonnegative"] = enriched["beta_smoothed"].clip(lower=0)

    mu_raw, mu_valid_mask = estimate_mu(
        enriched,
        epsilon=epsilon,
        min_infected_threshold=min_infected_threshold,
    )
    enriched["valid_mu_mask"] = mu_valid_mask
    enriched["mu_raw"] = mu_raw
    enriched["mu_smoothed"] = smooth_series(enriched["mu_raw"], smoothing_window)
    enriched["mu_smoothed_nonnegative"] = enriched["mu_smoothed"].clip(lower=0)

    # Consistency terms for plotting diagnostics.
    enriched["beta_s_i_over_n_smoothed"] = enriched["beta_smoothed"] * enriched["denominator_SI_over_N"]
    enriched["sigma_e_term"] = sigma * enriched["E_estimated"]
    enriched["mu_i_term_smoothed"] = enriched["mu_smoothed"] * enriched["I_estimated"]

    denom_raw = gamma + enriched["mu_raw"]
    denom_smooth = gamma + enriched["mu_smoothed"]
    safe_raw = denom_raw.where(denom_raw.abs() > epsilon)
    safe_smooth = denom_smooth.where(denom_smooth.abs() > epsilon)

    enriched["R0_proxy_raw"] = enriched["beta_raw"] / safe_raw
    enriched["R0_proxy_smoothed"] = enriched["beta_smoothed"] / safe_smooth
    susceptible_fraction = enriched["S_estimated"] / enriched[population_column]
    enriched["R_eff_proxy_raw"] = enriched["R0_proxy_raw"] * susceptible_fraction
    enriched["R_eff_proxy_smoothed"] = enriched["R0_proxy_smoothed"] * susceptible_fraction

    output_path = output_dir / f"covid_{country_slug}_seird_parameters.parquet"
    enriched.to_parquet(output_path)

    csv_output_path: Path | None = None
    if save_csv:
        csv_output_path = output_path.with_suffix(".csv")
        enriched.to_csv(csv_output_path, index=True)

    report_text = _build_parameter_report_text(
        country=country,
        dataset=enriched,
        latent_period_days=latent_period_days,
        infectious_period_days=infectious_period_days,
        sigma=sigma,
        gamma=gamma,
        derivative_method=derivative_method,
        derivative_smoothing_window=derivative_smoothing_window,
        smoothing_window=smoothing_window,
        epsilon=epsilon,
        min_infected_threshold=min_infected_threshold,
        min_exposed_threshold=min_exposed_threshold,
        min_denominator=min_denominator,
        population_column=population_column,
    )
    report_path = reports_dir / f"covid_{country_slug}_seird_parameter_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\n=== SEIRD Parameter Estimation Summary ===")
    print(report_text)

    LOGGER.info("SEIRD parameter dataset saved: %s", output_path)
    LOGGER.info("SEIRD parameter report saved: %s", report_path)

    return {
        "dataset": enriched,
        "output_path": output_path,
        "csv_output_path": csv_output_path,
        "report_path": report_path,
        "report_text": report_text,
        "sigma": sigma,
        "gamma": gamma,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for SEIRD parameter estimation."""
    parser = argparse.ArgumentParser(description="Estimate SEIRD parameters from reconstructed states")
    parser.add_argument("--country", type=str, default="France", help="Country label")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/covid_france_seird_ready.parquet"),
        help="Path to SEIRD-ready dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for SEIRD parameter dataset",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/processed/reports/seird"),
        help="Directory for SEIRD parameter report",
    )
    parser.add_argument("--latent-period-days", type=int, default=DEFAULT_LATENT_PERIOD_DAYS)
    parser.add_argument("--infectious-period-days", type=int, default=DEFAULT_INFECTIOUS_PERIOD_DAYS)
    parser.add_argument(
        "--derivative-method",
        type=str,
        choices=["gradient", "diff"],
        default=DEFAULT_DERIVATIVE_METHOD,
    )
    parser.add_argument("--derivative-smoothing-window", type=int, default=1)
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--min-infected-threshold", type=float, default=DEFAULT_MIN_INFECTED_THRESHOLD)
    parser.add_argument("--min-exposed-threshold", type=float, default=DEFAULT_MIN_EXPOSED_THRESHOLD)
    parser.add_argument("--min-denominator", type=float, default=DEFAULT_MIN_DENOMINATOR)
    parser.add_argument("--population-column", type=str, default="population")
    parser.add_argument("--save-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    run_seird_parameter_estimation_pipeline(
        country=args.country,
        input_path=args.input_path,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        latent_period_days=args.latent_period_days,
        infectious_period_days=args.infectious_period_days,
        derivative_method=args.derivative_method,
        derivative_smoothing_window=args.derivative_smoothing_window,
        smoothing_window=args.smoothing_window,
        epsilon=args.epsilon,
        min_infected_threshold=args.min_infected_threshold,
        min_exposed_threshold=args.min_exposed_threshold,
        min_denominator=args.min_denominator,
        population_column=args.population_column,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
