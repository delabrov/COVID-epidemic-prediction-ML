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
DEFAULT_DEATH_DELAY_DAYS = 14
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
    death_delay_days: int,
    window: int,
    epsilon: float,
    min_infected_threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """Estimate mu(t) with rolling integrated method: ΔD(window) / ΣI_lagged(window)."""
    if death_delay_days < 0:
        raise ValueError("death_delay_days must be >= 0")
    if window <= 0:
        raise ValueError("window must be > 0")
    if "D_estimated" not in df.columns:
        raise ValueError("Missing required column for mu estimation: D_estimated")
    if "I_lagged_for_death" not in df.columns:
        raise ValueError("Missing required column for mu estimation: I_lagged_for_death")

    deaths_cumulative = df["D_estimated"].astype(float)
    infected_lagged = df["I_lagged_for_death"].astype(float)

    if deaths_cumulative.empty:
        empty = pd.Series(index=df.index, dtype=float, name="mu_raw")
        return empty, pd.Series(index=df.index, dtype=bool)

    delta_d = deaths_cumulative - deaths_cumulative.shift(window)
    # For first window points, use partial accumulation from the series start.
    delta_d = delta_d.where(delta_d.notna(), deaths_cumulative - float(deaths_cumulative.iloc[0]))
    infected_sum = infected_lagged.rolling(window=window, min_periods=1).sum()
    denom_threshold = max(float(epsilon), float(min_infected_threshold))
    valid_mask = (
        (infected_sum >= denom_threshold)
        & (delta_d >= 0.0)
        & np.isfinite(infected_sum)
        & np.isfinite(delta_d)
    )
    mu = (delta_d / (infected_sum + float(epsilon))).where(valid_mask)
    mu = mu.replace([np.inf, -np.inf], np.nan)
    mu.name = "mu_raw"
    return mu, valid_mask


def detect_mu_stable_start_date(
    mu_smoothed: pd.Series,
    *,
    window_stability: int = 14,
    k: float = 8.0,
) -> pd.Timestamp | None:
    """Detect first stable mu date using robust threshold median + k*MAD."""
    mu_series = mu_smoothed.replace([np.inf, -np.inf], np.nan).dropna()
    if mu_series.empty:
        return None
    if window_stability <= 0:
        raise ValueError("window_stability must be > 0")

    if len(mu_series) < window_stability:
        LOGGER.warning(
            "mu(t) stability detection skipped: too few points (%s < %s)",
            len(mu_series),
            window_stability,
        )
        return mu_series.index.min()

    median_mu = float(mu_series.median())
    mad_mu = float(np.median(np.abs(mu_series.to_numpy(dtype=float) - median_mu)))

    if not np.isfinite(mad_mu) or mad_mu == 0.0:
        std_mu = float(mu_series.std())
        if not np.isfinite(std_mu) or std_mu == 0.0:
            LOGGER.warning("mu(t) stability detection fallback: MAD/std are zero, no exclusion applied")
            return mu_series.index.min()
        mu_upper = median_mu + k * std_mu
    else:
        mu_upper = median_mu + k * mad_mu

    stable_mask = (mu_smoothed < mu_upper) & mu_smoothed.notna()
    unstable_mask = (mu_smoothed >= mu_upper) & mu_smoothed.notna()
    stable_values = stable_mask.to_numpy(dtype=bool)
    index_values = stable_mask.index

    first_non_null = mu_smoothed.first_valid_index()
    if not isinstance(first_non_null, pd.Timestamp):
        return None
    first_unstable = unstable_mask[unstable_mask].index.min() if unstable_mask.any() else None

    if isinstance(first_unstable, pd.Timestamp):
        search_start_ts = first_unstable
    else:
        search_start_ts = first_non_null

    start_positions = np.where(index_values >= search_start_ts)[0]
    if len(start_positions) == 0:
        return first_non_null
    search_start_pos = int(start_positions[0])

    run_length = 0
    for idx in range(search_start_pos, len(stable_values)):
        is_stable = stable_values[idx]
        run_length = run_length + 1 if is_stable else 0
        if run_length >= window_stability:
            return index_values[idx - window_stability + 1]

    return first_non_null


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
    death_delay_days: int,
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
    lines.append(f"- death_delay_days: {death_delay_days}")
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
    lines.append(
        f"- mu(t) is estimated with rolling integrated method over window={smoothing_window}: "
        "ΔD / ΣI_lagged."
    )
    lines.append("- Deaths are lagging indicators and still affect mu(t) stability.")
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
    death_delay_days: int = DEFAULT_DEATH_DELAY_DAYS,
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
    if death_delay_days < 0:
        raise ValueError("death_delay_days must be >= 0")

    sigma = estimate_sigma(latent_period_days)
    gamma = estimate_gamma(infectious_period_days)
    LOGGER.info(
        "SEIRD mu(t) estimator: rolling integrated window=%s with death_delay_days=%s",
        smoothing_window,
        death_delay_days,
    )

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
    # Phenomenological delay proxy for deaths relative to infected prevalence.
    enriched["I_lagged_for_death"] = enriched["I_estimated"].astype(float).shift(death_delay_days)

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
        death_delay_days=death_delay_days,
        window=smoothing_window,
        epsilon=epsilon,
        min_infected_threshold=min_infected_threshold,
    )
    enriched["valid_mu_mask"] = mu_valid_mask
    enriched["mu_raw"] = mu_raw
    enriched["mu_smoothed"] = smooth_series(enriched["mu_raw"], smoothing_window)

    valid_start_date = detect_mu_stable_start_date(enriched["mu_smoothed"], window_stability=14, k=8.0)
    if valid_start_date is not None:
        exclusion_mask = enriched.index < valid_start_date
        if exclusion_mask.any():
            enriched.loc[exclusion_mask, "mu_raw"] = np.nan
            enriched.loc[exclusion_mask, "mu_smoothed"] = np.nan
            enriched.loc[exclusion_mask, "valid_mu_mask"] = False
            LOGGER.info("mu(t) exclusion applied before %s using median+MAD method", valid_start_date.date())
    else:
        LOGGER.warning("mu(t) exclusion skipped: no valid start date detected")

    enriched["mu_smoothed_nonnegative"] = enriched["mu_smoothed"].clip(lower=0)

    # Consistency terms for plotting diagnostics.
    enriched["beta_s_i_over_n_smoothed"] = enriched["beta_smoothed"] * enriched["denominator_SI_over_N"]
    enriched["sigma_e_term"] = sigma * enriched["E_estimated"]
    enriched["mu_i_term_smoothed"] = enriched["mu_smoothed"] * enriched["I_lagged_for_death"]

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
        death_delay_days=death_delay_days,
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
    parser.add_argument("--death-delay-days", type=int, default=DEFAULT_DEATH_DELAY_DAYS)
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
        death_delay_days=args.death_delay_days,
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
