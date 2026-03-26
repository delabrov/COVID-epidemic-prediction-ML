"""Estimate SEIRDV parameters from reconstructed SEIRDV state trajectories."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.modeling.estimate_seird_parameters import (
    detect_mu_stable_start_date,
    estimate_derivative,
    estimate_gamma,
    estimate_mu,
    estimate_sigma,
    smooth_series,
)

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
DEFAULT_VACCINE_EFFICACY = 0.6

REQUIRED_COLUMNS: tuple[str, ...] = (
    "S_estimated",
    "V_estimated",
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


def load_seirdv_ready_data(input_path: Path) -> pd.DataFrame:
    """Load SEIRDV-ready dataset (parquet/csv)."""
    if not input_path.exists():
        raise FileNotFoundError(f"SEIRDV-ready dataset not found: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input extension: {input_path.suffix}")
    return _ensure_datetime_index(df)


def validate_required_columns(df: pd.DataFrame, population_column: str) -> None:
    """Validate required columns for SEIRDV parameter estimation."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if population_column not in df.columns:
        missing.append(population_column)
    if missing:
        raise ValueError(f"Missing required SEIRDV columns: {missing}")


def estimate_beta_seirdv(
    df: pd.DataFrame,
    *,
    sigma: float,
    population_column: str,
    vaccine_efficacy: float,
    epsilon: float,
    min_infected_threshold: float,
    min_exposed_threshold: float,
    min_denominator: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Estimate beta(t) from dE/dt = beta*I/N*(S + (1-eps_v)*V) - sigma*E."""
    susceptible = df["S_estimated"].astype(float)
    vaccinated = df["V_estimated"].astype(float)
    exposed = df["E_estimated"].astype(float)
    infected = df["I_estimated"].astype(float)
    population = df[population_column].astype(float)

    effective_susceptible = susceptible + (1.0 - float(vaccine_efficacy)) * vaccinated
    denominator = (infected / population) * effective_susceptible
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
    return beta, denominator, effective_susceptible, valid_mask


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
    """Build summary statistics for key SEIRDV parameters."""
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


def detect_beta_stable_start_date(
    beta_smoothed: pd.Series,
    *,
    window_stability: int = 14,
    k: float = 8.0,
) -> tuple[pd.Timestamp | None, float | None, float | None, float | None]:
    """Detect first stable beta date using median + k*MAD threshold."""
    beta_series = beta_smoothed.replace([np.inf, -np.inf], np.nan).dropna()
    if beta_series.empty:
        LOGGER.warning("beta(t) exclusion skipped: no valid beta_smoothed values")
        return None, None, None, None
    if window_stability <= 0:
        raise ValueError("window_stability must be > 0")
    if len(beta_series) < window_stability:
        LOGGER.warning(
            "beta(t) exclusion skipped: too few points (%s < %s)",
            len(beta_series),
            window_stability,
        )
        return None, None, None, None

    beta_median = float(beta_series.median())
    beta_mad = float(np.median(np.abs(beta_series.to_numpy(dtype=float) - beta_median)))
    if not np.isfinite(beta_mad) or beta_mad == 0.0:
        LOGGER.warning("beta(t) exclusion skipped: invalid MAD value (%s)", beta_mad)
        return None, beta_median, beta_mad, None

    beta_upper = beta_median + float(k) * beta_mad
    stable_mask = (beta_smoothed < beta_upper) & beta_smoothed.notna()
    stable_values = stable_mask.to_numpy(dtype=bool)
    index_values = stable_mask.index

    first_non_null = beta_smoothed.first_valid_index()
    if not isinstance(first_non_null, pd.Timestamp):
        return None, beta_median, beta_mad, beta_upper

    run_length = 0
    for idx, is_stable in enumerate(stable_values):
        if index_values[idx] < first_non_null:
            continue
        run_length = run_length + 1 if is_stable else 0
        if run_length >= window_stability:
            return index_values[idx - window_stability + 1], beta_median, beta_mad, beta_upper

    LOGGER.warning(
        "beta(t) stable window not found; falling back to first non-null beta_smoothed date (%s)",
        first_non_null.date(),
    )
    return first_non_null, beta_median, beta_mad, beta_upper


def _build_parameter_report_text(
    *,
    country: str,
    dataset: pd.DataFrame,
    latent_period_days: int,
    infectious_period_days: int,
    death_delay_days: int,
    vaccine_efficacy: float,
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
    beta_exclusion_start_date: pd.Timestamp | None,
    beta_robust_upper: float | None,
    beta_median: float | None,
    beta_mad: float | None,
) -> str:
    valid_beta = int(dataset["valid_beta_mask"].fillna(False).sum()) if "valid_beta_mask" in dataset.columns else 0
    valid_mu = int(dataset["valid_mu_mask"].fillna(False).sum()) if "valid_mu_mask" in dataset.columns else 0

    lines: list[str] = []
    lines.append("SEIRDV Parameter Estimation Report")
    lines.append("=" * 37)
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
    lines.append(f"- vaccine_efficacy: {vaccine_efficacy:.4f}")
    lines.append(f"- sigma: {sigma:.8f}")
    lines.append(f"- gamma: {gamma:.8f}")
    lines.append(f"- derivative_method: {derivative_method}")
    lines.append(f"- derivative_pre_smoothing_window: {derivative_smoothing_window}")
    lines.append(f"- smoothing_window: {smoothing_window}")
    lines.append(f"- epsilon: {epsilon}")
    lines.append(f"- min_infected_threshold: {min_infected_threshold}")
    lines.append(f"- min_exposed_threshold: {min_exposed_threshold}")
    lines.append(f"- min_denominator_effective_term: {min_denominator}")
    lines.append(f"- population_column: {population_column}")
    lines.append("- beta robust exclusion method: median + MAD (k=8.0, stability_window=14)")
    lines.append(
        f"- beta exclusion start date: {beta_exclusion_start_date.date() if isinstance(beta_exclusion_start_date, pd.Timestamp) else 'not applied'}"
    )
    lines.append(
        f"- beta robust threshold (median + k*MAD): {beta_robust_upper if beta_robust_upper is not None else 'n/a'}"
    )
    lines.append(f"- beta median: {beta_median if beta_median is not None else 'n/a'}")
    lines.append(f"- beta MAD: {beta_mad if beta_mad is not None else 'n/a'}")
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
    lines.append("- V_estimated is directly observation-driven from vaccination stock columns.")
    lines.append("- vaccine_efficacy is fixed, not inferred from data in this stage.")
    lines.append("- Reconstructed compartments are proxies, not directly observed states.")
    lines.append(
        f"- mu(t) uses rolling integrated estimation over window={smoothing_window} with median+MAD exclusion."
    )
    lines.append("- Effective parameters should be interpreted as model-consistent proxies.")
    lines.append("")
    return "\n".join(lines)


def run_seirdv_parameter_estimation_pipeline(
    *,
    country: str,
    input_path: Path,
    output_dir: Path,
    reports_dir: Path,
    latent_period_days: int = DEFAULT_LATENT_PERIOD_DAYS,
    infectious_period_days: int = DEFAULT_INFECTIOUS_PERIOD_DAYS,
    death_delay_days: int = DEFAULT_DEATH_DELAY_DAYS,
    vaccine_efficacy: float = DEFAULT_VACCINE_EFFICACY,
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
    """Run SEIRDV parameter estimation and save enriched dataset/report."""
    if not (0.0 <= float(vaccine_efficacy) <= 1.0):
        raise ValueError("vaccine_efficacy must be within [0, 1]")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    country_slug = _slugify_country(country)
    dataset = load_seirdv_ready_data(input_path)
    validate_required_columns(dataset, population_column=population_column)
    if death_delay_days < 0:
        raise ValueError("death_delay_days must be >= 0")

    sigma = estimate_sigma(latent_period_days)
    gamma = estimate_gamma(infectious_period_days)
    LOGGER.info(
        "SEIRDV mu(t) estimator: rolling integrated window=%s with death_delay_days=%s",
        smoothing_window,
        death_delay_days,
    )

    enriched = dataset.copy()
    enriched["sigma"] = sigma
    enriched["gamma"] = gamma
    enriched["vaccine_efficacy"] = float(vaccine_efficacy)

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
    enriched["I_lagged_for_death"] = enriched["I_estimated"].astype(float).shift(death_delay_days)

    beta_raw, denominator, effective_susceptible, beta_valid_mask = estimate_beta_seirdv(
        enriched,
        sigma=sigma,
        population_column=population_column,
        vaccine_efficacy=vaccine_efficacy,
        epsilon=epsilon,
        min_infected_threshold=min_infected_threshold,
        min_exposed_threshold=min_exposed_threshold,
        min_denominator=min_denominator,
    )

    enriched["effective_susceptible_equivalent"] = effective_susceptible
    enriched["denominator_effective_svi_over_n"] = denominator
    enriched["valid_beta_mask"] = beta_valid_mask
    enriched["beta_raw"] = beta_raw
    enriched["beta_smoothed"] = smooth_series(enriched["beta_raw"], smoothing_window)
    beta_start_date, beta_median, beta_mad, beta_upper = detect_beta_stable_start_date(
        enriched["beta_smoothed"],
        window_stability=14,
        k=8.0,
    )
    beta_exclusion_start_date = beta_start_date
    if beta_start_date is not None:
        # Add a small post-start buffer to avoid rolling-window edge artifacts.
        beta_exclusion_start_date = beta_start_date + pd.Timedelta(days=max(smoothing_window - 1, 0))
        beta_exclusion_mask = enriched.index < beta_exclusion_start_date
        if beta_exclusion_mask.any():
            enriched.loc[beta_exclusion_mask, "beta_raw"] = np.nan
            enriched.loc[beta_exclusion_mask, "valid_beta_mask"] = False
            LOGGER.info(
                "beta(t) exclusion applied before %s using median+MAD method",
                beta_exclusion_start_date.date(),
            )

    # Recompute smoothed beta after exclusion to avoid contamination from unstable early values.
    enriched["beta_smoothed"] = smooth_series(enriched["beta_raw"], smoothing_window)
    if beta_exclusion_start_date is not None:
        enriched.loc[enriched.index < beta_exclusion_start_date, "beta_smoothed"] = np.nan
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

    enriched["beta_effective_contact_term_smoothed"] = (
        enriched["beta_smoothed"] * enriched["denominator_effective_svi_over_n"]
    )
    enriched["sigma_e_term"] = sigma * enriched["E_estimated"]
    enriched["mu_i_term_smoothed"] = enriched["mu_smoothed"] * enriched["I_lagged_for_death"]

    denom_raw = gamma + enriched["mu_raw"]
    denom_smooth = gamma + enriched["mu_smoothed"]
    safe_raw = denom_raw.where(denom_raw.abs() > epsilon)
    safe_smooth = denom_smooth.where(denom_smooth.abs() > epsilon)

    effective_fraction = enriched["effective_susceptible_equivalent"] / enriched[population_column]
    enriched["R0_proxy_raw"] = enriched["beta_raw"] / safe_raw
    enriched["R0_proxy_smoothed"] = enriched["beta_smoothed"] / safe_smooth
    enriched["R_eff_proxy_raw"] = enriched["R0_proxy_raw"] * effective_fraction
    enriched["R_eff_proxy_smoothed"] = enriched["R0_proxy_smoothed"] * effective_fraction

    output_path = output_dir / f"covid_{country_slug}_seirdv_parameters.parquet"
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
        vaccine_efficacy=vaccine_efficacy,
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
        beta_exclusion_start_date=beta_exclusion_start_date,
        beta_robust_upper=beta_upper,
        beta_median=beta_median,
        beta_mad=beta_mad,
    )
    report_path = reports_dir / f"covid_{country_slug}_seirdv_parameter_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\n=== SEIRDV Parameter Estimation Summary ===")
    print(report_text)

    LOGGER.info("SEIRDV parameter dataset saved: %s", output_path)
    LOGGER.info("SEIRDV parameter report saved: %s", report_path)

    return {
        "dataset": enriched,
        "output_path": output_path,
        "csv_output_path": csv_output_path,
        "report_path": report_path,
        "report_text": report_text,
        "sigma": sigma,
        "gamma": gamma,
        "vaccine_efficacy": vaccine_efficacy,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for SEIRDV parameter estimation."""
    parser = argparse.ArgumentParser(description="Estimate SEIRDV parameters from reconstructed states")
    parser.add_argument("--country", type=str, default="France", help="Country label")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/covid_france_seirdv_ready.parquet"),
        help="Path to SEIRDV-ready dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for SEIRDV parameter dataset",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/processed/reports/seirdv"),
        help="Directory for SEIRDV parameter report",
    )
    parser.add_argument("--latent-period-days", type=int, default=DEFAULT_LATENT_PERIOD_DAYS)
    parser.add_argument("--infectious-period-days", type=int, default=DEFAULT_INFECTIOUS_PERIOD_DAYS)
    parser.add_argument("--death-delay-days", type=int, default=DEFAULT_DEATH_DELAY_DAYS)
    parser.add_argument("--vaccine-efficacy", type=float, default=DEFAULT_VACCINE_EFFICACY)
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
    run_seirdv_parameter_estimation_pipeline(
        country=args.country,
        input_path=args.input_path,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        latent_period_days=args.latent_period_days,
        infectious_period_days=args.infectious_period_days,
        death_delay_days=args.death_delay_days,
        vaccine_efficacy=args.vaccine_efficacy,
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
