"""Generate SEIRD analysis plots from reconstructed states and estimated parameters."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _slugify_country(country: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure sorted DatetimeIndex named 'date'."""
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


def load_seird_parameter_dataset(input_path: Path) -> pd.DataFrame:
    """Load SEIRD parameter dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"SEIRD parameter dataset not found: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input extension: {input_path.suffix}")
    return _ensure_datetime_index(df)


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    LOGGER.info("Saved figure: %s", output_path)
    return output_path


def _safe_legend(axis: plt.Axes) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if handles and labels:
        axis.legend()


def _resolve_population_column(df: pd.DataFrame) -> str:
    for column in ("population", "population_for_sir"):
        if column in df.columns:
            return column
    raise ValueError("Population column missing in SEIRD parameter dataset")


def plot_seird_states(df: pd.DataFrame, output_path: Path, country: str) -> Path:
    """Plot reconstructed SEIRD compartments normalized by population on log scale."""
    population_column = _resolve_population_column(df)
    pop = df[population_column].astype(float)

    fig, axis = plt.subplots(figsize=(12, 5))
    for column in ["S_estimated", "E_estimated", "I_estimated", "R_estimated", "D_estimated"]:
        if column in df.columns:
            normalized = (df[column] / pop).astype(float)
            # Log-scale plotting requires strictly positive values.
            normalized = normalized.where(normalized > 0)
            axis.plot(df.index, normalized, linewidth=1.5, label=f"{column}/N")

    axis.set_title(f"Reconstructed SEIRD States (normalized, log scale) - {country}")
    axis.set_ylabel("Fraction of population (log scale)")
    axis.set_xlabel("Date")
    axis.set_yscale("log")
    axis.grid(alpha=0.2)
    _safe_legend(axis)
    return _save_figure(fig, output_path)


def plot_seird_beta(df: pd.DataFrame, output_path: Path, country: str) -> Path:
    """Plot raw and smoothed beta estimates."""
    fig, axis = plt.subplots(figsize=(12, 5))
    if "beta_raw" in df.columns:
        axis.plot(df.index, df["beta_raw"], alpha=0.35, linewidth=1.0, label="beta_raw")
    if "beta_smoothed" in df.columns:
        axis.plot(df.index, df["beta_smoothed"], linewidth=2.0, label="beta_smoothed")
    axis.axhline(0.0, linestyle="--", linewidth=1.0)
    axis.set_title(f"Estimated SEIRD beta(t) - {country}")
    axis.set_xlabel("Date")
    axis.set_ylabel("beta")
    axis.grid(alpha=0.2)
    _safe_legend(axis)
    return _save_figure(fig, output_path)


def plot_seird_mu(df: pd.DataFrame, output_path: Path, country: str) -> Path:
    """Plot raw and smoothed mu estimates."""
    fig, axis = plt.subplots(figsize=(12, 5))
    if "mu_raw" in df.columns:
        axis.plot(df.index, df["mu_raw"], alpha=0.35, linewidth=1.0, label="mu_raw")
    if "mu_smoothed" in df.columns:
        axis.plot(df.index, df["mu_smoothed"], linewidth=2.0, label="mu_smoothed")

        valid_start = df["mu_smoothed"].first_valid_index()
        if isinstance(valid_start, pd.Timestamp) and valid_start > df.index.min():
            axis.axvspan(
                df.index.min(),
                valid_start,
                alpha=0.3,
                color="red",
                label="Excluded unstable region",
            )

    axis.axhline(0.0, linestyle="--", linewidth=1.0)
    axis.set_title(f"Estimated SEIRD mu(t) - {country}")
    axis.set_xlabel("Date")
    axis.set_ylabel("mu")
    axis.grid(alpha=0.2)
    _safe_legend(axis)
    return _save_figure(fig, output_path)


def plot_seird_consistency(df: pd.DataFrame, output_path: Path, country: str) -> Path:
    """Plot consistency terms for dE/dt and dD/dt identities."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if "dE_dt" in df.columns:
        axes[0].plot(df.index, df["dE_dt"], linewidth=1.4, label="dE_dt")
    if "beta_s_i_over_n_smoothed" in df.columns:
        axes[0].plot(df.index, df["beta_s_i_over_n_smoothed"], linewidth=1.4, label="beta*S*I/N (smoothed)")
    if "sigma_e_term" in df.columns:
        axes[0].plot(df.index, df["sigma_e_term"], linewidth=1.4, label="sigma*E")
    axes[0].set_title(f"SEIRD Exposed Equation Consistency - {country}")
    axes[0].set_ylabel("People/day")
    axes[0].grid(alpha=0.2)
    _safe_legend(axes[0])

    if "dD_dt" in df.columns:
        axes[1].plot(df.index, df["dD_dt"], linewidth=1.4, label="dD_dt")
    if "mu_i_term_smoothed" in df.columns:
        axes[1].plot(df.index, df["mu_i_term_smoothed"], linewidth=1.4, label="mu*I_lagged (smoothed)")
    axes[1].set_title("SEIRD Death Equation Consistency")
    axes[1].set_ylabel("People/day")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.2)
    _safe_legend(axes[1])

    return _save_figure(fig, output_path)


def plot_seird_reff_proxy(df: pd.DataFrame, output_path: Path, country: str) -> Path | None:
    """Plot raw and smoothed R_eff proxy when available."""
    if "R_eff_proxy_raw" not in df.columns and "R_eff_proxy_smoothed" not in df.columns:
        LOGGER.warning("Skipping R_eff proxy plot: required columns not found")
        return None

    fig, axis = plt.subplots(figsize=(12, 5))
    if "R_eff_proxy_raw" in df.columns:
        axis.plot(df.index, df["R_eff_proxy_raw"], alpha=0.35, linewidth=1.0, label="R_eff_proxy_raw")
    if "R_eff_proxy_smoothed" in df.columns:
        axis.plot(df.index, df["R_eff_proxy_smoothed"], linewidth=2.0, label="R_eff_proxy_smoothed")
    axis.axhline(1.0, linestyle="--", linewidth=1.0)
    axis.set_title(f"SEIRD Effective Reproduction Proxy - {country}")
    axis.set_xlabel("Date")
    axis.set_ylabel("R_eff_proxy")
    axis.grid(alpha=0.2)
    _safe_legend(axis)
    return _save_figure(fig, output_path)


def plot_seird_summary(df: pd.DataFrame, output_path: Path, country: str) -> Path:
    """Plot multi-panel SEIRD summary."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    if "E_estimated" in df.columns:
        axes[0].plot(df.index, df["E_estimated"], label="E_estimated")
    axes[0].set_title(f"E_estimated - {country}")
    axes[0].set_ylabel("People")
    axes[0].grid(alpha=0.2)
    _safe_legend(axes[0])

    if "I_estimated" in df.columns:
        axes[1].plot(df.index, df["I_estimated"], label="I_estimated")
    axes[1].set_title("I_estimated")
    axes[1].set_ylabel("People")
    axes[1].grid(alpha=0.2)
    _safe_legend(axes[1])

    if "beta_smoothed" in df.columns:
        axes[2].plot(df.index, df["beta_smoothed"], label="beta_smoothed")
    axes[2].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[2].set_title("beta_smoothed")
    axes[2].set_ylabel("beta")
    axes[2].grid(alpha=0.2)
    _safe_legend(axes[2])

    if "mu_smoothed" in df.columns:
        axes[3].plot(df.index, df["mu_smoothed"], label="mu_smoothed")
    axes[3].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[3].set_title("mu_smoothed")
    axes[3].set_ylabel("mu")
    axes[3].grid(alpha=0.2)
    _safe_legend(axes[3])

    if "R_eff_proxy_smoothed" in df.columns:
        axes[4].plot(df.index, df["R_eff_proxy_smoothed"], label="R_eff_proxy_smoothed")
    axes[4].axhline(1.0, linestyle="--", linewidth=1.0)
    axes[4].set_title("R_eff_proxy_smoothed")
    axes[4].set_ylabel("R_eff_proxy")
    axes[4].set_xlabel("Date")
    axes[4].grid(alpha=0.2)
    _safe_legend(axes[4])

    return _save_figure(fig, output_path)


def _align_series_for_metrics(observed: pd.Series, reconstructed: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align and filter finite non-null points for metric computation."""
    mask = observed.notna() & reconstructed.notna() & np.isfinite(observed) & np.isfinite(reconstructed)
    return observed.loc[mask], reconstructed.loc[mask]


def _compute_fit_metrics(observed: pd.Series, reconstructed: pd.Series) -> tuple[float, float, float]:
    """Compute R², RMSE and MAE on aligned points."""
    aligned_obs, aligned_rec = _align_series_for_metrics(observed, reconstructed)
    if aligned_obs.empty:
        return float("nan"), float("nan"), float("nan")

    residuals = aligned_obs - aligned_rec
    mse = float(np.mean(np.square(residuals)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))

    ss_res = float(np.sum(np.square(residuals)))
    centered = aligned_obs - float(np.mean(aligned_obs))
    ss_tot = float(np.sum(np.square(centered)))
    r2 = float("nan") if ss_tot <= 0.0 else 1.0 - (ss_res / ss_tot)
    return r2, rmse, mae


def _plot_flow_comparison_and_residuals(
    *,
    index: pd.Index,
    observed: pd.Series,
    reconstructed: pd.Series,
    output_path: Path,
    title: str,
    observed_label: str,
    reconstructed_label: str,
) -> Path | None:
    residuals = observed - reconstructed
    aligned_obs, aligned_rec = _align_series_for_metrics(observed, reconstructed)
    if aligned_obs.empty:
        LOGGER.warning("Skipping flow comparison plot (%s): no aligned non-null points", title)
        return None

    r2, rmse, mae = _compute_fit_metrics(observed, reconstructed)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    top_axis, residual_axis = axes

    top_axis.plot(index, observed, linewidth=1.5, label=observed_label)
    top_axis.plot(index, reconstructed, linewidth=1.5, color="red", label=reconstructed_label)
    top_axis.set_title(title)
    top_axis.set_ylabel("People/day")
    top_axis.grid(alpha=0.2)
    _safe_legend(top_axis)

    rmse_display = str(int(round(rmse))) if np.isfinite(rmse) else "nan"
    mae_display = str(int(round(mae))) if np.isfinite(mae) else "nan"
    metrics_text = f"R²: {r2:.3f}\nRMSE: {rmse_display}\nMAE: {mae_display}"
    residual_axis.text(
        0.02,
        0.98,
        metrics_text,
        transform=residual_axis.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    residual_axis.plot(
        index,
        residuals,
        linewidth=1.2,
        color="black",
        label="Residuals (observed - reconstructed)",
    )
    residual_axis.axhline(0.0, linestyle="--", linewidth=1.0, color="gray")
    residual_axis.set_ylabel("Residuals")
    residual_axis.set_xlabel("Date")
    residual_axis.grid(alpha=0.2)
    _safe_legend(residual_axis)
    return _save_figure(fig, output_path)


def _plot_residual_histogram(
    residuals: pd.Series,
    output_path: Path,
    *,
    title: str,
    bins: int,
    central_quantile_range: tuple[float, float] | None = None,
) -> Path | None:
    clean_residuals = residuals.replace([np.inf, -np.inf], np.nan).dropna()
    if clean_residuals.empty:
        LOGGER.warning("Skipping residual histogram (%s): no valid residuals", title)
        return None

    histogram_values = clean_residuals
    if central_quantile_range is not None:
        low_q, high_q = central_quantile_range
        if 0.0 <= low_q < high_q <= 1.0:
            lower = float(clean_residuals.quantile(low_q))
            upper = float(clean_residuals.quantile(high_q))
            if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                histogram_values = clean_residuals[(clean_residuals >= lower) & (clean_residuals <= upper)]
                if histogram_values.empty:
                    histogram_values = clean_residuals

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.hist(histogram_values, bins=bins, alpha=0.85, color="#4d4d4d")
    axis.axvline(0.0, linestyle="--", linewidth=1.2, color="black")
    axis.set_title(title)
    axis.set_xlabel("Residual")
    axis.set_ylabel("Frequency")
    axis.grid(alpha=0.2)
    return _save_figure(fig, output_path)


def plot_seird_observed_vs_reconstructed_cases(df: pd.DataFrame, output_path: Path, country: str) -> Path | None:
    """Plot observed vs reconstructed cases with residuals."""
    required_columns = ["new_cases_7d_avg", "sigma", "E_estimated"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        LOGGER.warning("Skipping cases comparison plot: missing columns %s", missing)
        return None

    observed = df["new_cases_7d_avg"].astype(float)
    reconstructed = df["sigma"].astype(float) * df["E_estimated"].astype(float)
    return _plot_flow_comparison_and_residuals(
        index=df.index,
        observed=observed,
        reconstructed=reconstructed,
        output_path=output_path,
        title=f"Observed vs reconstructed case flow - {country}",
        observed_label="Observed cases (new_cases_7d_avg)",
        reconstructed_label="Reconstructed cases (sigma*E_estimated)",
    )


def plot_seird_observed_vs_reconstructed_deaths(df: pd.DataFrame, output_path: Path, country: str) -> Path | None:
    """Plot observed vs reconstructed deaths with residuals."""
    infected_column = "I_lagged_for_death" if "I_lagged_for_death" in df.columns else "I_estimated"
    required_columns = ["new_deaths_7d_avg", "mu_smoothed", infected_column]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        LOGGER.warning("Skipping deaths comparison plot: missing columns %s", missing)
        return None

    observed = df["new_deaths_7d_avg"].astype(float)
    reconstructed = df["mu_smoothed"].astype(float) * df[infected_column].astype(float)
    return _plot_flow_comparison_and_residuals(
        index=df.index,
        observed=observed,
        reconstructed=reconstructed,
        output_path=output_path,
        title=f"Observed vs reconstructed death flow - {country}",
        observed_label="Observed deaths (new_deaths_7d_avg)",
        reconstructed_label=f"Reconstructed deaths (mu*{infected_column})",
    )


def plot_seird_cases_residual_histogram(df: pd.DataFrame, output_path: Path, country: str) -> Path | None:
    """Plot histogram of case residuals."""
    required_columns = ["new_cases_7d_avg", "sigma", "E_estimated"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        LOGGER.warning("Skipping cases residual histogram: missing columns %s", missing)
        return None

    residuals = df["new_cases_7d_avg"].astype(float) - (
        df["sigma"].astype(float) * df["E_estimated"].astype(float)
    )
    return _plot_residual_histogram(
        residuals,
        output_path,
        title=f"Cases residual histogram - {country}",
        bins=55,
        central_quantile_range=(0.01, 0.99),
    )


def plot_seird_deaths_residual_histogram(df: pd.DataFrame, output_path: Path, country: str) -> Path | None:
    """Plot histogram of death residuals."""
    infected_column = "I_lagged_for_death" if "I_lagged_for_death" in df.columns else "I_estimated"
    required_columns = ["new_deaths_7d_avg", "mu_smoothed", infected_column]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        LOGGER.warning("Skipping deaths residual histogram: missing columns %s", missing)
        return None

    residuals = df["new_deaths_7d_avg"].astype(float) - (
        df["mu_smoothed"].astype(float) * df[infected_column].astype(float)
    )
    return _plot_residual_histogram(
        residuals,
        output_path,
        title=f"Deaths residual histogram - {country}",
        bins=80,
    )


def generate_seird_parameter_plots(df: pd.DataFrame, output_dir: Path, country: str) -> dict[str, Path | None]:
    """Generate full SEIRD analysis plot set."""
    dataset = _ensure_datetime_index(df)
    slug = _slugify_country(country)

    states_path = output_dir / f"covid_{slug}_seird_states.png"
    beta_path = output_dir / f"covid_{slug}_seird_beta_estimates.png"
    mu_path = output_dir / f"covid_{slug}_seird_mu_estimates.png"
    consistency_path = output_dir / f"covid_{slug}_seird_consistency.png"
    reff_path = output_dir / f"covid_{slug}_seird_reff_proxy.png"
    summary_path = output_dir / f"covid_{slug}_seird_parameter_summary.png"
    observed_vs_reconstructed_cases_path = (
        output_dir / f"covid_{slug}_seird_observed_vs_reconstructed_cases.png"
    )
    observed_vs_reconstructed_deaths_path = (
        output_dir / f"covid_{slug}_seird_observed_vs_reconstructed_deaths.png"
    )
    cases_residual_histogram_path = output_dir / f"covid_{slug}_seird_cases_residual_histogram.png"
    deaths_residual_histogram_path = output_dir / f"covid_{slug}_seird_deaths_residual_histogram.png"

    return {
        "states_plot_path": plot_seird_states(dataset, states_path, country),
        "beta_plot_path": plot_seird_beta(dataset, beta_path, country),
        "mu_plot_path": plot_seird_mu(dataset, mu_path, country),
        "consistency_plot_path": plot_seird_consistency(dataset, consistency_path, country),
        "reff_proxy_plot_path": plot_seird_reff_proxy(dataset, reff_path, country),
        "summary_plot_path": plot_seird_summary(dataset, summary_path, country),
        "observed_vs_reconstructed_cases_plot_path": plot_seird_observed_vs_reconstructed_cases(
            dataset,
            observed_vs_reconstructed_cases_path,
            country,
        ),
        "observed_vs_reconstructed_deaths_plot_path": plot_seird_observed_vs_reconstructed_deaths(
            dataset,
            observed_vs_reconstructed_deaths_path,
            country,
        ),
        "cases_residual_histogram_plot_path": plot_seird_cases_residual_histogram(
            dataset,
            cases_residual_histogram_path,
            country,
        ),
        "deaths_residual_histogram_plot_path": plot_seird_deaths_residual_histogram(
            dataset,
            deaths_residual_histogram_path,
            country,
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for SEIRD plotting."""
    parser = argparse.ArgumentParser(description="Generate SEIRD parameter analysis plots")
    parser.add_argument("--country", type=str, default="France")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/covid_france_seird_parameters.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/seird"),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    dataset = load_seird_parameter_dataset(args.input_path)
    generate_seird_parameter_plots(dataset, args.output_dir, country=args.country)


if __name__ == "__main__":
    main()
