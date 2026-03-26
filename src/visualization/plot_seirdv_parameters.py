"""Generate SEIRDV analysis plots from reconstructed states and estimated parameters."""

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

FRANCE_LOCKDOWN_PERIODS: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = (
    (pd.Timestamp("2020-03-17"), pd.Timestamp("2020-05-11")),
    (pd.Timestamp("2020-10-30"), pd.Timestamp("2020-12-15")),
    (pd.Timestamp("2021-04-03"), pd.Timestamp("2021-05-03")),
)


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


def load_dataset(input_path: Path, *, label: str) -> pd.DataFrame:
    """Load parameter dataset from parquet/csv."""
    if not input_path.exists():
        raise FileNotFoundError(f"{label} dataset not found: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input extension: {input_path.suffix}")
    return _ensure_datetime_index(df)


def load_seirdv_parameter_dataset(input_path: Path) -> pd.DataFrame:
    """Load SEIRDV parameter dataset."""
    return load_dataset(input_path, label="SEIRDV parameter")


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


def _add_france_lockdown_shading(axis: plt.Axes, index: pd.DatetimeIndex) -> None:
    """Shade French lockdown periods with a fill-between band."""
    if len(index) == 0:
        return
    y_min, y_max = axis.get_ylim()
    y_low = np.full(len(index), y_min)
    y_high = np.full(len(index), y_max)

    has_label = False
    for start, end in FRANCE_LOCKDOWN_PERIODS:
        mask = (index >= start) & (index <= end)
        if np.any(mask):
            axis.fill_between(
                index,
                y_low,
                y_high,
                where=mask,
                color="gray",
                alpha=0.18,
                label="Périodes de confinement (France)" if not has_label else None,
            )
            has_label = True
    axis.set_ylim(y_min, y_max)


def _resolve_population_column(df: pd.DataFrame) -> str:
    for column in ("population", "population_for_sir"):
        if column in df.columns:
            return column
    raise ValueError("Population column missing in SEIRDV parameter dataset")


def plot_seirdv_states(df: pd.DataFrame, output_path: Path, country: str) -> Path:
    """Plot reconstructed SEIRDV compartments in absolute values and normalized fractions."""
    population_column = _resolve_population_column(df)
    pop = df[population_column].astype(float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    columns = ["S_estimated", "V_estimated", "E_estimated", "I_estimated", "R_estimated", "D_estimated"]

    for column in columns:
        if column in df.columns:
            axes[0].plot(df.index, df[column].astype(float), linewidth=1.3, label=column)
    axes[0].set_title(f"Reconstructed SEIRDV States (absolute) - {country}")
    axes[0].set_ylabel("People")
    axes[0].grid(alpha=0.2)
    _safe_legend(axes[0])

    for column in columns:
        if column in df.columns:
            normalized = (df[column] / pop).astype(float)
            normalized = normalized.where(normalized > 0)
            axes[1].plot(df.index, normalized, linewidth=1.3, label=f"{column}/N")
    axes[1].set_title("Reconstructed SEIRDV States (normalized, log scale)")
    axes[1].set_ylabel("Fraction of population (log)")
    axes[1].set_xlabel("Date")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.2)
    _safe_legend(axes[1])
    return _save_figure(fig, output_path)


def plot_seirdv_vaccination_flow(df: pd.DataFrame, output_path: Path, country: str) -> Path | None:
    """Plot vaccination stock and estimated vaccination flows."""
    required = ["V_estimated", "nu_flow_raw", "nu_flow_smoothed"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        LOGGER.warning("Skipping SEIRDV vaccination flow plot: missing columns %s", missing)
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df.index, df["V_estimated"], linewidth=1.5, label="V_estimated")
    axes[0].set_title(f"Vaccinated Compartment V(t) - {country}")
    axes[0].set_ylabel("People")
    axes[0].grid(alpha=0.2)
    _safe_legend(axes[0])

    axes[1].plot(df.index, df["nu_flow_raw"], linewidth=1.2, alpha=0.4, label="nu_flow_raw")
    axes[1].plot(df.index, df["nu_flow_smoothed"], linewidth=1.7, label="nu_flow_smoothed")
    axes[1].set_title("Vaccination Flow Estimates")
    axes[1].set_ylabel("People/day")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.2)
    _safe_legend(axes[1])
    return _save_figure(fig, output_path)


def plot_seirdv_beta(df: pd.DataFrame, output_path: Path, country: str) -> Path:
    """Plot raw and smoothed beta estimates."""
    fig, axis = plt.subplots(figsize=(12, 5))
    if "beta_raw" in df.columns:
        axis.plot(df.index, df["beta_raw"], alpha=0.35, linewidth=1.0, label="beta_raw")
    if "beta_smoothed" in df.columns:
        axis.plot(df.index, df["beta_smoothed"], linewidth=2.0, label="beta_smoothed")
        valid_start = df["beta_smoothed"].first_valid_index()
        if isinstance(valid_start, pd.Timestamp) and valid_start > df.index.min():
            axis.axvspan(
                df.index.min(),
                valid_start,
                alpha=0.3,
                color="red",
                label="Excluded unstable region",
            )
            y_min, y_max = axis.get_ylim()
            y_text = y_max - 0.08 * (y_max - y_min)
            midpoint = df.index.min() + (valid_start - df.index.min()) / 2
            axis.text(
                midpoint,
                y_text,
                "Initial unstable phase\n(beta not reliable)",
                color="red",
                ha="center",
                va="top",
                fontsize=9,
            )
    axis.axhline(0.0, linestyle="--", linewidth=1.0)
    axis.set_title(f"Estimated SEIRDV beta(t) - {country}")
    axis.set_xlabel("Date")
    axis.set_ylabel("beta")
    axis.grid(alpha=0.2)
    _safe_legend(axis)
    return _save_figure(fig, output_path)


def plot_seirdv_mu(df: pd.DataFrame, output_path: Path, country: str) -> Path:
    """Plot raw and smoothed mu estimates with excluded unstable region."""
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
    axis.set_title(f"Estimated SEIRDV mu(t) - {country}")
    axis.set_xlabel("Date")
    axis.set_ylabel("mu")
    axis.grid(alpha=0.2)
    _safe_legend(axis)
    return _save_figure(fig, output_path)


def plot_seirdv_reff_proxy(df: pd.DataFrame, output_path: Path, country: str) -> Path | None:
    """Plot raw and smoothed R_eff proxy when available."""
    if "R_eff_proxy_raw" not in df.columns and "R_eff_proxy_smoothed" not in df.columns:
        LOGGER.warning("Skipping SEIRDV R_eff proxy plot: required columns not found")
        return None

    fig, axis = plt.subplots(figsize=(12, 5))
    if "R_eff_proxy_raw" in df.columns:
        axis.plot(df.index, df["R_eff_proxy_raw"], alpha=0.35, linewidth=1.0, label="R_eff_proxy_raw")
    if "R_eff_proxy_smoothed" in df.columns:
        axis.plot(df.index, df["R_eff_proxy_smoothed"], linewidth=2.0, label="R_eff_proxy_smoothed")
    axis.axhline(1.0, linestyle="--", linewidth=1.0)
    axis.set_title(f"SEIRDV Effective Reproduction Proxy - {country}")
    axis.set_xlabel("Date")
    axis.set_ylabel("R_eff_proxy")
    axis.grid(alpha=0.2)
    _safe_legend(axis)
    return _save_figure(fig, output_path)


def plot_seirdv_summary(df: pd.DataFrame, output_path: Path, country: str) -> Path:
    """Plot multi-panel SEIRDV summary."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    if "V_estimated" in df.columns:
        axes[0].plot(df.index, df["V_estimated"], label="V_estimated")
    axes[0].set_title(f"V_estimated - {country}")
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
    aligned_obs, _ = _align_series_for_metrics(observed, reconstructed)
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


def plot_seirdv_observed_vs_reconstructed_cases(df: pd.DataFrame, output_path: Path, country: str) -> Path | None:
    """Plot observed vs reconstructed cases with residuals."""
    required_columns = ["new_cases_7d_avg", "sigma", "E_estimated"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        LOGGER.warning("Skipping SEIRDV cases comparison plot: missing columns %s", missing)
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


def plot_seirdv_observed_vs_reconstructed_deaths(df: pd.DataFrame, output_path: Path, country: str) -> Path | None:
    """Plot observed vs reconstructed deaths with residuals."""
    infected_column = "I_lagged_for_death" if "I_lagged_for_death" in df.columns else "I_estimated"
    required_columns = ["new_deaths_7d_avg", "mu_smoothed", infected_column]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        LOGGER.warning("Skipping SEIRDV deaths comparison plot: missing columns %s", missing)
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


def plot_seirdv_observed_vs_reconstructed_with_lockdowns(
    df: pd.DataFrame,
    output_path: Path,
    country: str,
) -> Path | None:
    """Plot observed cases/deaths in one figure with French lockdown shading."""
    required_columns = ["new_cases_7d_avg", "new_deaths_7d_avg"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        LOGGER.warning("Skipping SEIRDV lockdown comparison plot: missing columns %s", missing)
        return None

    observed_cases = df["new_cases_7d_avg"].astype(float)
    observed_deaths = df["new_deaths_7d_avg"].astype(float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df.index, observed_cases, linewidth=1.5, label="Observed cases")
    _add_france_lockdown_shading(axes[0], pd.DatetimeIndex(df.index))
    axes[0].set_title(f"Observed cases with lockdown periods - {country}")
    axes[0].set_ylabel("People/day")
    axes[0].grid(alpha=0.2)
    _safe_legend(axes[0])

    axes[1].plot(df.index, observed_deaths, linewidth=1.5, label="Observed deaths")
    _add_france_lockdown_shading(axes[1], pd.DatetimeIndex(df.index))
    axes[1].set_title("Observed deaths with lockdown periods")
    axes[1].set_ylabel("People/day")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.2)
    _safe_legend(axes[1])

    return _save_figure(fig, output_path)


def plot_seirdv_cases_residual_histogram(df: pd.DataFrame, output_path: Path, country: str) -> Path | None:
    """Plot histogram of case residuals."""
    required_columns = ["new_cases_7d_avg", "sigma", "E_estimated"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        LOGGER.warning("Skipping SEIRDV cases residual histogram: missing columns %s", missing)
        return None

    residuals = df["new_cases_7d_avg"].astype(float) - (
        df["sigma"].astype(float) * df["E_estimated"].astype(float)
    )
    return _plot_residual_histogram(
        residuals,
        output_path,
        title=f"SEIRDV cases residual histogram - {country}",
        bins=55,
        central_quantile_range=(0.01, 0.99),
    )


def plot_seirdv_deaths_residual_histogram(df: pd.DataFrame, output_path: Path, country: str) -> Path | None:
    """Plot histogram of death residuals."""
    infected_column = "I_lagged_for_death" if "I_lagged_for_death" in df.columns else "I_estimated"
    required_columns = ["new_deaths_7d_avg", "mu_smoothed", infected_column]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        LOGGER.warning("Skipping SEIRDV deaths residual histogram: missing columns %s", missing)
        return None

    residuals = df["new_deaths_7d_avg"].astype(float) - (
        df["mu_smoothed"].astype(float) * df[infected_column].astype(float)
    )
    return _plot_residual_histogram(
        residuals,
        output_path,
        title=f"SEIRDV deaths residual histogram - {country}",
        bins=80,
    )


def _extract_lag_weights(df: pd.DataFrame, *, prefix: str) -> tuple[np.ndarray, np.ndarray] | None:
    lag_pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    lag_columns: list[tuple[int, str]] = []
    for column in df.columns:
        match = lag_pattern.match(column)
        if match:
            lag_columns.append((int(match.group(1)), column))
    if not lag_columns:
        return None

    lag_columns.sort(key=lambda item: item[0])
    lags = np.array([item[0] for item in lag_columns], dtype=int)
    first_row = df.iloc[0]
    weights = np.array([float(first_row[item[1]]) for item in lag_columns], dtype=float)
    return lags, weights


def plot_seirdv_profiles(df: pd.DataFrame, output_path: Path, country: str) -> Path | None:
    """Plot infectivity and latent profile weights over lag days."""
    infectivity = _extract_lag_weights(df, prefix="infectivity_weight_lag_")
    latent = _extract_lag_weights(df, prefix="latent_weight_lag_")
    if infectivity is None or latent is None:
        LOGGER.warning("Skipping SEIRDV profile plot: missing infectivity or latent weight columns")
        return None

    lags_i, weights_i = infectivity
    lags_e, weights_e = latent

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    axes[0].bar(lags_i, weights_i, width=0.8, alpha=0.9)
    axes[0].set_title(f"SEIRDV Infectivity Profile Weights - {country}")
    axes[0].set_ylabel("Weight")
    axes[0].set_xlabel("Lag day")
    axes[0].grid(alpha=0.2)

    axes[1].bar(lags_e, weights_e, width=0.8, alpha=0.9)
    axes[1].set_title("SEIRDV Latent Profile Weights")
    axes[1].set_ylabel("Weight")
    axes[1].set_xlabel("Lag day")
    axes[1].grid(alpha=0.2)
    return _save_figure(fig, output_path)


def plot_seird_vs_seirdv_comparison(
    seirdv_df: pd.DataFrame,
    *,
    seird_parameter_path: Path,
    output_path: Path,
    country: str,
) -> Path | None:
    """Plot SEIRD vs SEIRDV comparison of key trajectories and parameters."""
    if not seird_parameter_path.exists():
        LOGGER.warning("Skipping SEIRD vs SEIRDV comparison: missing SEIRD dataset at %s", seird_parameter_path)
        return None

    try:
        seird_df = load_dataset(seird_parameter_path, label="SEIRD parameter")
    except Exception as exc:  # pragma: no cover - defensive branch
        LOGGER.warning("Skipping SEIRD vs SEIRDV comparison: failed to load SEIRD dataset (%s)", exc)
        return None

    required = ["I_estimated", "beta_smoothed", "mu_smoothed", "R_eff_proxy_smoothed"]
    missing_seird = [column for column in required if column not in seird_df.columns]
    missing_seirdv = [column for column in required if column not in seirdv_df.columns]
    if missing_seird or missing_seirdv:
        LOGGER.warning(
            "Skipping SEIRD vs SEIRDV comparison: missing columns (SEIRD=%s, SEIRDV=%s)",
            missing_seird,
            missing_seirdv,
        )
        return None

    common_index = seird_df.index.intersection(seirdv_df.index)
    if len(common_index) == 0:
        LOGGER.warning("Skipping SEIRD vs SEIRDV comparison: no common dates")
        return None

    seird = seird_df.loc[common_index]
    seirdv = seirdv_df.loc[common_index]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(common_index, seird["I_estimated"], linewidth=1.4, label="SEIRD I_estimated")
    axes[0].plot(common_index, seirdv["I_estimated"], linewidth=1.4, label="SEIRDV I_estimated")
    axes[0].set_title(f"SEIRD vs SEIRDV comparison - {country}")
    axes[0].set_ylabel("I (people)")
    axes[0].grid(alpha=0.2)
    _safe_legend(axes[0])

    axes[1].plot(common_index, seird["beta_smoothed"], linewidth=1.4, label="SEIRD beta_smoothed")
    axes[1].plot(common_index, seirdv["beta_smoothed"], linewidth=1.4, label="SEIRDV beta_smoothed")
    axes[1].set_ylabel("beta")
    axes[1].grid(alpha=0.2)
    _safe_legend(axes[1])

    axes[2].plot(common_index, seird["mu_smoothed"], linewidth=1.4, label="SEIRD mu_smoothed")
    axes[2].plot(common_index, seirdv["mu_smoothed"], linewidth=1.4, label="SEIRDV mu_smoothed")
    axes[2].set_ylabel("mu")
    axes[2].grid(alpha=0.2)
    _safe_legend(axes[2])

    axes[3].plot(common_index, seird["R_eff_proxy_smoothed"], linewidth=1.4, label="SEIRD R_eff_proxy_smoothed")
    axes[3].plot(common_index, seirdv["R_eff_proxy_smoothed"], linewidth=1.4, label="SEIRDV R_eff_proxy_smoothed")
    axes[3].axhline(1.0, linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("R_eff_proxy")
    axes[3].set_xlabel("Date")
    axes[3].grid(alpha=0.2)
    _safe_legend(axes[3])

    return _save_figure(fig, output_path)


def generate_seirdv_parameter_plots(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    country: str,
    seird_parameter_path: Path | None = None,
) -> dict[str, Path | None]:
    """Generate full SEIRDV analysis plot set."""
    dataset = _ensure_datetime_index(df)
    slug = _slugify_country(country)

    states_path = output_dir / f"covid_{slug}_seirdv_states.png"
    vaccination_flow_path = output_dir / f"covid_{slug}_seirdv_vaccination_flow.png"
    beta_path = output_dir / f"covid_{slug}_seirdv_beta_estimates.png"
    mu_path = output_dir / f"covid_{slug}_seirdv_mu_estimates.png"
    reff_path = output_dir / f"covid_{slug}_seirdv_reff_proxy.png"
    summary_path = output_dir / f"covid_{slug}_seirdv_parameter_summary.png"
    observed_vs_reconstructed_cases_path = output_dir / f"covid_{slug}_seirdv_observed_vs_reconstructed_cases.png"
    observed_vs_reconstructed_deaths_path = output_dir / f"covid_{slug}_seirdv_observed_vs_reconstructed_deaths.png"
    observed_vs_reconstructed_lockdowns_path = (
        output_dir / f"covid_{slug}_seirdv_observed_vs_reconstructed_lockdowns.png"
    )
    cases_residual_histogram_path = output_dir / f"covid_{slug}_seirdv_cases_residual_histogram.png"
    deaths_residual_histogram_path = output_dir / f"covid_{slug}_seirdv_deaths_residual_histogram.png"
    profiles_path = output_dir / f"covid_{slug}_seirdv_profiles.png"
    comparison_path = output_dir / f"covid_{slug}_seirdv_compartment_comparison.png"

    return {
        "states_plot_path": plot_seirdv_states(dataset, states_path, country),
        "vaccination_flow_plot_path": plot_seirdv_vaccination_flow(dataset, vaccination_flow_path, country),
        "beta_plot_path": plot_seirdv_beta(dataset, beta_path, country),
        "mu_plot_path": plot_seirdv_mu(dataset, mu_path, country),
        "reff_proxy_plot_path": plot_seirdv_reff_proxy(dataset, reff_path, country),
        "summary_plot_path": plot_seirdv_summary(dataset, summary_path, country),
        "observed_vs_reconstructed_cases_plot_path": plot_seirdv_observed_vs_reconstructed_cases(
            dataset,
            observed_vs_reconstructed_cases_path,
            country,
        ),
        "observed_vs_reconstructed_deaths_plot_path": plot_seirdv_observed_vs_reconstructed_deaths(
            dataset,
            observed_vs_reconstructed_deaths_path,
            country,
        ),
        "observed_vs_reconstructed_lockdowns_plot_path": plot_seirdv_observed_vs_reconstructed_with_lockdowns(
            dataset,
            observed_vs_reconstructed_lockdowns_path,
            country,
        ),
        "cases_residual_histogram_plot_path": plot_seirdv_cases_residual_histogram(
            dataset,
            cases_residual_histogram_path,
            country,
        ),
        "deaths_residual_histogram_plot_path": plot_seirdv_deaths_residual_histogram(
            dataset,
            deaths_residual_histogram_path,
            country,
        ),
        "profiles_plot_path": plot_seirdv_profiles(dataset, profiles_path, country),
        "comparison_plot_path": plot_seird_vs_seirdv_comparison(
            dataset,
            seird_parameter_path=(
                seird_parameter_path if seird_parameter_path is not None else Path("data/processed/covid_france_seird_parameters.parquet")
            ),
            output_path=comparison_path,
            country=country,
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for SEIRDV plotting."""
    parser = argparse.ArgumentParser(description="Generate SEIRDV parameter analysis plots")
    parser.add_argument("--country", type=str, default="France")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/covid_france_seirdv_parameters.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/seirdv"),
    )
    parser.add_argument(
        "--seird-parameter-path",
        type=Path,
        default=Path("data/processed/covid_france_seird_parameters.parquet"),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    dataset = load_seirdv_parameter_dataset(args.input_path)
    generate_seirdv_parameter_plots(
        dataset,
        output_dir=args.output_dir,
        country=args.country,
        seird_parameter_path=args.seird_parameter_path,
    )


if __name__ == "__main__":
    main()
