"""Generate SEIRD analysis plots from reconstructed states and estimated parameters."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    """Plot reconstructed SEIRD compartments in absolute and normalized scales."""
    population_column = _resolve_population_column(df)
    pop = df[population_column].astype(float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for column in ["S_estimated", "E_estimated", "I_estimated", "R_estimated", "D_estimated"]:
        if column in df.columns:
            axes[0].plot(df.index, df[column], linewidth=1.5, label=column)
    axes[0].set_title(f"Reconstructed SEIRD States - {country}")
    axes[0].set_ylabel("People")
    axes[0].grid(alpha=0.2)
    _safe_legend(axes[0])

    for column in ["S_estimated", "E_estimated", "I_estimated", "R_estimated", "D_estimated"]:
        if column in df.columns:
            axes[1].plot(df.index, df[column] / pop, linewidth=1.5, label=f"{column}/N")
    axes[1].set_title("Reconstructed SEIRD States (normalized by population)")
    axes[1].set_ylabel("Fraction of population")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.2)
    _safe_legend(axes[1])
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
        axes[1].plot(df.index, df["mu_i_term_smoothed"], linewidth=1.4, label="mu*I (smoothed)")
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

    return {
        "states_plot_path": plot_seird_states(dataset, states_path, country),
        "beta_plot_path": plot_seird_beta(dataset, beta_path, country),
        "mu_plot_path": plot_seird_mu(dataset, mu_path, country),
        "consistency_plot_path": plot_seird_consistency(dataset, consistency_path, country),
        "reff_proxy_plot_path": plot_seird_reff_proxy(dataset, reff_path, country),
        "summary_plot_path": plot_seird_summary(dataset, summary_path, country),
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
        default=Path("outputs/figures"),
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
