"""Plot SIR parameter estimation outputs."""

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
    """Return filesystem-safe country slug."""
    return re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe is indexed by sorted DatetimeIndex."""
    output = df.copy()
    if isinstance(output.index, pd.DatetimeIndex):
        output.index = pd.to_datetime(output.index, errors="coerce")
    elif "date" in output.columns:
        output["date"] = pd.to_datetime(output["date"], errors="coerce")
        output = output.dropna(subset=["date"]).set_index("date")
    else:
        raise ValueError("Input dataframe must have DatetimeIndex or a 'date' column")

    output = output[~output.index.isna()].sort_index()
    output.index.name = "date"
    return output


def load_sir_parameter_dataset(input_path: Path) -> pd.DataFrame:
    """Load SIR parameter dataset from parquet or CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"SIR parameter dataset not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported dataset extension: {input_path.suffix}")

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
    for column in ("population_for_sir", "population"):
        if column in df.columns:
            return column
    raise ValueError("Population column missing: expected 'population_for_sir' or 'population'")


def plot_beta_estimates(df: pd.DataFrame, output_path: Path, country: str) -> Path:
    """Plot raw and smoothed beta(t) estimates."""
    fig, axis = plt.subplots(figsize=(12, 5))
    if "beta_raw" in df.columns:
        axis.plot(df.index, df["beta_raw"], alpha=0.35, linewidth=1.0, label="beta_raw")
    if "beta_smoothed" in df.columns:
        axis.plot(df.index, df["beta_smoothed"], linewidth=2.0, label="beta_smoothed")

    axis.axhline(0.0, linestyle="--", linewidth=1.0, label="0 reference")
    axis.set_title(f"Estimated Transmission Parameter beta(t) - {country}")
    axis.set_xlabel("Date")
    axis.set_ylabel("beta")
    axis.grid(alpha=0.2)
    _safe_legend(axis)

    return _save_figure(fig, output_path)


def plot_reff_estimates(df: pd.DataFrame, output_path: Path, country: str) -> Path:
    """Plot raw and smoothed effective reproduction number estimates."""
    fig, axis = plt.subplots(figsize=(12, 5))
    if "R_eff_raw" in df.columns:
        axis.plot(df.index, df["R_eff_raw"], alpha=0.35, linewidth=1.0, label="R_eff_raw")
    if "R_eff_smoothed" in df.columns:
        axis.plot(df.index, df["R_eff_smoothed"], linewidth=2.0, label="R_eff_smoothed")

    axis.axhline(1.0, linestyle="--", linewidth=1.0, label="R_eff = 1")
    axis.set_title(f"Estimated Effective Reproduction Number - {country}")
    axis.set_xlabel("Date")
    axis.set_ylabel("R_eff")
    axis.grid(alpha=0.2)
    _safe_legend(axis)

    return _save_figure(fig, output_path)


def plot_dI_dt_consistency(df: pd.DataFrame, output_path: Path, country: str) -> Path:
    """Plot dI/dt consistency terms from SIR equation."""
    if "dI_dt" not in df.columns or "I_estimated" not in df.columns or "gamma" not in df.columns:
        raise ValueError("Required columns missing for dI/dt consistency plot")

    population_column = _resolve_population_column(df)
    beta_column = "beta_smoothed" if "beta_smoothed" in df.columns else "beta_raw"

    denominator = (
        df["denominator_SI_over_N"]
        if "denominator_SI_over_N" in df.columns
        else (df["S_estimated"].astype(float) * df["I_estimated"].astype(float)) / df[population_column].astype(float)
    )

    transmission = df[beta_column] * denominator
    recovery = df["gamma"] * df["I_estimated"]

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(df.index, df["dI_dt"], linewidth=1.5, label="dI_dt")
    axis.plot(df.index, transmission, linewidth=1.5, label=f"{beta_column} * S * I / N")
    axis.plot(df.index, recovery, linewidth=1.5, label="gamma * I_estimated")
    axis.set_title(f"dI/dt Consistency Terms - {country}")
    axis.set_xlabel("Date")
    axis.set_ylabel("People/day")
    axis.grid(alpha=0.2)
    _safe_legend(axis)

    return _save_figure(fig, output_path)


def plot_sir_parameter_summary(df: pd.DataFrame, output_path: Path, country: str) -> Path:
    """Plot multi-panel summary for core SIR parameters and dynamics."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    if "I_estimated" in df.columns:
        axes[0].plot(df.index, df["I_estimated"], label="I_estimated")
    axes[0].set_title(f"I_estimated - {country}")
    axes[0].set_ylabel("People")
    axes[0].grid(alpha=0.2)
    _safe_legend(axes[0])

    if "dI_dt" in df.columns:
        axes[1].plot(df.index, df["dI_dt"], label="dI_dt")
    axes[1].set_title("dI_dt")
    axes[1].set_ylabel("People/day")
    axes[1].grid(alpha=0.2)
    _safe_legend(axes[1])

    if "beta_smoothed" in df.columns:
        axes[2].plot(df.index, df["beta_smoothed"], label="beta_smoothed")
    axes[2].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[2].set_title("beta_smoothed")
    axes[2].set_ylabel("beta")
    axes[2].grid(alpha=0.2)
    _safe_legend(axes[2])

    if "R_eff_smoothed" in df.columns:
        axes[3].plot(df.index, df["R_eff_smoothed"], label="R_eff_smoothed")
    axes[3].axhline(1.0, linestyle="--", linewidth=1.0)
    axes[3].set_title("R_eff_smoothed")
    axes[3].set_ylabel("R_eff")
    axes[3].set_xlabel("Date")
    axes[3].grid(alpha=0.2)
    _safe_legend(axes[3])

    return _save_figure(fig, output_path)


def generate_sir_parameter_plots(df: pd.DataFrame, output_dir: Path, country: str) -> dict[str, Path]:
    """Generate required SIR parameter estimation figures."""
    dataset = _ensure_datetime_index(df)
    country_slug = _slugify_country(country)

    beta_path = output_dir / f"covid_{country_slug}_beta_estimates.png"
    reff_path = output_dir / f"covid_{country_slug}_reff_estimates.png"
    didt_path = output_dir / f"covid_{country_slug}_dI_dt_consistency.png"
    summary_path = output_dir / f"covid_{country_slug}_sir_parameter_summary.png"

    return {
        "beta_plot_path": plot_beta_estimates(dataset, beta_path, country),
        "reff_plot_path": plot_reff_estimates(dataset, reff_path, country),
        "didt_plot_path": plot_dI_dt_consistency(dataset, didt_path, country),
        "summary_plot_path": plot_sir_parameter_summary(dataset, summary_path, country),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for SIR parameter plotting."""
    parser = argparse.ArgumentParser(description="Generate SIR parameter estimation plots")
    parser.add_argument("--country", type=str, default="France", help="Country label for titles")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/covid_france_sir_parameters.parquet"),
        help="Path to enriched SIR-parameter dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory for output figures",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    dataset = load_sir_parameter_dataset(args.input_path)
    generate_sir_parameter_plots(dataset, args.output_dir, country=args.country)


if __name__ == "__main__":
    main()
