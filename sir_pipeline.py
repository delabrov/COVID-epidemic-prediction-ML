"""Independent pipeline for SIR preparation + parameter estimation + plotting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.modeling.estimate_sir_parameters import run_sir_parameter_estimation_pipeline
from src.sir.sir_preparation import (
    DEFAULT_END_DATE,
    DEFAULT_POPULATION_COLUMN,
    DEFAULT_SIGNAL_COLUMN,
    DEFAULT_START_DATE,
    run_sir_preparation_pipeline,
)
from src.visualization.plot_sir_parameters import generate_sir_parameter_plots


def parse_args() -> argparse.Namespace:
    """Parse CLI args for full independent SIR pipeline."""
    parser = argparse.ArgumentParser(description="Run SIR preparation and parameter estimation pipeline")

    parser.add_argument("--country", type=str, default="France", help="Country label")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/analysis/covid_france_analysis_daily.parquet"),
        help="Input dataset path for SIR preparation (parquet/csv)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sir/inputs"),
        help="Directory for SIR prepared datasets",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/processed/reports/sir"),
        help="Directory for SIR preparation reports",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures/sir"),
        help="Directory for SIR preparation figures",
    )
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE, help="Study window start date")
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE, help="Study window end date")
    parser.add_argument(
        "--signal-column",
        type=str,
        default=DEFAULT_SIGNAL_COLUMN,
        help="Incidence signal used for SIR preparation",
    )
    parser.add_argument(
        "--population-column",
        type=str,
        default=DEFAULT_POPULATION_COLUMN,
        help="Population column used in SIR stages",
    )
    parser.add_argument(
        "--infectious-period-days",
        type=int,
        default=14,
        help="Infectious period assumption for SIR initialization and gamma",
    )
    parser.add_argument("--save-csv", action="store_true", help="Also save SIR-prepared dataset as CSV")

    parser.add_argument(
        "--skip-parameter-estimation",
        action="store_true",
        help="Run only SIR preparation stage (steps 1-3)",
    )
    parser.add_argument(
        "--parameters-output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for SIR parameter datasets",
    )
    parser.add_argument(
        "--parameters-reports-dir",
        type=Path,
        default=Path("data/processed/reports/sir"),
        help="Directory for SIR parameter reports",
    )
    parser.add_argument(
        "--parameters-figures-dir",
        type=Path,
        default=Path("outputs/figures/sir"),
        help="Directory for SIR parameter plots",
    )
    parser.add_argument(
        "--beta-smoothing-window",
        type=int,
        default=7,
        help="Smoothing window for beta and reproduction numbers",
    )
    parser.add_argument(
        "--derivative-method",
        type=str,
        choices=["gradient", "diff"],
        default="gradient",
        help="Finite-difference method for dI/dt",
    )
    parser.add_argument(
        "--derivative-smoothing-window",
        type=int,
        default=1,
        help="Optional smoothing window on I_estimated before dI/dt",
    )
    parser.add_argument(
        "--min-infected-threshold",
        type=float,
        default=10.0,
        help="Minimum I(t) threshold for valid beta estimation",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Small numerical threshold for safe division",
    )
    parser.add_argument(
        "--min-denominator",
        type=float,
        default=1.0,
        help="Minimum denominator S*I/N threshold for valid beta estimation",
    )
    parser.add_argument(
        "--save-parameters-csv",
        action="store_true",
        help="Also save SIR parameter dataset as CSV",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    sir_result = run_sir_preparation_pipeline(
        country=args.country,
        input_path=args.input_path,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        signal_column=args.signal_column,
        population_column=args.population_column,
        infectious_period_days=args.infectious_period_days,
        save_csv=args.save_csv,
    )

    if args.skip_parameter_estimation:
        logging.info("Skipping SIR parameter estimation stage (--skip-parameter-estimation)")
        return

    parameter_result = run_sir_parameter_estimation_pipeline(
        country=args.country,
        input_path=sir_result["sir_data_path"],
        output_dir=args.parameters_output_dir,
        reports_dir=args.parameters_reports_dir,
        infectious_period_days=args.infectious_period_days,
        derivative_method=args.derivative_method,
        derivative_smoothing_window=args.derivative_smoothing_window,
        beta_smoothing_window=args.beta_smoothing_window,
        min_infected_threshold=args.min_infected_threshold,
        epsilon=args.epsilon,
        min_denominator=args.min_denominator,
        population_column=args.population_column,
        save_csv=args.save_parameters_csv,
    )

    plot_paths = generate_sir_parameter_plots(
        parameter_result["dataset"],
        output_dir=args.parameters_figures_dir,
        country=args.country,
    )

    logging.info("SIR parameter plots generated: %s", plot_paths)


if __name__ == "__main__":
    main()
