"""Independent pipeline for SEIRDV preparation + parameter estimation + plotting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.modeling.estimate_seirdv_parameters import run_seirdv_parameter_estimation_pipeline
from src.modeling.prepare_seirdv_data import (
    DEFAULT_CASES_SIGNAL,
    DEFAULT_DEATHS_SIGNAL,
    DEFAULT_END_DATE,
    DEFAULT_POPULATION_COLUMN,
    DEFAULT_START_DATE,
    DEFAULT_VACCINE_EFFICACY,
    run_seirdv_preparation_pipeline,
)
from src.visualization.plot_seirdv_parameters import generate_seirdv_parameter_plots


def parse_args() -> argparse.Namespace:
    """Parse CLI args for independent SEIRDV pipeline."""
    parser = argparse.ArgumentParser(description="Run SEIRDV preparation and parameter estimation pipeline")
    parser.add_argument("--country", type=str, default="France", help="Country label")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/analysis/covid_france_analysis_daily.parquet"),
        help="Input dataset path for SEIRDV preparation (parquet/csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for SEIRDV datasets",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/processed/reports/seirdv"),
        help="Directory for SEIRDV reports",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures/seirdv"),
        help="Directory for SEIRDV plots",
    )
    parser.add_argument(
        "--seird-parameter-path",
        type=Path,
        default=Path("data/processed/covid_france_seird_parameters.parquet"),
        help="SEIRD parameter dataset used for SEIRD vs SEIRDV comparison figure",
    )

    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE)
    parser.add_argument("--cases-signal", type=str, default=DEFAULT_CASES_SIGNAL)
    parser.add_argument("--deaths-signal", type=str, default=DEFAULT_DEATHS_SIGNAL)
    parser.add_argument("--population-column", type=str, default=DEFAULT_POPULATION_COLUMN)
    parser.add_argument("--latent-period-days", type=int, default=5)
    parser.add_argument("--infectious-period-days", type=int, default=14)
    parser.add_argument("--death-delay-days", type=int, default=14)
    parser.add_argument("--vaccine-efficacy", type=float, default=DEFAULT_VACCINE_EFFICACY)

    parser.add_argument("--infectivity-profile", type=str, choices=["uniform", "gamma"], default="gamma")
    parser.add_argument("--infectivity-shape", type=float, default=3.0)
    parser.add_argument("--infectivity-scale", type=float, default=2.0)
    parser.add_argument("--latent-profile", type=str, choices=["uniform", "gamma"], default="uniform")
    parser.add_argument("--latent-shape", type=float, default=2.0)
    parser.add_argument("--latent-scale", type=float, default=2.0)

    parser.add_argument("--save-csv", action="store_true", help="Also save SEIRDV-ready and SEIRDV-parameter CSVs")
    parser.add_argument(
        "--skip-parameter-estimation",
        action="store_true",
        help="Run only SEIRDV preparation stage",
    )
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--derivative-method", type=str, choices=["gradient", "diff"], default="gradient")
    parser.add_argument("--derivative-smoothing-window", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--min-infected-threshold", type=float, default=10.0)
    parser.add_argument("--min-exposed-threshold", type=float, default=10.0)
    parser.add_argument("--min-denominator", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    prep_result = run_seirdv_preparation_pipeline(
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

    if args.skip_parameter_estimation:
        logging.info("Skipping SEIRDV parameter estimation stage (--skip-parameter-estimation)")
        return

    parameter_result = run_seirdv_parameter_estimation_pipeline(
        country=args.country,
        input_path=prep_result["output_path"],
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

    plot_paths = generate_seirdv_parameter_plots(
        parameter_result["dataset"],
        output_dir=args.figures_dir,
        country=args.country,
        seird_parameter_path=args.seird_parameter_path,
    )
    logging.info("SEIRDV parameter plots generated: %s", plot_paths)


if __name__ == "__main__":
    main()
