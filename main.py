"""Run the full epidemiological data pipeline: download, preprocess, visualize."""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.download_data import download_owid_data
from src.data.preprocess_data import (
    build_output_filename,
    extract_country_data,
    load_raw_data,
    preprocess_country_data,
    print_data_quality_report,
    save_processed_data,
)
from src.data.prepare_analysis_data import run_analysis_data_preparation
from src.diagnostics.validate_timeseries import run_timeseries_diagnostics
from src.modeling.estimate_seird_parameters import run_seird_parameter_estimation_pipeline
from src.modeling.estimate_sir_parameters import run_sir_parameter_estimation_pipeline
from src.modeling.prepare_seird_data import (
    DEFAULT_CASES_SIGNAL as SEIRD_DEFAULT_CASES_SIGNAL,
    DEFAULT_DEATHS_SIGNAL as SEIRD_DEFAULT_DEATHS_SIGNAL,
    DEFAULT_END_DATE as SEIRD_DEFAULT_END_DATE,
    DEFAULT_POPULATION_COLUMN as SEIRD_DEFAULT_POPULATION_COLUMN,
    DEFAULT_START_DATE as SEIRD_DEFAULT_START_DATE,
    run_seird_preparation_pipeline,
)
from src.sir.sir_preparation import (
    DEFAULT_END_DATE as SIR_DEFAULT_END_DATE,
    DEFAULT_POPULATION_COLUMN as SIR_DEFAULT_POPULATION_COLUMN,
    DEFAULT_SIGNAL_COLUMN as SIR_DEFAULT_SIGNAL_COLUMN,
    DEFAULT_START_DATE as SIR_DEFAULT_START_DATE,
    run_sir_preparation_pipeline,
)
from src.visualization.plot_data import generate_all_plots
from src.visualization.plot_seird_parameters import generate_seird_parameter_plots
from src.visualization.plot_sir_parameters import generate_sir_parameter_plots

LOGGER = logging.getLogger(__name__)


def _slugify_country(country: str) -> str:
    """Build filesystem-safe country slug."""
    return re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")


def save_pipeline_terminal_output_report(
    *,
    country: str,
    reports_dir: Path,
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    diagnostics_result: dict[str, Any],
    analysis_result: dict[str, Any] | None,
    sir_preparation_result: dict[str, Any] | None,
    sir_parameters_result: dict[str, Any] | None,
    sir_plots_result: dict[str, Path] | None,
    seird_preparation_result: dict[str, Any] | None,
    seird_parameters_result: dict[str, Any] | None,
    seird_plots_result: dict[str, Path | None] | None,
    data_quality_report_text: str,
    processed_output_path: Path,
    figures_dir: Path,
) -> Path:
    """Save terminal-like pipeline outputs and diagnostics into a TXT report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    country_slug = _slugify_country(country)
    report_path = reports_dir / f"covid_{country_slug}_pipeline_run_output.txt"

    lines: list[str] = []
    lines.append("COVID Pipeline Run Output")
    lines.append("=" * 40)
    lines.append(f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Country: {country}")
    lines.append("")
    lines.append("Run summary")
    lines.append(f"- Raw dataset rows: {len(raw_df)}")
    lines.append(f"- Raw dataset columns: {len(raw_df.columns)}")
    lines.append(f"- Processed rows: {len(processed_df)}")
    lines.append(f"- Processed columns: {len(processed_df.columns)}")
    lines.append(f"- Processed parquet: {processed_output_path}")
    lines.append(f"- Figures directory: {figures_dir}")
    if analysis_result is not None:
        lines.append(f"- Analysis daily dataset: {analysis_result.get('daily_output_path')}")
        lines.append(f"- Analysis weekly dataset: {analysis_result.get('weekly_output_path')}")
    if sir_preparation_result is not None:
        lines.append(f"- SIR prepared dataset: {sir_preparation_result.get('sir_data_path')}")
    if sir_parameters_result is not None:
        lines.append(f"- SIR parameter dataset: {sir_parameters_result.get('output_path')}")
    if seird_preparation_result is not None:
        lines.append(f"- SEIRD prepared dataset: {seird_preparation_result.get('output_path')}")
    if seird_parameters_result is not None:
        lines.append(f"- SEIRD parameter dataset: {seird_parameters_result.get('output_path')}")
    lines.append("")
    lines.append("Diagnostics artifacts")
    lines.append(f"- Validation report: {diagnostics_result.get('report_path')}")
    lines.append(f"- Coverage CSV: {diagnostics_result.get('coverage_path')}")
    lines.append(f"- Cleaning impact CSV: {diagnostics_result.get('cleaning_path')}")
    lines.append(f"- Missing dates CSV: {diagnostics_result.get('missing_dates_path')}")
    if analysis_result is not None:
        lines.append(f"- Study-window report: {analysis_result.get('report_path')}")
        lines.append(f"- Study-window segments CSV: {analysis_result.get('segments_path')}")
        lines.append(f"- Study-window coverage CSV: {analysis_result.get('coverage_path')}")
        lines.append(f"- Study-window plot: {analysis_result.get('window_plot_path')}")
        lines.append(f"- Missingness heatmap: {analysis_result.get('missingness_plot_path')}")
        lines.append(f"- Daily vs weekly plot: {analysis_result.get('daily_weekly_plot_path')}")
        lines.append(f"- Normalized trends plot: {analysis_result.get('normalized_plot_path')}")
    if sir_preparation_result is not None:
        lines.append(f"- SIR framework report: {sir_preparation_result.get('framework_path')}")
        lines.append(f"- SIR initial conditions report: {sir_preparation_result.get('initial_conditions_path')}")
        lines.append(f"- SIR preparation report: {sir_preparation_result.get('preparation_report_path')}")
    if sir_parameters_result is not None:
        lines.append(f"- SIR parameter report: {sir_parameters_result.get('report_path')}")
    if sir_plots_result is not None:
        lines.append(f"- SIR beta plot: {sir_plots_result.get('beta_plot_path')}")
        lines.append(f"- SIR R_eff plot: {sir_plots_result.get('reff_plot_path')}")
        lines.append(f"- SIR dI/dt consistency plot: {sir_plots_result.get('didt_plot_path')}")
        lines.append(f"- SIR summary plot: {sir_plots_result.get('summary_plot_path')}")
    if seird_preparation_result is not None:
        lines.append(f"- SEIRD preparation report: {seird_preparation_result.get('report_path')}")
    if seird_parameters_result is not None:
        lines.append(f"- SEIRD parameter report: {seird_parameters_result.get('report_path')}")
    if seird_plots_result is not None:
        lines.append(f"- SEIRD states plot: {seird_plots_result.get('states_plot_path')}")
        lines.append(f"- SEIRD beta plot: {seird_plots_result.get('beta_plot_path')}")
        lines.append(f"- SEIRD mu plot: {seird_plots_result.get('mu_plot_path')}")
        lines.append(f"- SEIRD consistency plot: {seird_plots_result.get('consistency_plot_path')}")
        lines.append(f"- SEIRD R_eff proxy plot: {seird_plots_result.get('reff_proxy_plot_path')}")
        lines.append(f"- SEIRD summary plot: {seird_plots_result.get('summary_plot_path')}")
    lines.append("")
    lines.append("Temporal Validation Report")
    lines.append("-" * 40)
    lines.append(str(diagnostics_result.get("report_text", "")))
    lines.append("")
    lines.append("Data Quality Report")
    lines.append("-" * 40)
    lines.append(data_quality_report_text)
    lines.append("")
    if analysis_result is not None:
        lines.append("Study Window Report")
        lines.append("-" * 40)
        lines.append(str(analysis_result.get("report_text", "")))
        lines.append("")
    if sir_preparation_result is not None:
        lines.append("SIR Preparation Report")
        lines.append("-" * 40)
        lines.append(str(sir_preparation_result.get("report_text", "")))
        lines.append("")
    if sir_parameters_result is not None:
        lines.append("SIR Parameter Estimation Report")
        lines.append("-" * 40)
        lines.append(str(sir_parameters_result.get("report_text", "")))
        lines.append("")
    if seird_preparation_result is not None:
        lines.append("SEIRD Preparation Report")
        lines.append("-" * 40)
        lines.append(str(seird_preparation_result.get("report_text", "")))
        lines.append("")
    if seird_parameters_result is not None:
        lines.append("SEIRD Parameter Estimation Report")
        lines.append("-" * 40)
        lines.append(str(seird_parameters_result.get("report_text", "")))
        lines.append("")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")

    LOGGER.info("Pipeline run output report saved: %s", report_path)
    return report_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the full pipeline."""
    parser = argparse.ArgumentParser(description="OWID epidemiological analysis pipeline")
    parser.add_argument("--country", type=str, default="France", help="Country to analyze")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of OWID raw data",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw/owid_covid_data.csv"),
        help="Path to raw OWID CSV",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed output",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/processed/reports"),
        help="Directory for diagnostics reports",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory for generated figures",
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        default=Path("data/processed/analysis"),
        help="Directory for analysis-ready datasets",
    )
    parser.add_argument(
        "--analysis-figures-dir",
        type=Path,
        default=Path("outputs/figures/analysis"),
        help="Directory for analysis diagnostic figures",
    )
    parser.add_argument(
        "--analysis-min-row-coverage",
        type=float,
        default=0.5,
        help="Minimum row coverage ratio for study-window selection",
    )
    parser.add_argument(
        "--analysis-min-window-days",
        type=int,
        default=180,
        help="Minimum preferred study-window length in days",
    )
    parser.add_argument(
        "--skip-analysis-prep",
        action="store_true",
        help="Skip study-window selection and analysis dataset preparation",
    )
    parser.add_argument(
        "--analysis-save-csv",
        action="store_true",
        help="Also save analysis daily/weekly outputs as CSV",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save processed dataset as CSV",
    )
    parser.add_argument(
        "--skip-sir-stages",
        action="store_true",
        help="Skip SIR preparation and SIR parameter estimation stages",
    )
    parser.add_argument(
        "--sir-input-path",
        type=Path,
        default=None,
        help="Optional input path for SIR preparation (defaults to analysis daily output)",
    )
    parser.add_argument(
        "--sir-output-dir",
        type=Path,
        default=Path("data/sir/inputs"),
        help="Directory for SIR-prepared datasets",
    )
    parser.add_argument(
        "--sir-reports-dir",
        type=Path,
        default=Path("data/processed/reports/sir"),
        help="Directory for SIR preparation reports",
    )
    parser.add_argument(
        "--sir-figures-dir",
        type=Path,
        default=Path("outputs/figures/sir"),
        help="Directory for SIR preparation plots",
    )
    parser.add_argument(
        "--sir-start-date",
        type=str,
        default=SIR_DEFAULT_START_DATE,
        help="SIR study window start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--sir-end-date",
        type=str,
        default=SIR_DEFAULT_END_DATE,
        help="SIR study window end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--sir-signal-column",
        type=str,
        default=SIR_DEFAULT_SIGNAL_COLUMN,
        help="Signal column used for SIR preparation",
    )
    parser.add_argument(
        "--sir-population-column",
        type=str,
        default=SIR_DEFAULT_POPULATION_COLUMN,
        help="Population column used for SIR stages",
    )
    parser.add_argument(
        "--infectious-period-days",
        type=int,
        default=14,
        help="Infectious period assumption for gamma = 1 / infectious_period_days",
    )
    parser.add_argument(
        "--sir-save-csv",
        action="store_true",
        help="Also save SIR-prepared dataset as CSV",
    )
    parser.add_argument(
        "--sir-parameters-output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for SIR parameter estimation dataset",
    )
    parser.add_argument(
        "--sir-parameters-reports-dir",
        type=Path,
        default=Path("data/processed/reports/sir"),
        help="Directory for SIR parameter estimation report",
    )
    parser.add_argument(
        "--sir-parameters-figures-dir",
        type=Path,
        default=Path("outputs/figures/sir"),
        help="Directory for SIR parameter estimation plots",
    )
    parser.add_argument(
        "--sir-parameters-save-csv",
        action="store_true",
        help="Also save SIR parameter estimation dataset as CSV",
    )
    parser.add_argument(
        "--beta-smoothing-window",
        type=int,
        default=7,
        help="Rolling window used to smooth beta and reproduction estimates",
    )
    parser.add_argument(
        "--derivative-method",
        type=str,
        choices=["gradient", "diff"],
        default="gradient",
        help="Method used for dI/dt estimation",
    )
    parser.add_argument(
        "--derivative-smoothing-window",
        type=int,
        default=1,
        help="Optional smoothing window for I_estimated before differentiation",
    )
    parser.add_argument(
        "--min-infected-threshold",
        type=float,
        default=10.0,
        help="Minimum I(t) threshold for valid beta(t) estimation",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Numerical epsilon used for safe division",
    )
    parser.add_argument(
        "--min-denominator",
        type=float,
        default=1.0,
        help="Minimum S*I/N denominator threshold for valid beta(t) estimation",
    )
    parser.add_argument(
        "--skip-seird-stages",
        action="store_true",
        help="Skip SEIRD preparation and parameter-estimation stages",
    )
    parser.add_argument(
        "--seird-input-path",
        type=Path,
        default=None,
        help="Optional input path for SEIRD preparation (defaults to analysis daily output)",
    )
    parser.add_argument(
        "--seird-output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for SEIRD datasets",
    )
    parser.add_argument(
        "--seird-reports-dir",
        type=Path,
        default=Path("data/processed/reports/seird"),
        help="Directory for SEIRD reports",
    )
    parser.add_argument(
        "--seird-figures-dir",
        type=Path,
        default=Path("outputs/figures/seird"),
        help="Directory for SEIRD plots",
    )
    parser.add_argument(
        "--seird-start-date",
        type=str,
        default=SEIRD_DEFAULT_START_DATE,
        help="SEIRD study window start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--seird-end-date",
        type=str,
        default=SEIRD_DEFAULT_END_DATE,
        help="SEIRD study window end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--seird-cases-signal",
        type=str,
        default=SEIRD_DEFAULT_CASES_SIGNAL,
        help="SEIRD incidence signal column",
    )
    parser.add_argument(
        "--seird-deaths-signal",
        type=str,
        default=SEIRD_DEFAULT_DEATHS_SIGNAL,
        help="SEIRD deaths signal column",
    )
    parser.add_argument(
        "--seird-population-column",
        type=str,
        default=SEIRD_DEFAULT_POPULATION_COLUMN,
        help="SEIRD population column",
    )
    parser.add_argument(
        "--latent-period-days",
        type=int,
        default=5,
        help="Latent period assumption for SEIRD sigma = 1/latent_period_days",
    )
    parser.add_argument(
        "--seird-save-csv",
        action="store_true",
        help="Also save SEIRD-ready and SEIRD-parameters datasets as CSV",
    )
    parser.add_argument(
        "--seird-derivative-method",
        type=str,
        choices=["gradient", "diff"],
        default="gradient",
        help="Derivative method used for SEIRD dE/dt and dD/dt",
    )
    parser.add_argument(
        "--seird-derivative-smoothing-window",
        type=int,
        default=1,
        help="Optional smoothing window for SEIRD compartments before differentiation",
    )
    parser.add_argument(
        "--seird-smoothing-window",
        type=int,
        default=7,
        help="Smoothing window for SEIRD beta and mu",
    )
    parser.add_argument(
        "--seird-epsilon",
        type=float,
        default=1e-8,
        help="Numerical epsilon for SEIRD safe-division rules",
    )
    parser.add_argument(
        "--seird-min-infected-threshold",
        type=float,
        default=10.0,
        help="Minimum I threshold for SEIRD parameter validity",
    )
    parser.add_argument(
        "--seird-min-exposed-threshold",
        type=float,
        default=10.0,
        help="Minimum E threshold for SEIRD beta validity",
    )
    parser.add_argument(
        "--seird-min-denominator",
        type=float,
        default=1.0,
        help="Minimum S*I/N threshold for SEIRD beta validity",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> Path:
    """Execute full pipeline and return processed dataset path."""
    download_owid_data(output_path=args.raw_path, force=args.force_download)

    raw_df = load_raw_data(args.raw_path)
    raw_country_df = extract_country_data(raw_df, country=args.country)
    processed_df = preprocess_country_data(raw_df, country=args.country, raw_country_df=raw_country_df)

    diagnostics_result = run_timeseries_diagnostics(
        country=args.country,
        raw_country_df=raw_country_df,
        processed_df=processed_df,
        reports_dir=args.reports_dir,
    )

    analysis_result: dict[str, Any] | None = None
    if not args.skip_analysis_prep:
        analysis_result = run_analysis_data_preparation(
            country=args.country,
            processed_df=processed_df,
            output_dir=args.analysis_output_dir,
            reports_dir=args.reports_dir,
            figures_dir=args.analysis_figures_dir,
            min_row_coverage=args.analysis_min_row_coverage,
            min_window_days=args.analysis_min_window_days,
            save_csv=args.analysis_save_csv,
        )

    output_path = args.processed_dir / build_output_filename(args.country)
    save_processed_data(processed_df, output_path, save_csv=args.save_csv)
    data_quality_report_text = print_data_quality_report(processed_df)

    sir_preparation_result: dict[str, Any] | None = None
    sir_parameters_result: dict[str, Any] | None = None
    sir_plots_result: dict[str, Path] | None = None
    seird_preparation_result: dict[str, Any] | None = None
    seird_parameters_result: dict[str, Any] | None = None
    seird_plots_result: dict[str, Path | None] | None = None

    if not args.skip_sir_stages:
        if args.sir_input_path is not None:
            sir_input_path = args.sir_input_path
        elif analysis_result is not None:
            sir_input_path = Path(analysis_result["daily_output_path"])
        else:
            sir_input_path = output_path

        sir_preparation_result = run_sir_preparation_pipeline(
            country=args.country,
            input_path=sir_input_path,
            output_dir=args.sir_output_dir,
            reports_dir=args.sir_reports_dir,
            figures_dir=args.sir_figures_dir,
            start_date=args.sir_start_date,
            end_date=args.sir_end_date,
            signal_column=args.sir_signal_column,
            population_column=args.sir_population_column,
            infectious_period_days=args.infectious_period_days,
            save_csv=args.sir_save_csv,
        )

        sir_parameters_result = run_sir_parameter_estimation_pipeline(
            country=args.country,
            input_path=Path(sir_preparation_result["sir_data_path"]),
            output_dir=args.sir_parameters_output_dir,
            reports_dir=args.sir_parameters_reports_dir,
            infectious_period_days=args.infectious_period_days,
            derivative_method=args.derivative_method,
            derivative_smoothing_window=args.derivative_smoothing_window,
            beta_smoothing_window=args.beta_smoothing_window,
            min_infected_threshold=args.min_infected_threshold,
            epsilon=args.epsilon,
            min_denominator=args.min_denominator,
            population_column=args.sir_population_column,
            save_csv=args.sir_parameters_save_csv,
        )

        sir_plots_result = generate_sir_parameter_plots(
            sir_parameters_result["dataset"],
            output_dir=args.sir_parameters_figures_dir,
            country=args.country,
        )
    else:
        LOGGER.info("Skipping SIR stages (--skip-sir-stages)")

    if not args.skip_seird_stages:
        if args.seird_input_path is not None:
            seird_input_path = args.seird_input_path
        elif analysis_result is not None:
            seird_input_path = Path(analysis_result["daily_output_path"])
        else:
            seird_input_path = output_path

        seird_preparation_result = run_seird_preparation_pipeline(
            country=args.country,
            input_path=seird_input_path,
            output_dir=args.seird_output_dir,
            reports_dir=args.seird_reports_dir,
            start_date=args.seird_start_date,
            end_date=args.seird_end_date,
            cases_signal=args.seird_cases_signal,
            deaths_signal=args.seird_deaths_signal,
            population_column=args.seird_population_column,
            latent_period_days=args.latent_period_days,
            infectious_period_days=args.infectious_period_days,
            save_csv=args.seird_save_csv,
        )

        seird_parameters_result = run_seird_parameter_estimation_pipeline(
            country=args.country,
            input_path=Path(seird_preparation_result["output_path"]),
            output_dir=args.seird_output_dir,
            reports_dir=args.seird_reports_dir,
            latent_period_days=args.latent_period_days,
            infectious_period_days=args.infectious_period_days,
            derivative_method=args.seird_derivative_method,
            derivative_smoothing_window=args.seird_derivative_smoothing_window,
            smoothing_window=args.seird_smoothing_window,
            epsilon=args.seird_epsilon,
            min_infected_threshold=args.seird_min_infected_threshold,
            min_exposed_threshold=args.seird_min_exposed_threshold,
            min_denominator=args.seird_min_denominator,
            population_column=args.seird_population_column,
            save_csv=args.seird_save_csv,
        )

        seird_plots_result = generate_seird_parameter_plots(
            seird_parameters_result["dataset"],
            output_dir=args.seird_figures_dir,
            country=args.country,
        )
    else:
        LOGGER.info("Skipping SEIRD stages (--skip-seird-stages)")

    generate_all_plots(processed_df, args.figures_dir, country=args.country)
    save_pipeline_terminal_output_report(
        country=args.country,
        reports_dir=args.reports_dir,
        raw_df=raw_df,
        processed_df=processed_df,
        diagnostics_result=diagnostics_result,
        analysis_result=analysis_result,
        sir_preparation_result=sir_preparation_result,
        sir_parameters_result=sir_parameters_result,
        sir_plots_result=sir_plots_result,
        seird_preparation_result=seird_preparation_result,
        seird_parameters_result=seird_parameters_result,
        seird_plots_result=seird_plots_result,
        data_quality_report_text=data_quality_report_text,
        processed_output_path=output_path,
        figures_dir=args.figures_dir,
    )

    LOGGER.info("Pipeline completed successfully.")
    LOGGER.info("Processed dataset: %s", output_path)
    LOGGER.info("Figures directory: %s", args.figures_dir)

    return output_path


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
