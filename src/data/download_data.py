"""Download OWID COVID-19 data and store it locally."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import requests

OWID_COVID_DATA_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
OWID_FALLBACK_DATA_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
)

LOGGER = logging.getLogger(__name__)


def download_owid_data(
    output_path: Path,
    *,
    force: bool = False,
    url: str = OWID_COVID_DATA_URL,
    timeout_seconds: int = 120,
) -> Path:
    """Download the OWID COVID-19 CSV dataset.

    Args:
        output_path: Destination CSV file path.
        force: If True, re-download even if the file already exists.
        url: Dataset URL.
        timeout_seconds: HTTP timeout for the request.

    Returns:
        Path to the downloaded or reused CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        LOGGER.info("Raw data already exists. Reusing file: %s", output_path)
        _log_dataset_shape(output_path)
        return output_path

    candidate_urls = _build_candidate_urls(url)
    LOGGER.info("Attempting dataset download from %s source(s)", len(candidate_urls))
    errors: list[str] = []

    for candidate_url in candidate_urls:
        try:
            LOGGER.info("Downloading OWID dataset from %s", candidate_url)
            _download_file(candidate_url, output_path, timeout_seconds=timeout_seconds)
            LOGGER.info("Dataset saved to: %s", output_path)
            _log_dataset_shape(output_path)
            return output_path
        except requests.RequestException as exc:
            error_message = f"{candidate_url} -> {exc.__class__.__name__}: {exc}"
            errors.append(error_message)
            LOGGER.warning("Download failed for %s", candidate_url)
            LOGGER.warning("Reason: %s", exc)

    joined_errors = "\n".join(f"- {err}" for err in errors)
    raise ConnectionError(
        "Unable to download OWID dataset from all configured URLs.\n"
        "Likely cause: DNS/network/proxy issue in the environment.\n"
        f"Tried:\n{joined_errors}\n"
        "Workaround: download the CSV manually and place it at data/raw/owid_covid_data.csv."
    )


def _build_candidate_urls(primary_url: str) -> list[str]:
    """Build URL candidates list with fallback while preserving order and uniqueness."""
    return list(dict.fromkeys([primary_url, OWID_FALLBACK_DATA_URL]))


def _download_file(url: str, output_path: Path, *, timeout_seconds: int) -> None:
    """Download a remote file to output path using atomic replace."""
    with requests.get(url, timeout=timeout_seconds, stream=True) as response:
        response.raise_for_status()
        with NamedTemporaryFile("wb", delete=False, dir=str(output_path.parent)) as temp_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    temp_file.write(chunk)
            temp_path = Path(temp_file.name)

    temp_path.replace(output_path)


def _log_dataset_shape(csv_path: Path) -> None:
    """Log the row and column count of a CSV file."""
    dataframe = pd.read_csv(csv_path, low_memory=False)
    LOGGER.info("File path: %s", csv_path)
    LOGGER.info("Rows: %s | Columns: %s", len(dataframe), len(dataframe.columns))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the download script."""
    parser = argparse.ArgumentParser(description="Download OWID COVID-19 dataset")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/raw/owid_covid_data.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=OWID_COVID_DATA_URL,
        help="Override dataset URL",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="HTTP timeout in seconds",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    download_owid_data(
        output_path=args.output_path,
        force=args.force,
        url=args.url,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
