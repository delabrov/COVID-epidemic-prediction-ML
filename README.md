# COVID Epidemiological Data Pipeline

Clean and modular Python pipeline for epidemiological analysis with OWID data.

## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   └── figures/
├── src/
│   ├── data/
│   │   ├── download_data.py
│   │   └── preprocess_data.py
│   └── visualization/
│       └── plot_data.py
└── main.py
```

## What It Does

1. Download OWID COVID-19 dataset.
2. Preprocess one country time series for analysis.
3. Save cleaned dataset in parquet (and optional CSV).
4. Generate exploratory figures.

No machine learning is included at this stage.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run Full Pipeline

```bash
python main.py --country France
```

Optional flags:

```bash
python main.py --country France --force-download --save-csv
```

## Run Each Step Separately

Download:

```bash
python -m src.data.download_data
```

Preprocess:

```bash
python -m src.data.preprocess_data --country France --save-csv
```

Plot:

```bash
python -m src.visualization.plot_data --input-path data/processed/covid_france.parquet --country France
```
