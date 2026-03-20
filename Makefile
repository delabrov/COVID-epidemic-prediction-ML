PYTHON ?= python3
COUNTRY ?= France

.PHONY: download preprocess plots pipeline

download:
	$(PYTHON) -m src.data.download_data

preprocess:
	$(PYTHON) -m src.data.preprocess_data --country "$(COUNTRY)"

plots:
	$(PYTHON) -m src.visualization.plot_data --input-path data/processed/covid_$$(echo "$(COUNTRY)" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g').parquet --country "$(COUNTRY)"

pipeline:
	$(PYTHON) main.py --country "$(COUNTRY)"
