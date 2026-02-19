.PHONY: install install-dev test train backtest paper-run fetch-data dashboard

install:
	python3 -m pip install .

install-dev:
	python3 -m pip install ".[dev]"

test:
	python3 -m pytest

fetch-data:
	python3 -m src.cli fetch-data

train:
	python3 -m src.cli train

backtest:
	python3 -m src.cli backtest

paper-run:
	python3 -m src.cli paper-run

dashboard:
	streamlit run src/dashboard/app.py
