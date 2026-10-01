PY ?= python
VENV := .venv
BIN := $(VENV)/Scripts
ifeq ($(OS),Windows_Nt)
  BIN := $(VENV)/Scripts
else
  BIN := $(VENV)/bin
endif
PYTHON := $(BIN)/python
PIP := $(PYTHON) -m pip

.DEFAULT_GOAL := help
.PHONY: help setup install bootstrap demo demo-fast api dashboard test lint report eval-judge clean clean-data

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv and install dependencies
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "\nOptional but recommended: pip install sentence-transformers"

install: ## Install dependencies into an existing virtualenv
	$(PIP) install -r requirements.txt

bootstrap: ## Fetch data, train the champion, freeze the reference, validate the judge
	$(PYTHON) scripts/bootstrap.py

demo: ## Full 14-window production shift with the real LLM
	$(PYTHON) scripts/demo_shift.py --reset

demo-fast: ## Same structure, offline encoder + mock judge (~50s)
	$(PYTHON) scripts/demo_shift.py --fast --reset

demo-nopromote: ## Show the incident stays open without the retrain
	$(PYTHON) scripts/demo_shift.py --reset --no-promote

api: ## Serve inference + monitoring API on :8000
	$(PYTHON) -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload

dashboard: ## Streamlit dashboard on :8501
	$(PYTHON) -m streamlit run dashboard/app.py --server.port 8501

test: ## Run the test suite
	$(PYTHON) -m pytest -q

test-cov: ## Run the test suite with coverage
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing --cov-report=html

lint: ## Lint
	$(PYTHON) -m ruff check . || true
	@echo "\nInstall ruff with: pip install ruff"

report: ## Regenerate reports from the most recent run
	$(PYTHON) scripts/export_report.py

eval-judge: ## Run the golden judge regression suite
	$(PYTHON) scripts/eval_judge.py

clean: ## Remove telemetry, artifacts and caches
	-rm -rf artifacts data/telemetry.db data/reports data/runs
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
	@echo "Cleaned. data/raw and data/cache are kept so reruns stay offline."

clean-data: ## Also remove the downloaded dataset and encoder cache
	$(MAKE) clean
	-rm -rf data/raw data/cache data/bundled
	@echo "Next run will re-download Banking77."
