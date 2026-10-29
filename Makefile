# Convenience targets. On Windows, install make or run the equivalent python -m / pip commands.
PY ?= python

.PHONY: install install-dev download train serve test lint docker-build docker-run benchmark rollback

install:
	$(PY) -m pip install -r requirements.txt

install-dev: install
	$(PY) -m pip install -r requirements-dev.txt

download:
	$(PY) scripts/download_data.py

train:
	$(PY) -m src.training.train

serve:
	$(PY) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q --cov=src --cov-report=term-missing

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format src tests scripts

docker-build:
	docker build -t ml-serving-api:latest .

docker-run:
	docker compose up --build

benchmark:
	$(PY) scripts/benchmark_latency.py

rollback:
	$(PY) scripts/rollback.py --version $(VERSION)
