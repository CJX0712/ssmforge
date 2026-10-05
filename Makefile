.PHONY: install lint test demo benchmark ablation all clean

PY ?= python

install:
	$(PY) -m pip install ruff==0.16.10 pytest pytest-cov
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

test:
	$(PY) -m pytest -q -W ignore::UserWarning --cov=ssmforge --cov-report=term

demo:
	$(PY) examples/run_demo.py

benchmark:
	$(PY) -m ssmforge.cli benchmark

ablation:
	$(PY) -m ssmforge.cli ablation --task adding --T 1000

all: lint test demo

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
