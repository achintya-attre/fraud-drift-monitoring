.PHONY: help install test train evaluate monitor simulate retrain all clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install required dependencies"
	@echo "  make test       - Run pytest test suite"
	@echo "  make train      - Train models and run cross-validation"
	@echo "  make evaluate   - Evaluate models and run threshold optimization"
	@echo "  make monitor    - Run drift monitoring on nominal test data"
	@echo "  make simulate   - Run drift monitoring with synthetic drift simulation"
	@echo "  make all        - Run end-to-end pipeline (test, train, evaluate, simulate)"

install:
	python -m pip install -r requirements.txt

test:
	pytest tests/ -v

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

monitor:
	python scripts/monitor.py

simulate:
	python scripts/monitor.py --simulate-drift

all: test train evaluate simulate

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
