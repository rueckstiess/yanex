.PHONY: install test lint format clean build

# Development setup
install:
	pip install -e ".[dev]"

# Testing
test:
	pytest

test-cov:
	pytest --cov=yanex --cov-report=html --cov-report=term-missing

# Code quality
lint:
	ruff check yanex tests
	mypy yanex

format:
	ruff format yanex tests

format-check:
	ruff format --check yanex tests

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Build
build: clean
	python -m build

# All checks
check: format-check lint test