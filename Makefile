.PHONY: install test lint format clean build build-web clean-web

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

format:
	ruff format yanex tests

format-check:
	ruff format --check yanex tests

# Web UI
build-web:
	./build_web_ui.sh

clean-web:
	rm -rf yanex/web/out
	rm -rf yanex/web/.next
	rm -rf yanex/web/node_modules

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Build (includes web UI)
build: clean build-web
	python -m build

# All checks
check: format-check lint test