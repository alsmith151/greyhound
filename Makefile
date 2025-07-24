.PHONY: help install install-dev test lint format type-check clean build docs

help:  ## Display this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install .

install-dev:  ## Install the package in development mode with all dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests
	pytest tests/ --cov=greyhound --cov-report=term-missing

test-all:  ## Run tests across all Python versions using tox
	tox

lint:  ## Run linting checks
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:  ## Format code
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

type-check:  ## Run type checking
	mypy src/

security:  ## Run security checks
	bandit -r src/
	safety check

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	python -m build

docs:  ## Build documentation
	cd docs && make html

docs-serve:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

release-test:  ## Build and upload to TestPyPI
	make clean
	make build
	twine check dist/*
	twine upload --repository testpypi dist/*

release:  ## Build and upload to PyPI
	make clean
	make build
	twine check dist/*
	twine upload dist/*
