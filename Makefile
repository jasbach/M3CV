.PHONY: help install install-dev test test-unit test-e2e lint format clean docs

# Default target
help:
	@echo "M3CV - Medical Multi-Modal Computer Vision"
	@echo ""
	@echo "Available targets:"
	@echo "  make install        - Install all packages"
	@echo "  make install-dev    - Install all packages with development dependencies"
	@echo "  make test           - Run all tests (unit + E2E)"
	@echo "  make test-unit      - Run only unit tests (skip E2E)"
	@echo "  make test-e2e       - Run only E2E integration tests"
	@echo "  make lint           - Run linter (ruff check)"
	@echo "  make format         - Format code (ruff format)"
	@echo "  make clean          - Remove build artifacts and caches"
	@echo ""
	@echo "E2E Test Setup:"
	@echo "  uv run python scripts/setup_e2e_test_data.py  - Setup E2E test data"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make install"
	@echo "  2. uv run m3cv-prep inspect /path/to/dicom/"
	@echo "  3. uv run m3cv-prep pack /path/to/dicom/ --out-path data.h5"

# Install all packages
install:
	@echo "Installing M3CV packages..."
	uv sync
	@echo ""
	@echo "✓ Installation complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  - Run tests: make test"
	@echo "  - Inspect DICOM: uv run m3cv-prep inspect /path/to/dicom/"
	@echo "  - Pack DICOM: uv run m3cv-prep pack /path/to/dicom/ --out-path data.h5"

# Install with development dependencies
install-dev:
	@echo "Installing M3CV packages with development dependencies..."
	uv sync --all-extras
	@echo "✓ Development installation complete!"

# Run all tests
test:
	@echo "Running all tests..."
	@echo ""
	@echo "=== m3cv-dataprep ==="
	uv run pytest packages/m3cv-dataprep/src/tests/ -v
	@echo ""
	@echo "=== m3cv-data (unit + E2E) ==="
	uv run pytest packages/m3cv-data/src/tests/ -v
	@echo ""
	@echo "=== m3cv-models ==="
	uv run pytest packages/m3cv-models/src/tests/ -v
	@echo ""
	@echo "✓ All tests completed!"

# Run only unit tests (skip E2E)
test-unit:
	@echo "Running unit tests (E2E skipped)..."
	@echo ""
	@echo "=== m3cv-dataprep ==="
	uv run pytest packages/m3cv-dataprep/src/tests/ -v
	@echo ""
	@echo "=== m3cv-data (unit only) ==="
	uv run pytest packages/m3cv-data/src/tests/ -m "not e2e" -v
	@echo ""
	@echo "=== m3cv-models ==="
	uv run pytest packages/m3cv-models/src/tests/ -v
	@echo ""
	@echo "✓ Unit tests completed!"

# Run only E2E integration tests
test-e2e:
	@echo "Running E2E integration tests..."
	@echo ""
	@if [ ! -d "packages/m3cv-data/src/tests/fixtures/e2e_test_patients" ]; then \
		echo "⚠️  E2E test data not found!"; \
		echo ""; \
		echo "Setup E2E test data first:"; \
		echo "  uv run python setup_e2e_test_data.py"; \
		echo ""; \
		echo "Or see: packages/m3cv-data/src/tests/fixtures/README.md"; \
		exit 1; \
	fi
	uv run pytest packages/m3cv-data/src/tests/test_integration_e2e.py -v
	@echo ""
	@echo "✓ E2E tests completed!"

# Quick test (fast, no verbose)
test-quick:
	@echo "Running quick tests..."
	@uv run pytest packages/m3cv-dataprep/src/tests/ -q --tb=no
	@uv run pytest packages/m3cv-data/src/tests/ -m "not e2e" -q --tb=no
	@uv run pytest packages/m3cv-models/src/tests/ -q --tb=no
	@echo ""
	@echo "✓ Quick tests completed!"

# Run linter
lint:
	@echo "Running linter..."
	uv run ruff check .
	@echo "✓ Linting complete!"

# Format code
format:
	@echo "Formatting code..."
	uv run ruff format .
	@echo "✓ Formatting complete!"

# Check formatting without modifying files
format-check:
	@echo "Checking code formatting..."
	uv run ruff format --check .
	@echo "✓ Format check complete!"

# Run both lint and format check
check: lint format-check
	@echo "✓ All checks passed!"

# Clean build artifacts and caches
clean:
	@echo "Cleaning build artifacts and caches..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✓ Clean complete!"

# Deep clean (including virtual environment)
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf .venv
	@echo "✓ Deep clean complete!"
	@echo ""
	@echo "Run 'make install' to reinstall"

# Generate documentation (placeholder)
docs:
	@echo "Documentation generation not yet implemented"
	@echo "See README files:"
	@echo "  - README.md (main)"
	@echo "  - packages/m3cv-dataprep/README.md"
	@echo "  - packages/m3cv-data/README.md"
	@echo "  - packages/m3cv-models/README.md"

# Show package versions
versions:
	@echo "Package versions:"
	@echo "  m3cv-dataprep: $$(grep '^version' packages/m3cv-dataprep/pyproject.toml | cut -d'"' -f2)"
	@echo "  m3cv-data:     $$(grep '^version' packages/m3cv-data/pyproject.toml | cut -d'"' -f2)"
	@echo "  m3cv-models:   $$(grep '^version' packages/m3cv-models/pyproject.toml | cut -d'"' -f2)"
	@echo ""
	@echo "Python: $$(uv run python --version)"
	@echo "uv:     $$(uv --version)"

# Setup E2E test data
setup-e2e:
	@echo "Setting up E2E test data..."
	uv run python scripts/setup_e2e_test_data.py
	@echo ""
	@echo "✓ E2E test data setup complete!"
	@echo ""
	@echo "Run E2E tests: make test-e2e"

# CI-friendly test target (no E2E)
test-ci: test-unit
	@echo "✓ CI tests completed!"

# Development workflow
dev: install-dev format lint test-unit
	@echo "✓ Development workflow complete!"
