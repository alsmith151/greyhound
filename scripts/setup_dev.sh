#!/bin/bash
# Development environment setup script

set -e

echo "Setting up Greyhound ML development environment..."

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p logs
mkdir -p wandb

echo "Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Configure your Weights & Biases account: wandb login"
echo "2. Review the configuration files in configs/"
echo "3. Run tests: pytest"
echo "4. Start training: greyhound-train --config-path configs --config-name train"
