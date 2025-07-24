# Greyhound ML Package

A modern Python package for machine learning model training with PyTorch Lightning and Hydra configuration.

## Features

- 🚀 Modern Python packaging with pyproject.toml
- ⚡ PyTorch Lightning for scalable training
- 🔧 Hydra for configuration management
- 📊 Weights & Biases integration for experiment tracking
- 🧪 Comprehensive testing with pytest
- 🎨 Code formatting with Black and Ruff
- 📝 Type checking with MyPy
- 🔄 Pre-commit hooks for code quality

## Installation

### Development Installation

```bash
git clone https://github.com/yourusername/greyhound.git
cd greyhound
pip install -e ".[dev]"
```

### Production Installation

```bash
pip install greyhound
```

## Quick Start

### Training a Model

```bash
greyhound-train --config-path configs --config-name train
```

### Evaluating a Model

```bash
greyhound-eval --config-path configs --config-name eval --checkpoint path/to/checkpoint.ckpt
```

### Using the Python API

```python
from greyhound.models import SimpleClassifier
from greyhound.trainers import LightningTrainer
from greyhound.data import DataModule

# Create model and data
model = SimpleClassifier(num_classes=10)
data_module = DataModule(batch_size=32)

# Train
trainer = LightningTrainer()
trainer.fit(model, data_module)
```

## Project Structure

```
greyhound/
├── src/greyhound/           # Main package
│   ├── models/              # Model definitions
│   ├── data/                # Data loading and processing
│   ├── trainers/            # Training logic
│   ├── utils/               # Utility functions
│   └── cli.py               # Command line interface
├── configs/                 # Hydra configuration files
├── tests/                   # Test suite
├── docs/                    # Documentation
└── notebooks/               # Example notebooks
```

## Configuration

This project uses Hydra for configuration management. All configuration files are stored in the `configs/` directory.

## Development

### Setup Development Environment

```bash
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## License

MIT License - see LICENSE file for details.
