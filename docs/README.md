# Documentation for Greyhound ML Package

This directory contains the documentation for the Greyhound ML package.

## Structure

- `api/`: API documentation generated from docstrings
- `tutorials/`: Step-by-step tutorials and examples
- `guides/`: How-to guides for specific tasks
- `reference/`: Reference documentation

## Building Documentation

To build the documentation locally:

```bash
pip install -e ".[docs]"
cd docs
make html
```

The built documentation will be available in `_build/html/index.html`.
