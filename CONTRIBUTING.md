# Contributing to Singularity

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/singularity.git`
3. Install dev dependencies: `pip install -e '.[dev]'`
4. Create a branch: `git checkout -b my-feature`

## Development

- Run tests: `pytest`
- Format code: `black singularity/`
- Lint: `ruff check singularity/`

## Pull Requests

1. Update tests if needed
2. Run `black` and `ruff` before submitting
3. Write clear commit messages
4. Open a PR against `main`

## Code Style

- We use [black](https://github.com/psf/black) for formatting (line length: 100)
- We use [ruff](https://github.com/astral-sh/ruff) for linting
- Type hints are encouraged
- Write docstrings for public functions

## Reporting Issues

Please use [GitHub Issues](https://github.com/wisent-ai/singularity/issues).
