# Contributing

Thank you for your interest in contributing to the ABT reference implementation.

## Development Setup

```bash
git clone https://github.com/VikasKumar8048/Average-Binary-Tree-ABT
cd Average-Binary-Tree-ABT
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v                          # all tests
pytest tests/unit/ -v                     # unit tests only
pytest tests/property/ -v                 # property-based tests
pytest tests/ --cov=abt --cov-report=term # with coverage
```

## Code Style

```bash
ruff check abt/ tests/
black abt/ tests/
mypy abt/
```

## Guiding Principle

The paper is the single source of truth. Contributions must:
- Not modify formal definitions or theorem statements
- Not alter mathematical notation
- Keep the implementation faithful to Algorithm 1
- Add tests for any new behaviour
- Pass `python reproduce_paper.py` with ALL PASS

## Reporting Issues

Open an issue at https://github.com/VikasKumar8048/Average-Binary-Tree-ABT/issues
