.PHONY: check lint format typecheck test install

install:
	pip install -e ".[dev,eval]"

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

test:
	pytest

check: lint format typecheck test
