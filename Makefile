.PHONY: install install-all install-dev test lint format run analyze pocket web clean

install:            ## minimal install (numpy only)
	pip install -e .

install-all:        ## recommended: plots, bio, web, tests
	pip install -e ".[all]"

install-dev:        ## alias for install-all
	pip install -e ".[all]"

test:               ## run the test suite
	pytest -q

lint:               ## static checks
	ruff check src tests

format:             ## auto-format
	black src tests
	ruff check --fix src tests

run:                ## curated demo → examples/pipeline_7rfw
	protein-features --pdb data/7rfw.pdb --out examples/pipeline_7rfw

analyze:            ## curated demo → examples/stepwise_7rfw
	protein-features-analyze --pdb data/7rfw.pdb --out examples/stepwise_7rfw

pocket:             ## curated demo → examples/pocket_7rfw
	protein-features-pocket --pdb data/7rfw.pdb --out examples/pocket_7rfw

web:                ## local upload UI + API
	protein-features-web

clean:              ## remove build/test artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
