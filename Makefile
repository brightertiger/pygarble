PYTHON ?= python

.PHONY: benchmark benchmark-check benchmark-prepare paper help install install-dev test lint format clean build upload-test upload docs docs-clean docs-serve docs-deploy

help:
	@echo "Available commands:"
	@echo "  benchmark-check   Verify study analysis and saved predictions (offline)"
	@echo "  benchmark-prepare Download pinned research corpus/model assets"
	@echo "  benchmark         Run the full paper comparison to reproduction-full/"
	@echo "  paper             Build the paper PDF and ignored upload bundle"
	@echo "  install      Install the package"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  upload-test  Upload to test PyPI"
	@echo "  upload       Upload to PyPI"
	@echo "  docs         Build documentation"
	@echo "  docs-clean   Clean documentation build"
	@echo "  docs-serve   Serve documentation locally"
	@echo "  docs-deploy  Deploy documentation to GitHub Pages"

install:
	pip install .

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	black --check pygarble tests paper/scripts paper/regression
	isort --check-only pygarble tests paper/scripts paper/regression
	flake8 pygarble tests paper/scripts paper/regression
	mypy pygarble

format:
	isort pygarble tests paper/scripts paper/regression
	black pygarble tests paper/scripts paper/regression

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

build: clean
	python -m build

upload-test: build
	python -m twine upload --repository testpypi dist/*

upload: build
	python -m twine upload dist/*

docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

docs-serve: docs
	cd docs/_build/html && python -m http.server 8000

docs-deploy: docs
	@echo "Deploying documentation to GitHub Pages..."
	@echo "Note: This requires GitHub Actions to be set up for automatic deployment"
	@echo "Manual deployment steps:"
	@echo "1. Ensure GitHub Pages is enabled in repository settings"
	@echo "2. Push to main branch or create a release"
	@echo "3. GitHub Actions will automatically deploy the docs"
	@echo ""
	@echo "Current documentation build is ready in docs/_build/html/"

# Use a Python 3.12 research environment; the full run is optional and local.
benchmark-check:
	$(PYTHON) -m unittest paper.scripts.study.test_study paper.scripts.study.test_full_corpus paper.scripts.study.test_chunk_metrics

benchmark-prepare:
	$(PYTHON) -m paper.scripts.study.full_corpus --download-model

benchmark:
	$(PYTHON) -m paper.scripts.study.full_corpus --output paper/study/reproduction-full

paper:
	$(PYTHON) -m paper.scripts.study.build_paper
