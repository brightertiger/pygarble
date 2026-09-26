# Contributing to pygarble

pygarble is an MIT-licensed Python project for local text screening and
gibberish detection. Bug reports, documentation improvements and detection
rules with positive and negative examples are welcome.

Start with the [contributor guide](docs/contributing.rst) and
[architecture guide](docs/architecture.rst). Native screening belongs in
`pygarble/screening/`; gibberish checks belong in `pygarble/gibberish/`.
Preserve the compatibility pointers at the old import paths.

```bash
git clone https://github.com/brightertiger/pygarble.git
cd pygarble
python -m pip install -e '.[dev,docs]'
git switch -c fix/describe-your-change
python -m pytest -q
```

Keep the base package free of runtime dependencies. Optional integrations
must be explicitly selected and run locally. Add regression cases for
behavior changes; document coverage limits and false positives. See the
contributor guide for formatting, typing, golden data and backend checks.

Open a pull request explaining the problem, resulting behavior and checks
you ran. Use synthetic examples; do not include real credentials or personal
data. Report vulnerabilities through [SECURITY.md](SECURITY.md).

Keep discussions constructive and focused on reproducible behavior. Release
and discoverability steps are in the [maintainer guide](docs/publishing.rst).
