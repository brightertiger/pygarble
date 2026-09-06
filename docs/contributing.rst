Contributing
============

We welcome contributions to pygarble! This guide will help you get started.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/brightertiger/pygarble.git
   cd pygarble

3. Create a virtual environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .

Development Workflow
--------------------

1. Create a feature branch:

.. code-block:: bash

   git checkout -b feature/amazing-feature

2. Make your changes
3. Run tests:

.. code-block:: bash

   pytest tests/

4. Run linting:

.. code-block:: bash

   flake8 pygarble/
   black pygarble/
   mypy pygarble/

5. Run all CI checks locally:

.. code-block:: bash

   ./scripts/test_ci.sh

6. Commit your changes:

.. code-block:: bash

   git commit -m "Add amazing feature"

7. Push to your fork:

.. code-block:: bash

   git push origin feature/amazing-feature

8. Open a Pull Request on GitHub

Adding New Strategies
---------------------

To add a new detection strategy:

1. Create a new file in ``pygarble/strategies/``
2. Inherit from ``BaseStrategy``
3. Implement ``_predict_impl()`` and ``_predict_proba_impl()`` methods
4. Add the strategy to the ``Strategy`` enum and lazy implementation map in ``registry.py``
5. Register its supported settings in ``options.py``
6. Add tests in ``tests/test_strategies.py``
7. Regenerate documentation with ``python scripts/update_strategy_docs.py``

Example strategy implementation:

.. code-block:: python

   from .base import BaseStrategy

   class MyCustomStrategy(BaseStrategy):
       def _predict_impl(self, text: str) -> bool:
           # Implement your detection logic
           return False

       def _predict_proba_impl(self, text: str) -> float:
           # Return heuristic score (0.0 to 1.0)
           return 0.0

Code Style
----------

We follow these coding standards:

- **PEP 8**: Python style guide
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **No docstrings**: As per project rules, we don't add docstrings to functions or classes

Testing
-------

All new code must include tests. We use pytest for testing.

- Tests go in the ``tests/`` directory
- Test files should be named ``test_*.py``
- Test functions should be named ``test_*``
- Aim for high test coverage

Documentation
-------------

- Update the README.md for user-facing changes
- Update this documentation for API changes
- Add examples for new features
- Keep the changelog up to date

Pull Request Guidelines
-----------------------

- Keep PRs focused on a single feature or bugfix
- Include tests for new functionality
- Update documentation as needed
- Ensure all CI checks pass
- Write clear commit messages

Issue Reporting
---------------

When reporting issues:

1. Check existing issues first
2. Use the issue template
3. Provide a minimal reproduction case
4. Include Python version and package version
5. Describe expected vs actual behavior

Release Process
---------------

Releases are managed by maintainers:

1. Update version numbers in ``pyproject.toml`` and ``docs/conf.py``
2. Update ``CHANGELOG.md``
3. Create a release tag
4. Build and publish to PyPI

Questions?
----------

If you have questions about contributing, please:

- Open an issue for discussion
- Check existing issues and discussions
- Review the codebase and tests for examples
