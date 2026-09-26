Module layout and compatibility
===============================

pygarble provides two focused APIs within one installable package:
``pygarble.screening`` for secrets, PII and profanity, and
``pygarble.gibberish`` for gibberish detection. Existing top-level imports
remain supported. The base install has no runtime dependencies; optional
screening backends are selected explicitly.

Choose an entry point
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - API
     - Default work
     - Guide
   * - ``pygarble.screening.Scanner``
     - Native secrets, PII and profanity checks
     - :doc:`standalone-screening`
   * - ``pygarble.gibberish.EnsembleDetector``
     - English gibberish profile
     - :doc:`strategy-guide`
   * - ``pygarble.Scanner``
     - All four categories, including gibberish
     - :doc:`screening`

The two scanners have different configuration contracts. Optional backends
and custom detectors belong to the standalone scanner. Gibberish profiles,
voting and threshold calibration belong to the gibberish API; the combined
scanner retains its existing gibberish settings.

Source tree
-----------

.. code-block:: text

   pygarble/
     screening/
       pii/          # detector, patterns and checksums
       profanity/    # detector, normalization and word lists
       secrets/      # detector, patterns and entropy
       backends/     # optional local integrations
       base.py       # detector protocols and backend errors
       _engine.py    # shared scan, filtering and redaction flow
       scanner.py    # standalone configuration and detector composition
       cli.py        # document-oriented command line
     gibberish/
       detector.py   # one strategy
       ensemble.py   # profiles and voting
       strategies/   # individual checks, loaded lazily
       analysis.py   # immutable gibberish result types
       calibration.py
       preprocessing.py
       registry.py
       scoring.py
       options.py
     data/           # shared dictionaries and portable JSON tables
     findings.py     # Finding, ScanReport and Redaction
     redaction.py    # overlap handling and rendering
     validation.py  # common input validation
     scanner.py     # compatible four-category API
     cli.py         # existing line-oriented commands

The old detector modules and category folders also remain at the root as
small compatibility pointers. Add implementation changes to the canonical
folders above. Shared data stays at the root because both modules use it;
for example, profanity masking consults the English dictionary. Portable
JSON copies stay under ``pygarble/data/`` for existing consumers. Runtime
detectors use Python tables, not those JSON copies.

Processing and cost
-------------------

The standalone scanner selects native rules, optional backends and custom
detectors at construction. Each detector emits findings against the original
text. The shared engine validates spans, combines results and applies the
confidence policy. Redaction merges overlapping qualifying findings before
replacing text. Reports omit source values; redaction returns the transformed
text and its findings.

Gibberish detectors extract text features and evaluate the selected
strategies. Ensembles combine applicable evidence according to their voting
policy. Importing screening loads lightweight shared result types but does
not load the gibberish ensemble, its strategies or its dictionary. A native
rule may load shared data later when it needs it. Optional Python packages
load only when their backend is selected. Gitleaks starts a local subprocess
per document; reuse scanner instances to reuse Python detector state.

No built-in path calls an LLM, downloads an inference model or verifies
credentials with a remote service. Custom detector implementations control
their own processing. Measure the configurations and input sizes you intend
to deploy; the native-rule throughput figures do not include optional tools.

Compatibility pointers
----------------------

.. list-table:: Representative import paths
   :header-rows: 1
   :widths: 45 55

   * - Supported existing path
     - Canonical path for new code
   * - ``pygarble.GarbleDetector``, ``pygarble.core.GarbleDetector``
     - ``pygarble.gibberish.GarbleDetector``
   * - ``pygarble.EnsembleDetector``, ``pygarble.ensemble.EnsembleDetector``
     - ``pygarble.gibberish.EnsembleDetector``
   * - ``pygarble.Strategy``, ``pygarble.registry.Strategy``
     - ``pygarble.gibberish.Strategy``
   * - ``pygarble.analysis``, ``pygarble.calibration``
     - ``pygarble.gibberish.analysis``, ``pygarble.gibberish.calibration``
   * - ``pygarble.strategies`` and its submodules
     - ``pygarble.gibberish.strategies`` and its submodules
   * - ``pygarble.pii.patterns``
     - ``pygarble.screening.pii.patterns``
   * - ``pygarble.profanity.wordlist``
     - ``pygarble.screening.profanity.wordlist``
   * - ``pygarble.secrets.patterns``
     - ``pygarble.screening.secrets.patterns``

Old and new paths share classes, strategy enums, rule tables and module
caches. The old strategy package forwards exports lazily; its submodules
point to canonical modules. There are no duplicate detection implementations
or new deprecation warnings. The top-level ``Scanner`` continues to mean
the combined scanner; changing that import to ``pygarble.screening`` is an
explicit choice to omit gibberish.

Pickles referencing the old paths still load. Class introspection and newly
written pickles use canonical paths; loading those new pickles in older
pygarble versions is not guaranteed. See :doc:`migration` for a runnable
import example and :doc:`contributing` for the compatibility checks.

Research and maintenance layout
-------------------------------

The paper describes the software separately from its empirical validation.
The repository follows that division:

.. code-block:: text

   pygarble/           # installable library and compatibility pointers
   tests/              # package unit and integration tests
   docs/               # maintained website and user guides
   paper/
     study/            # research protocol, code, predictions and manuscript
     regression/       # engineering fixtures, golden checks and throughput
     scripts/          # data/docs generation and documentation checks

Only ``pygarble`` is installed as library code. The optional neural environment
belongs to ``paper/study/`` and does not add dependencies to the base package.
The manuscript, measured results and benchmark scope are linked from
:doc:`benchmarks`. Generated paper build files live in the ignored
``paper/study/.cache/publication/`` directory.

Within gibberish analysis, shared features retain token offsets and lexical
novelty for the selected strategies. Evidence consists of a score, applicability,
reason and optional spans; the ensemble combines only applicable signals.
``analyze()`` records all signals, while ``predict()`` can short-circuit ``any``
and ``all`` decisions. The paper's CPU comparison measures the full score and
applicability path, so it should not be presented as a timing guarantee for
every API call.
