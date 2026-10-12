Publishing and discoverability
==============================

pygarble is already open source: the GitHub repository is public and the
library code is MIT licensed. PyPI distributes releases independently from
``main``. A repository change does not update the installed package until a
new version is published. Keep release metadata and documentation aligned
so people and search-assisted tools can tell which APIs are available.

Search and AI discovery
-----------------------

The HTML documentation build generates:

* Canonical URLs under ``https://brightertiger.github.io/pygarble/`` and
  descriptions for each authored page, plus social-sharing metadata.
* ``sitemap.xml`` containing authored guides and API documentation. Search
  pages, generated indexes and duplicate source-code HTML are excluded.
* ``llms.txt`` with a concise project summary, guide links and links to
  Sphinx's plain-text reStructuredText sources. It is an optional navigation
  aid for retrieval tools, not a training-data submission mechanism.
* Homepage ``SoftwareSourceCode`` structured data describing the public
  repository. It makes no claims about ratings, adoption or an unpublished
  release and does not guarantee enhanced search presentation.

There is no guaranteed way to make a model learn or recommend a package.
Google says its AI search features use the same SEO fundamentals as ordinary
search and do not require special AI files. Helpful documentation, accurate
package metadata, runnable examples and real community references are the
priority. ``llms.txt`` is a proposal, not a universal crawler contract.

Verify Google Search Console
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Open `Google Search Console <https://search.google.com/search-console/>`_
   using the account that should own the property. Add a **URL-prefix**
   property for ``https://brightertiger.github.io/pygarble/``. A Domain
   property for ``github.io`` is not appropriate: you do not control its DNS.
2. Choose **HTML tag** verification. Copy only the ``content`` value from
   Google's ``google-site-verification`` meta tag. This is a public
   verification identifier, not a password or Google access token.
3. In the GitHub repository, set the Actions repository variable
   ``GOOGLE_SITE_VERIFICATION`` to that value. The documentation build reads
   it and adds the tag to the homepage. Keep it configured after verification.
   To also enable Google Analytics, set the repository variable
   ``GOOGLE_ANALYTICS_KEY`` to the measurement ID (``G-XXXXXXXXXX``).
   ``docs/conf.py`` passes it to the theme's built-in Google Analytics
   integration, which loads gtag.js only after a visitor accepts analytics
   cookies in the consent banner. Without the variable no analytics code is
   emitted.
4. Merge the discovery changes, then let the documentation workflow deploy.
   If setting the variable later, rerun **Build and Deploy Documentation**
   on ``main``. Confirm the exact token is present in the live homepage's
   HTML before clicking **Verify** in Search Console.
5. In **Sitemaps**, submit
   ``https://brightertiger.github.io/pygarble/sitemap.xml``. Use **URL
   Inspection** to request indexing of the homepage and a few key guides,
   including ``standalone-screening.html`` and ``quickstart.html``.
6. Monitor Page indexing and Performance. Requests can take time and do not
   guarantee indexing or ranking. Repeatedly requesting the same URL does
   not accelerate crawling. Use Search Console rather than a ``site:``
   search as the authoritative account-level status check.

``robots.txt`` is scoped to the host root,
``https://brightertiger.github.io/robots.txt``. The build also writes
``/pygarble/robots.txt`` (allow all, plus the sitemap URL) to match the
sister project's site, but Google does not read a robots file below the host
root, so it neither blocks nor announces anything. A missing root robots
file does not itself block crawling; submit this project's sitemap directly
in Search Console. Each sitemap entry carries a ``lastmod`` date from the
source file's last commit. A host-root robots policy must be managed separately and
must account for other projects on the host.

Documentation hosting
~~~~~~~~~~~~~~~~~~~~~

Use **Settings → Pages → Build and deployment → Source: GitHub Actions**.
The repository's Sphinx workflow builds and deploys the documentation with
the sphinx-immaterial theme, a Sphinx port of Material for MkDocs configured
in ``docs/conf.py`` to match the sister project's site. A
parallel legacy Pages build from the raw repository can replace the HTML
with different content. Verify the live homepage, sitemap and ``llms.txt``
after a deployment; a green build alone is not proof of what is being served.
The workflow's manual trigger redeploys existing source, without publishing
anything to PyPI.

Community discovery
~~~~~~~~~~~~~~~~~~~

Keep the repository description and topics accurate: Python, text screening,
PII detection, secret detection, profanity detection, gibberish and redaction.
Link the GitHub repository, documentation and PyPI project to each other.
The root contributor guide, security policy and issue templates help users
report useful problems and contribute fixes.

After a release, publish a concise announcement with a working example,
coverage limits, measured costs and links to docs. Suitable places include a
maintainer blog and relevant Python/security communities, subject to their
posting rules. Seek useful feedback; avoid repetitive directory submissions
or claims of universal detection. These are maintainer publishing steps,
not automated messages sent by this repository.

Publish a new package version
-----------------------------

``.github/workflows/release.yml`` publishes releases. It has two triggers:

* **A pushed** ``vX.Y.Z`` **tag** publishes to PyPI and then creates the
  GitHub Release.
* **A manual run** (``workflow_dispatch``) is a dry run. It builds
  ``X.Y.Z.devN``, where ``N`` is the run number, and publishes that to
  TestPyPI only.

Both paths run the tests on Python 3.8 and 3.12, lint and the golden checks.
They build the sdist and wheel once, check them with ``twine check
--strict``, record GitHub build provenance for them and smoke-test the wheel
alone on Python 3.8. Every later job publishes those same files. Uploads use
PyPI Trusted Publishing through GitHub OIDC, so no API token is stored
anywhere. The PyPI upload also carries PEP 740 attestations, which
``pypa/gh-action-pypi-publish`` produces by default.

One-time maintainer setup
~~~~~~~~~~~~~~~~~~~~~~~~~

These settings live outside the repository, so merging the workflow does not
configure them. Do them before the first dry run.

1. **GitHub environments.** In **Settings → Environments**, create ``pypi``
   with yourself as a required reviewer and a deployment rule for tags
   matching ``v*``. Create ``testpypi`` with a deployment rule for the
   ``main`` branch. GitHub creates an environment without protection the
   first time a job uses it, so create both first.
2. **PyPI Trusted Publisher.** As an owner of the ``pygarble`` project on
   pypi.org, open **Manage → Publishing** and add a GitHub publisher with
   these fields:

   .. list-table::
      :header-rows: 1

      * - Field
        - Value
      * - Owner
        - ``brightertiger``
      * - Repository
        - ``pygarble``
      * - Workflow filename
        - ``release.yml``
      * - Environment
        - ``pypi``

3. **TestPyPI Trusted Publisher.** Add the same fields on test.pypi.org with
   the environment ``testpypi``. If the project does not exist on TestPyPI,
   add it as a pending publisher.
4. **Secrets.** Delete the ``PYPI_API_TOKEN``, ``TEST_PYPI_API_TOKEN`` and
   ``PAT_TOKEN`` repository secrets, which no workflow uses, and revoke the
   PyPI token. Add ``CODECOV_TOKEN`` from the Codecov project settings;
   without it the coverage upload is rejected, although CI stays green.
5. **Zenodo.** Sign in to zenodo.org with GitHub, open **GitHub** in the
   account menu, press **Sync** and switch the repository on. Zenodo then
   archives every published GitHub Release and mints a version DOI, taking
   the authors and title from ``CITATION.cff``. It must be on before the
   release is published, because Zenodo does not archive earlier releases.

Trusted Publishers are not visible through the GitHub API. Check the other
settings with:

.. code-block:: bash

   R=repos/brightertiger/pygarble
   gh api $R/environments --jq '.environments[] | {name, rules: [.protection_rules[].type]}'
   gh api $R/actions/secrets --jq '.secrets[].name'   # CODECOV_TOKEN only
   gh api $R/hooks --jq '.[].config.url'              # includes a zenodo.org hook

Release checklist
~~~~~~~~~~~~~~~~~

1. **Prepare a release branch.** Set ``pygarble.__version__``. Rename the
   changelog's ``[Unreleased]`` entries to ``## [X.Y.Z] - YYYY-MM-DD``,
   leave an empty ``## [Unreleased]`` above it and update the compare links
   at the bottom. The GitHub Release notes are that section, so it must
   exist under exactly that heading. Update ``version`` and
   ``date-released`` in ``CITATION.cff`` and the version in the README's
   software citation.
2. **Check and merge.** Run ``make lint``, ``pytest``, the golden checks and
   the strict documentation build below. Open a PR, resolve CI failures and
   merge it.
3. **Dry run.** Run the workflow from ``main`` and install the result from
   TestPyPI:

   .. code-block:: bash

      gh workflow run release.yml --ref main
      gh run watch "$(gh run list -w release.yml -L1 --json databaseId -q '.[0].databaseId')" --exit-status
      pip install -i https://test.pypi.org/simple/ --no-deps "pygarble==X.Y.Z.devN"

4. **Tag the merge commit.** Use an annotated tag whose name is ``v`` plus
   ``pygarble.__version__``. The build job stops before anything is
   published if the two differ.

   .. code-block:: bash

      git switch main && git pull --ff-only && git log -1 --oneline
      git tag -a vX.Y.Z -m "pygarble X.Y.Z"
      git push origin vX.Y.Z

5. **Approve.** Open the run and approve the ``pypi`` deployment under
   **Review deployments**. The GitHub Release follows automatically after
   the PyPI upload succeeds.

0.9.0 and 0.10.0 were development milestones that were never published, and
must never be tagged.

What a release produces
~~~~~~~~~~~~~~~~~~~~~~~

* The sdist and wheel on PyPI, with PEP 740 attestations shown as provenance
  on the project page.
* GitHub build provenance attestations for both files, checkable with
  ``gh attestation verify``.
* A GitHub Release named ``pygarble X.Y.Z`` on the tag, with both files
  attached. Its notes are the changelog section plus install, upgrade-guide
  and preprint links.
* A Zenodo record with a DOI for that version, created from the GitHub
  Release.

No container image is built.

Verify the release
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3.12 -m venv /tmp/pg && /tmp/pg/bin/pip install --no-cache-dir pygarble==X.Y.Z
   /tmp/pg/bin/pygarble --version && /tmp/pg/bin/pygarble check -t "qxzjkwpv bnmqwer"
   gh release download vX.Y.Z -D /tmp/pg-dl
   gh attestation verify /tmp/pg-dl/*.whl -R brightertiger/pygarble

Then check these by hand:

* The PyPI page renders the README and shows the documentation link, both
  authors and provenance.
* The documentation site describes the released version.
* Zenodo shows the new record. Add its DOI to the README, the documentation
  and ``CITATION.cff`` in a follow-up PR.

Do not treat a built artifact, passing CI or a pushed tag as a completed
release. Confirm the PyPI project and install from it.

Roll back
~~~~~~~~~

* **A job before** ``publish`` **fails.** Nothing was published. Delete the
  tag with ``git push origin :refs/tags/vX.Y.Z``, fix the problem by PR and
  tag again.
* **The** ``publish`` **or** ``github-release`` **job fails.** Rerun it with
  ``gh run rerun <run-id> --failed``, which reuses the files already built.
  Never rebuild a version that may have partly reached PyPI.
* **Bad files reached PyPI.** Yank the release on PyPI with a reason, keep
  the tag and the GitHub Release, and ship ``X.Y.Z+1``. PyPI never accepts
  a filename again, even after deletion, and a Zenodo record cannot be
  withdrawn. Say in the next release's notes why the version was yanked.

Validation and maintenance
--------------------------

.. code-block:: bash

   python -m sphinx -n -W --keep-going -b html docs /tmp/pygarble-docs
   python paper/scripts/check_discovery.py /tmp/pygarble-docs

The discovery check validates sitemap coverage and ``lastmod`` dates, the
project-path ``robots.txt``, canonical URLs, descriptions,
structured data, index links and optional verification metadata in actual
built HTML. Add a description in ``docs/_ext/discovery.py`` when adding a
new guide. These are build-time features; they add no package dependencies
or inference cost.

References
----------

* `Google: request a recrawl <https://developers.google.com/search/docs/crawling-indexing/ask-google-to-recrawl>`_
* `Google: submit a sitemap <https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap>`_
* `Google: AI features and your website <https://developers.google.com/search/docs/appearance/ai-features>`_
* `Search Console ownership verification <https://support.google.com/webmasters/answer/9008080>`_
* `Google: robots.txt scope <https://developers.google.com/search/docs/crawling-indexing/robots/robots_txt>`_
* `The llms.txt proposal <https://llmstxt.org/>`_
* `PyPI: add a Trusted Publisher <https://docs.pypi.org/trusted-publishers/adding-a-publisher/>`_
* `PyPI: publish using GitHub OIDC <https://docs.pypi.org/trusted-publishers/using-a-publisher/>`_
* `PyPA: publishing with GitHub Actions <https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>`_
* `GitHub: artifact attestations <https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations/using-artifact-attestations-to-establish-provenance-for-builds>`_
* `Zenodo: GitHub integration <https://help.zenodo.org/docs/github/>`_
