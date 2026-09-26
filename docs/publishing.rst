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
``https://brightertiger.github.io/robots.txt``. A file at
``/pygarble/robots.txt`` would not control Google crawling. This project does
not publish a misleading project-path robots file. A missing root robots
file does not itself block crawling; submit this project's sitemap directly
in Search Console. A host-root robots policy must be managed separately and
must account for other projects on the host.

Documentation hosting
~~~~~~~~~~~~~~~~~~~~~

Use **Settings → Pages → Build and deployment → Source: GitHub Actions**.
The repository's Sphinx workflow builds and deploys the documentation. A
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

The existing release workflow runs when a ``v*`` tag is pushed. It tests the
package, checks that the tag equals ``pygarble.__version__``, builds wheel
and sdist artifacts, and uploads them to PyPI. The publishing job uses PyPI
Trusted Publishing with GitHub OIDC and the ``pypi`` GitHub environment;
there is no long-lived API token in the workflow.

One-time PyPI setup
~~~~~~~~~~~~~~~~~~~

As an owner of the existing ``pygarble`` project, open its **Publishing**
settings and add a GitHub Trusted Publisher with these exact fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - GitHub owner
     - ``brightertiger``
   * - Repository
     - ``pygarble``
   * - Workflow filename
     - ``release.yml``
   * - Environment
     - ``pypi``

Configure the GitHub ``pypi`` environment before the next release. A required
reviewer can make uploading an explicit final maintainer action. The PyPI
publisher record and the workflow's environment must match. Configure this
before replacing the token-based release workflow; merging the change does
not configure PyPI for you. Remove the old API token only after checking
that no other publishing process still needs it.

Release checklist
~~~~~~~~~~~~~~~~~

1. Choose the next unused version. The module separation and optional
   backends are additive changes suitable for a new minor release; the
   maintainer chooses the final version. Do not tag the existing version
   merely to make the source and PyPI appear synchronized.
2. On a release branch, update ``pygarble.__version__``, turn the changelog's
   Unreleased entries into dated release notes, and replace source-only
   installation notices with the actual release version. Update README,
   installation and migration guidance together.
3. Run the full tests, optional backend tests, formatting/type checks,
   generated-data and golden checks, and strict documentation build. Open a
   PR and resolve CI failures before merging.
4. Build with ``python -m build`` and validate with
   ``python -m twine check dist/*``. Install the wheel in a fresh environment
   without runtime dependencies and exercise both canonical and old imports.
   Check the README links as rendered by PyPI, not just GitHub-relative links.
5. Verify the Trusted Publisher setup. Once the release is approved and the
   version change is merged, tag that exact commit ``v<version>`` and push
   the tag. **This publishes to PyPI and also triggers the existing Docker
   image publishing workflow.** A draft GitHub release alone does not trigger
   this tag-based package workflow.
6. Verify the published package version and metadata, create the matching
   GitHub release with changelog notes, and check the live docs. Published
   files cannot be replaced with different content under the same version;
   use a new version for a correction.

Do not treat a built artifact, passing CI or a pushed branch as a completed
release. Confirm the actual PyPI project and download/install the release.
If using TestPyPI for a rehearsal, configure its publisher separately; its
credentials and project are distinct from production PyPI.

Validation and maintenance
--------------------------

.. code-block:: bash

   python -m sphinx -n -W --keep-going -b html docs /tmp/pygarble-docs
   python paper/scripts/check_discovery.py /tmp/pygarble-docs

The discovery check validates sitemap coverage, canonical URLs, descriptions,
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
