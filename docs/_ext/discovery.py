"""Discovery metadata and indexes for the published HTML documentation."""

import datetime
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urljoin
from xml.etree import ElementTree as ET

REPOSITORY = "https://github.com/brightertiger/pygarble"
DESCRIPTION = (
    "pygarble is a local Python library for PII detection, secret scanning, "
    "profanity detection, gibberish checks and redaction. No LLM calls."
)
DESCRIPTIONS = {
    "index": DESCRIPTION,
    "benchmarks": "pygarble's preprint benchmark on survey junk and tweets "
    "against two CPU baselines. Results, intervals, cost and reproduction.",
    "installation": "Install pygarble and optional local screening backends. "
    "Understand released versions, source installs and dependencies.",
    "quickstart": "Screen secrets, PII and profanity with pygarble, "
    "or check English gibberish independently. Runnable Python examples.",
    "standalone-screening": "Use pygarble.screening with native rules, "
    "phonenumbers, stdnum, detect-secrets and Gitleaks. Options and costs.",
    "pii": "Detect emails, phone numbers, cards and structured identifiers "
    "with local patterns and checksums. PII coverage and locale packs.",
    "secrets": "Find API keys, private keys, credentials and generic secrets "
    "with pygarble. Native rules and optional local secret scanners.",
    "profanity": "Detect English profanity and obfuscated words locally. "
    "Configure word lists, severity tiers and allowlists without an LLM.",
    "cli": "Scan or redact documents with pygarble-screen, or run "
    "line-based gibberish checks with pygarble. Options and exit codes.",
    "screening": "Use the compatible combined scanner for secrets, "
    "PII, profanity and gibberish. Findings, confidence and redaction modes.",
    "strategy-guide": "Choose pygarble gibberish detection profiles and "
    "strategies for English text, encoding problems and degenerate output.",
    "strategies": "Reference for pygarble's gibberish strategies, profile "
    "members and accepted configuration settings.",
    "calibration": "Choose a gibberish threshold using labelled examples "
    "and pygarble's precision, recall and false-positive-rate reports.",
    "api": "Python API reference for pygarble screening, optional backends, "
    "gibberish detection, calibration and result types.",
    "examples": "Runnable Python examples for local text screening, "
    "redaction, batching, English-field validation and explanations.",
    "migration": "Upgrade pygarble while preserving existing imports. "
    "Compare standalone screening with the original combined scanner.",
    "architecture": "Explore pygarble's screening and gibberish modules, "
    "shared data, lazy loading and compatibility pointers.",
    "contributing": "Contribute to pygarble: development setup, validation, "
    "rule changes, compatibility checks and optional backend tests.",
    "publishing": "Maintainer guide to pygarble releases, PyPI Trusted "
    "Publishing, Google Search Console and documentation discovery.",
}


def page_url(app: Any, name: str) -> str:
    path = "" if name == "index" else app.builder.get_target_uri(name)
    return urljoin(app.config.html_baseurl, path)


def page_context(
    app: Any,
    name: str,
    template: str,
    context: Dict[str, Any],
    doctree: Any,
) -> None:
    authored = name in app.env.found_docs
    context["discovery_indexable"] = authored
    context["discovery_description"] = DESCRIPTIONS.get(name, DESCRIPTION)
    context["discovery_index"] = urljoin(app.config.html_baseurl, "llms.txt")
    context["google_site_verification"] = os.environ.get(
        "GOOGLE_SITE_VERIFICATION", ""
    )
    context["pageurl"] = page_url(app, name)
    if name == "index":
        # Describe source code, without promising search-engine rich results
        # or labelling unreleased source as an available PyPI release.
        context["discovery_schema"] = json.dumps(
            {
                "@context": "https://schema.org",
                "@type": "SoftwareSourceCode",
                "name": "pygarble",
                "description": DESCRIPTION,
                "url": app.config.html_baseurl,
                "codeRepository": REPOSITORY,
                "programmingLanguage": "Python",
                "runtimePlatform": "Python 3.8+",
                "license": REPOSITORY + "/blob/main/LICENSE",
                "sameAs": ["https://pypi.org/project/pygarble/"],
            }
        ).replace("<", "\\u003c")


def last_modified(app: Any, name: str) -> str:
    """Return the source file's last commit date, else today's date."""
    source = Path(app.env.doc2path(name))
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cs", "--", source.name],
            cwd=str(source.parent),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        stamp = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        stamp = ""
    # Uncommitted pages and builds outside a git checkout use the build date.
    return stamp or datetime.date.today().isoformat()


def build_indexes(app: Any, exception: Optional[Exception]) -> None:
    if exception is not None or app.builder.name != "html":
        return
    pages = sorted(app.env.found_docs)
    namespace = "http://www.sitemaps.org/schemas/sitemap/0.9"
    ET.register_namespace("", namespace)
    root = ET.Element("{" + namespace + "}urlset")
    links = [
        "# pygarble",
        "",
        "> " + DESCRIPTION,
        "",
        "These are current-source docs; consult installation and migration "
        "before assuming an API is in a published PyPI version.",
        "Use pygarble.screening for PII, secrets and profanity; "
        "pygarble.gibberish for gibberish. The top-level Scanner keeps "
        "all four categories. Scores are heuristics, not probabilities.",
        "",
        "## Documentation",
        "",
    ]
    for name in pages:
        url = page_url(app, name)
        entry = ET.SubElement(root, "{" + namespace + "}url")
        ET.SubElement(entry, "{" + namespace + "}loc").text = url
        ET.SubElement(entry, "{" + namespace + "}lastmod").text = (
            last_modified(app, name)
        )
        title = app.env.titles[name].astext()
        links.append(f"- [{title}]({url}): {DESCRIPTIONS.get(name, title)}")
    links.extend(
        [
            "",
            "## Project",
            "",
            f"- [Source and issues]({REPOSITORY})",
            "- [Published package](https://pypi.org/project/pygarble/)",
            "",
            "## Plain-text sources",
            "",
            "Sphinx publishes reStructuredText sources alongside HTML. "
            "Autodoc directives are expanded in the HTML API reference.",
            "",
        ]
    )
    for name in pages:
        source = urljoin(
            app.config.html_baseurl, "_sources/" + name + ".rst.txt"
        )
        links.append(f"- [{name}]({source})")
    output = Path(app.outdir)
    ET.ElementTree(root).write(
        str(output / "sitemap.xml"), encoding="utf-8", xml_declaration=True
    )
    (output / "llms.txt").write_text("\n".join(links) + "\n", encoding="utf-8")
    # Served at /pygarble/robots.txt. Crawlers only honour the host-root
    # file, so this is advisory; it mirrors the thereabouts site.
    robots = [
        "User-agent: *",
        "Allow: /",
        "Sitemap: " + urljoin(app.config.html_baseurl, "sitemap.xml"),
    ]
    (output / "robots.txt").write_text(
        "\n".join(robots) + "\n", encoding="utf-8"
    )


def setup(app: Any) -> Dict[str, Any]:
    app.connect("html-page-context", page_context)
    # Run after the theme (default priority 500): sphinx-immaterial writes its
    # own sitemap.xml when site_url is set, listing generated pages such as
    # genindex and search; this one lists authored pages only and must win.
    app.connect("build-finished", build_indexes, priority=900)
    return {
        "version": "1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
