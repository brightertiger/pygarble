"""Validate the actual HTML discovery output before a Pages deployment."""

import argparse
import json
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET

BASE = "https://brightertiger.github.io/pygarble/"


class Head(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__()
        self.in_head = False
        self.in_schema = False
        self.meta: Dict[str, str] = {}
        self.links: List[Tuple[str, str]] = []
        self.schema = ""
        self.feed(html)

    def handle_starttag(self, tag: str, attrs: list) -> None:
        data = dict(attrs)
        if tag == "head":
            self.in_head = True
        if not self.in_head:
            return
        if tag == "meta":
            self.meta[data.get("name", data.get("property", ""))] = data.get(
                "content", ""
            )
        if tag == "link":
            self.links.append((data.get("rel", ""), data.get("href", "")))
        if tag == "script" and data.get("type") == "application/ld+json":
            self.in_schema = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self.in_head = False
        if tag == "script":
            self.in_schema = False

    def handle_data(self, data: str) -> None:
        if self.in_schema:
            self.schema += data


def check(directory: Path) -> None:
    sources = sorted((directory / "_sources").glob("*.rst.txt"))
    assert sources, "no authored documentation sources"
    names = [p.name[: -len(".rst.txt")] for p in sources]
    expected = {BASE if n == "index" else BASE + n + ".html" for n in names}
    sitemap = ET.parse(str(directory / "sitemap.xml"))
    urls = [node.text for node in sitemap.findall(".//{*}loc")]
    assert len(urls) == len(set(urls)), "duplicate sitemap URLs"
    assert (
        set(urls) == expected
    ), "sitemap must contain all authored pages only"
    descriptions = set()
    for name in names:
        url = BASE if name == "index" else BASE + name + ".html"
        page = Head((directory / (name + ".html")).read_text(encoding="utf-8"))
        assert [v for k, v in page.links if k == "canonical"] == [url], name
        assert "noindex" not in page.meta.get("robots", ""), name
        description = page.meta.get("description", "")
        assert description and description not in descriptions, name
        descriptions.add(description)
        assert page.meta["og:description"] == description, name
        assert page.meta["og:url"] == url, name
        assert ("describedby", BASE + "llms.txt") in page.links, name
    homepage = Head((directory / "index.html").read_text(encoding="utf-8"))
    schema = json.loads(homepage.schema)
    assert schema["@type"] == "SoftwareSourceCode"
    assert schema["name"] == "pygarble"
    assert schema["url"] == BASE
    assert (
        schema["codeRepository"] == "https://github.com/brightertiger/pygarble"
    )
    assert "softwareVersion" not in schema, "do not label source as a release"
    token = os.environ.get("GOOGLE_SITE_VERIFICATION")
    if token:
        assert homepage.meta.get("google-site-verification") == token
    for filename in ("search.html", "genindex.html", "py-modindex.html"):
        path = directory / filename
        if path.exists():
            assert "noindex" in Head(path.read_text()).meta.get("robots", "")
    index = (directory / "llms.txt").read_text(encoding="utf-8")
    assert index.startswith("# pygarble\n")
    for url in expected:
        assert "](" + url + ")" in index, url
    # Verify every local link in the text index, including raw RST sources.
    import re

    for target in re.findall(r"\]\(([^)]+)\)", index):
        if target.startswith(BASE):
            relative = unquote(urlsplit(target).path[len("/pygarble/") :])
            assert (directory / (relative or "index.html")).is_file(), target
    print(f"Discovery checks passed for {len(names)} documentation pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check(parser.parse_args().directory)
