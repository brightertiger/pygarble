# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Add the project root to the Python path
DOCS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS_DIR.parent))
sys.path.insert(0, str(DOCS_DIR / '_ext'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pygarble'
copyright = '2026, pygarble contributors'
author = 'pygarble contributors'

import pygarble  # noqa: E402

version = pygarble.__version__
release = pygarble.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'discovery',
    'sphinx_immaterial',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_baseurl = 'https://brightertiger.github.io/pygarble/'
html_title = 'pygarble — local PII, secrets, profanity and gibberish detection'
html_theme = 'sphinx_immaterial'
html_static_path = []

# Theme options mirror the Material for MkDocs configuration of the sister
# project at https://brightertiger.github.io/thereabouts/ (its mkdocs.yml).
html_theme_options = {
    'site_url': html_baseurl,
    'repo_url': 'https://github.com/brightertiger/pygarble',
    'repo_name': 'brightertiger/pygarble',
    'edit_uri': 'edit/main/docs/',
    'features': [
        'navigation.tabs',
        'navigation.sections',
        'navigation.top',
        'search.suggest',
        'content.code.copy',
    ],
    'palette': [
        {
            'media': '(prefers-color-scheme: light)',
            'scheme': 'default',
            'toggle': {
                'icon': 'material/weather-night',
                'name': 'Switch to dark mode',
            },
        },
        {
            'media': '(prefers-color-scheme: dark)',
            'scheme': 'slate',
            'toggle': {
                'icon': 'material/weather-sunny',
                'name': 'Switch to light mode',
            },
        },
    ],
    # Keep every level of the navigation tree expanded, as navigation_depth
    # 4 did under the previous theme; the theme has no depth option.
    'globaltoc_collapse': False,
    'consent': {
        'title': 'Cookie consent',
        'description': (
            'This site uses cookies to measure visits and improve the '
            'documentation. You can accept or reject analytics cookies.'
        ),
    },
}

# Google Analytics through the theme's native integration. The measurement
# id comes from the environment at build time (a repository variable in
# docs.yml); without it the build emits no analytics code at all.
_ga_key = os.environ.get('GOOGLE_ANALYTICS_KEY', '').strip()
if _ga_key:
    html_theme_options['analytics'] = {
        'provider': 'google',
        'property': _ga_key,
    }

# -- Options for autodoc extension -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# The theme files a class's constructor parameters under ``Class.__init__``,
# so ``__init__`` is not documented as a separate member (that would define
# every parameter twice). Its docstring is folded into the class instead.
autoclass_content = 'both'

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}
