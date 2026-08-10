"""Sphinx configuration for the musicality API docs."""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "musicality"
copyright = "2026, luczeng"
author = "luczeng"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_inherit_docstrings = False
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
