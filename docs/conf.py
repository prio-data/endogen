# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.insert(0, os.path.abspath(".."))

project = "ENDOGEN"
copyright = "2023, PRIO"
author = "PRIO-DATA team"
release = "0.01"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx_inline_tabs",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# https://sphinx-copybutton.readthedocs.io/en/latest/use.html#strip-and-configure-input-prompts-for-code-cells
copybutton_prompt_text = r">>> |\.\.\. |\$ |\(endogen_env\) \$ "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#00718f",
        "color-brand-content": "#00718f",
        "color-background-secondary": "#fafafa",
        "color-problematic": "#c41230",
    },
    "dark_css_variables": {
        "color-brand-primary": "#00718f",
        "color-brand-content": "#00718f",
        "color-problematic": "#e87722",
    },
}
