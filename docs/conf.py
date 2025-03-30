# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'BayesLIM'
copyright = '2025, Nicholas Kern'
author = 'Nicholas Kern'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['nbsphinx', 'myst_parser']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
master_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ["source/_static"]
html_logo = "source/_static/img/icon_dark.jpg"
html_theme_options = {
    "source_directory": "docs/",
    "source_repository": "https://github.com/nkern/BayesLIM",
    "source_branch": "main",
}

