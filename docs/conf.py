# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'BayesLIM'
copyright = '2025, Nicholas Kern'
author = 'Nicholas Kern'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['nbsphinx', 'myst_parser', 'sphinx.ext.mathjax']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
master_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    '.md': 'markdown',
}
myst_enable_extensions = [
    "amsmath",
    "dollarmath"
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ["source/_static"]
html_logo = "source/_static/img/icon_dark.jpg"
html_theme_options = {
    "source_directory": "docs/",
    "source_repository": "https://github.com/BayesLIM/BayesLIM",
    "source_branch": "main",
}

