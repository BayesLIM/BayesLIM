# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'BayesLIM'
copyright = '2025, Nicholas Kern'
author = 'Nicholas Kern'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['nbsphinx']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ["source/_static"]
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/nkern/BayesLIM",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
}


import os
import sys

readme_file = os.path.join(os.path.abspath("../"), "README.rst")
index_file = os.path.join(os.path.abspath("../docs"), "index.rst")


def build_custom_docs(app):
    sys.path.append(os.getcwd())
    import make_index

    make_index.write_index_rst(readme_file, write_file=index_file)


def setup(app):
    app.connect("builder-inited", build_custom_docs)

