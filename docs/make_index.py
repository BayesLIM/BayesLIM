"""
Reformat README.rst relative paths
"""

import inspect
import os

def write_index_rst(readme_file, write_file=None):
	with open(readme_file) as f:
		txt = ''.join(f.readlines())

	# replace relative paths to docs/source/_static
	txt.replace('docs/source/_static', 'source/_static')

	# add other txt
	txt += (
		"\n"
		"Contents\n"
		"========\n"
		".. toctree::\n"
		"   :maxdepth: 1\n"
		""
		"   examples\n"
		)

	txt.replace("\u2018", "'").replace("\u2019", "'").replace("\xa0", " ")

	print(txt)

	with open(write_file, "w") as F:
		F.write(txt)