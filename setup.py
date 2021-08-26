from setuptools import setup
import os
import ast

# get version from __init__.py
init_file = os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), 'bayeslim/__init__.py')
with open(init_file, 'r') as f:
    lines = f.readlines()
    for l in lines:
        if "__version__" in l:
            version = ast.literal_eval(l.split('=')[1].strip())

def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths


data_files = package_files('bayeslim', 'data') + package_files('bayeslim', 'config')

setup(
    name            = 'bayeslim',
    version         = version,
    license         = 'MIT',
    description     = 'Bayesian Calibration and Signal Extraction for Line Intensity Mapping',
    author          = 'Nicholas Kern',
    url             = "http://github.com/nkern/bayeslim",
    package_data    = {'bayeslim': data_files},
    include_package_data = True,
    packages        = ['bayeslim'],
    package_dir     = {'bayeslim': 'bayeslim'},
    zip_safe        = False
    )
