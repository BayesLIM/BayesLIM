from setuptools import setup
import os
import ast

# get version from __init__.py
init_file = os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), 'bayescal/__init__.py')
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


data_files = package_files('bayescal', 'data') + package_files('bayescal', 'config')

setup(
    name            = 'bayescal',
    version         = version,
    license         = 'MIT',
    description     = 'Bayesian Interferometric Calibration and Imaging',
    author          = 'Nicholas Kern',
    url             = "http://github.com/nkern/bayescal",
    package_data    = {'bayescal': data_files},
    include_package_data = True,
    packages        = ['bayescal'],
    package_dir     = {'bayescal': 'bayescal'},
    zip_safe        = False
    )