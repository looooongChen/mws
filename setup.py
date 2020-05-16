from __future__ import absolute_import, print_function
from setuptools import setup, find_packages, Extension
# from numpy.distutils.misc_util import get_numpy_include_dirs
import numpy as np
from os import path
from glob import glob

_dir = path.dirname(__file__)

with open(path.join(_dir,'README.md'), encoding="utf-8") as f:
    long_description = f.read()

qhull_root = path.join(_dir, 'stardist', 'lib', 'qhull_src', 'src')
qhull_src = sorted(glob(path.join(qhull_root, '*', '*.c*')))[::-1]


setup(
    name='mws',
    version='0.1.0',
    description='mutex watershed',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/looooongChen/mws',
    author='Long Chen',
    author_email='looooong.chen@gmail.com',
    license='BSD 3-Clause License',
    packages=find_packages(),
    python_requires='>=3.5',

    ext_modules=[
        Extension(
            'mws.src.mws',
            sources=['mws/src/mws.cpp'],
            include_dirs=[np.get_include()],
        )
    ],

    install_requires=[
        'numpy'
    ],
)
