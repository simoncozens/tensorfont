#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

config = {
    'name': 'tensorfont',
    'author': 'Simon Cozens',
    'author_email': 'simon@simon-cozens.org',
    'url': 'https://github.com/simoncozens/tensorfont',
    'description': 'Turn font glyphs into numpy arrays',
    'long_description': open('README.rst', 'r').read(),
    'license': 'MIT',
    'version': '0.0.3',
    'install_requires': ["scikit-image", "numpy", "freetype-py", "scipy"],
    'classifiers': [
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta"

    ],
    'packages': find_packages(),
}

if __name__ == '__main__':
    setup(**config)
