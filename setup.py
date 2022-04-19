# !/usr/bin/env python
"""Installation script for setuptools"""

from setuptools import find_packages, setup

setup(
  name='admon',
  version='0.0.1 dev',
  description='Understand airport delay pattern with DMoN.',
  author='Juanwu Lu, Ying Zhou',
  packages=find_packages(),
  install_requires=[
    'numpy>=1.8.0',
    'torch>=1.4.0',
    'tensorboard',
  ],
  python_requires='>=3.7'
)
