#!/usr/bin/env python

import os
import sys
from setuptools import setup, find_packages

files = []


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name="pynm",
      version="0.1",
      author="Mitsuhisa Ohta",
      author_email="ohtamans@gmail.com",
      description="Python Machine Learning Library",
      long_description=read('README.md'),
      url="",
      license="MIT",
      packages=find_packages(),
      package_data={"pynm": files},
      include_package_data=True,
      install_requires=read('requirements.txt').splitlines(),
      test_suite="nose.collector",
      entry_points="""
      [console_scripts]
      pynm-metric = pynm.metric.cli:main
      pynm-bandit = pynm.reinforce.bandit.cli:main
      """)
