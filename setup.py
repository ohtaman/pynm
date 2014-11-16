#!/usr/bin/env python

import os
import sys
from setuptools import setup, find_packages

files = []

with open(os.sep.join((os.path.dirname(__file__), "requirements.txt"))) as f:
    requires = f.read().splitlines()

setup(name="pynm",
      version="0.1",
      author="Mitsuhisa Ohta",
      author_email="ohtamans@gmail.com",
      description="Python Machine Learning Library",
      url="",
      packages=find_packages(),
      package_data={"pynm": files},
      include_package_data=True,
      install_requires=requires,
      test_suite="nose.collector",
      entry_points="""
      [console_scripts]
      pynm-metric = pynm.metric.cli:main
      """)
