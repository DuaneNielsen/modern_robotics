#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='robotics',
      version='0.0.1',
      packages=find_packages(),
      install_requires=[
          'torch',
          'vpython'
      ],
      extras_require={
          'dev': [
              'pytest',
              'matplotlib'
          ]
      }
      )
