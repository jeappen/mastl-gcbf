#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="gcbfplus",
    version="0.1.0",
    description='Jax Official Implementation of CoRL Paper: Joe Eappen, '
                'Zikang Xiong, Dipam Patel, Aniket Bera, Suresh Jagannathan: '
                "Scaling Safe Multi-Agent Control for Signal Temporal Logic Specifications",
    author="Joe Eappen",
    author_email="jeappen@purdue.edu",
    url="https://github.com/jeappen/mastl-gcbf",
    install_requires=[],
    packages=find_packages(),
)
