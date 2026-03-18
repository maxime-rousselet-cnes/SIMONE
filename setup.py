"""
To install via pip install -e . in the root of the repository. This will make the package available
in the current environment.
"""

from setuptools import find_packages, setup

setup(
    name="simone",
    packages=find_packages(),
    version="0.0.1",
    description="Symbolic Implementation for Modeling of Orbits and Normal Equations.",
    author="Maxime Rousselet",
)
