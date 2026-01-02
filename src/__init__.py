"""
Functions to call from outside the src package: to be called from main scripts.
"""

from .forward_simulation import test_forward_simulation
from .observation import test_observations
from .quadrature import test_quadrature

functions = [test_forward_simulation, test_observations, test_quadrature]
