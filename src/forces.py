"""
Regroups all forces defined in the different force modules.
"""

import pkgutil
from importlib import import_module
from inspect import getmembers, isfunction
from sys import modules
from typing import Callable

from sympy import Expr, Matrix, MutableDenseMatrix

from .parameters import SimulationParameters
from .utils import speed

pkg = modules[__name__].__package__ or __name__.rpartition(".")[0]
all_forces: dict[str, Callable] = {}

for _, modname, ispkg in pkgutil.iter_modules(getattr(import_module(pkg), "__path__", [])):

    if "force" in modname:

        module = import_module(f"{pkg}.{modname}")

        for name, func in getmembers(module, isfunction):

            if func.__module__ == module.__name__:

                all_forces[name] = func

all_forces.pop("__annotate__")


def symbolic_propagator(
    state_vector: MutableDenseMatrix,
    parameters: dict[str, Expr],
    simulation_parameters: SimulationParameters,
) -> MutableDenseMatrix:
    """
    Builds a simulation-specific symbolic propagator. Yet to be evaluated for parameters before
    being integrated on state and time.
    """

    return Matrix.vstack(
        speed(state_vector=state_vector),
        sum(
            [
                (
                    all_forces[force](state_vector, parameters, force_parameters)
                    if force_parameters
                    else all_forces[force](state_vector, parameters)
                )
                for force, force_parameters in simulation_parameters.simulated_forces.items()
            ],
            start=MutableDenseMatrix.zeros(rows=3, cols=1),
        ),
    )
