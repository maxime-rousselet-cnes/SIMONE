"""
Regroups all forces defined in the different force modules.
"""

from inspect import getmembers, isfunction
from sys import modules
from typing import Callable

from sympy import Expr, MutableDenseMatrix

from .parameters import SimulationParameters

all_forces = {
    name: force
    for name, force in getmembers(
        modules[__name__], isfunction
    )  # TODO: loop on modules in same folder.
    if force.__module__ == __name__
}


def symbolic_propagator(
    simulation_parameters: SimulationParameters,
) -> Callable[[MutableDenseMatrix, dict[str, Expr]], MutableDenseMatrix]:
    """
    Builds a simulation-specific symbolic propagator. Yet to be evaluated for parameters before
    being integrated.
    """

    return lambda state_vector, parameters: sum(
        [
            (
                all_forces[force](state_vector, parameters, force_parameters)
                if force_parameters
                else all_forces[force](state_vector, parameters)
            )
            for force, force_parameters in simulation_parameters.simulated_forces
        ]
    )
