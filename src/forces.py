"""
Regroups all forces defined in the different force modules.
"""

from typing import Callable

from sympy import Expr, MutableDenseMatrix

from .gravitational_forces import gravitational_forces
from .parameters import SimulationParameters
from .test_forces import test_forces

all_forces = gravitational_forces | test_forces


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
