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
from .utils import STATE_VECTOR_LINE, STATE_VECTOR_MATRIX, speed

pkg = modules[__name__].__package__ or __name__.rpartition(".")[0]
all_forces: dict[str, Callable] = {}

for _, modname, ispkg in pkgutil.iter_modules(getattr(import_module(pkg), "__path__", [])):

    if "force" in modname:

        module = import_module(f"{pkg}.{modname}")

        for name, func in getmembers(module, isfunction):

            if func.__module__ == module.__name__:

                all_forces[name] = func


def symbolic_propagator(
    state_vector: MutableDenseMatrix,
    simulation_parameters: SimulationParameters,
) -> MutableDenseMatrix:
    """
    Builds a simulation-specific symbolic propagator. Yet to be evaluated for parameter expressions
    before being integrated on state and time.
    """

    return Matrix.vstack(
        speed(state_vector=state_vector),
        sum(
            (
                (
                    all_forces[force](
                        state_vector, simulation_parameters.parameter_expressions, force_parameters
                    )
                    if force_parameters
                    else all_forces[force](
                        state_vector, simulation_parameters.parameter_expressions
                    )
                )
                for force, force_parameters in simulation_parameters.simulated_forces.items()
            ),
            start=MutableDenseMatrix.zeros(rows=3, cols=1),
        ),
    )


def vetor_variation_equation(dynamic: MutableDenseMatrix, parameter: Expr) -> MutableDenseMatrix:
    """
    Applies the variation method to algebraically derive the time-dependent behavior of a partial
    derivative to integrate on the satellite's dynamic quadrature points.
    """

    return Matrix(
        [
            [variation_equation(expression=expression, parameter=parameter)]
            for expression in dynamic.flat()
        ]
    )


def variation_equation(expression: Expr, parameter: Expr) -> Expr:
    """
    Applies the variation method to algebraically derive a partial derivative expression with
    respect to aparameter.
    """

    return MutableDenseMatrix(
        [expression.diff(state_parameter) for state_parameter in STATE_VECTOR_LINE]
    ).dot(b=STATE_VECTOR_MATRIX.diff(parameter)) + expression.diff(parameter)
