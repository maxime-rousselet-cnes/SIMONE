from datetime import datetime
from inspect import getmembers, isfunction
from sys import modules

from sympy import Expr, Matrix, MutableDenseMatrix

from utils import norm, position

from .parameters import TimeDependentParameter


class ParameterizedTestForceParameters:

    dummy_parameter: float


def parameterized_test_force(
    state_vector: MutableDenseMatrix,
    parameters: dict[str, Expr],
    parameter_test_force_parameters: ParameterizedTestForceParameters,
) -> MutableDenseMatrix:
    """
    Virtual force to test the simulation's architecture. Analog to a constant Earth radiation
    pressure. Has a scale parameter.
    """

    return (
        parameter_test_force_parameters.dummy_parameter
        * (parameters["Earth_radius"] / norm(vector=position(state_vector=state_vector))) ** 2
        * position(state_vector=state_vector)
    )


class ParameterizedTimeTestForceParameters:

    time_dependent_parameter: TimeDependentParameter


from .utils import lagrange_polynomial_interpolation


def parameterized_time_test_force(
    state_vector: MutableDenseMatrix,
    parameters: dict[str, Expr],
    parameterized_time_test_force_parameters: ParameterizedTimeTestForceParameters,
    time: Expr,
) -> MutableDenseMatrix:
    """
    Virtual force to test the simulation's architecture. Constant and radial. Has a time-dependent scale parameter.
    """

    return (
        parameter_test_force_parameters.dummy_parameter
        * (parameters["Earth_radius"] / norm(vector=position(state_vector=state_vector))) ** 2
        * position(state_vector=state_vector)
    )


test_forces = {
    name: force
    for name, force in getmembers(modules[__name__], isfunction)
    if force.__module__ == __name__
}
