from sympy import Expr, MutableDenseMatrix

from utils import norm, position

from .parameters import ForceParameters, TimeDependentParameter
from .utils import piecewise_lagrange


class ParameterizedTestForceParameters(ForceParameters):

    dummy_parameter: Expr


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


class TimeTestForceParameters:

    time_dependent_parameter: TimeDependentParameter


def parameterized_time_test_force(
    state_vector: MutableDenseMatrix,
    parameters: dict[str, Expr],
    time_test_force_parameters: TimeTestForceParameters,
    time: Expr,
) -> MutableDenseMatrix:
    """
    Virtual force to test the simulation's architecture. Constant and radial. Has a time-dependent
    scale parameter.
    """

    return (
        piecewise_lagrange(
            t=time,
            t_syms=time_test_force_parameters.time_dependent_parameter.time_sampling_expressions,
            y_syms=time_test_force_parameters.time_dependent_parameter.parameter_value_expressions,
            order=time_test_force_parameters.time_dependent_parameter.interpolation_order,
        )
        * (parameters["Earth_radius"] / norm(vector=position(state_vector=state_vector))) ** 2
        * position(state_vector=state_vector)
    )
