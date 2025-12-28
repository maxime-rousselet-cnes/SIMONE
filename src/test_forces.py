"""
Dummy accelerations to test the architecture.
"""

from sympy import Expr, MutableDenseMatrix

from .parameters import ForceParameters, TimeDependentParameter
from .utils import norm, piecewise_lagrange, position


class ParameterizedTestForceParameters(ForceParameters):
    """
    Force that has a non-invertible parameter (to represent typically, maximum degree of spherical
    harmonics) and an invertible parameter to scale a planetary radiation force.
    """

    dummy_parameter: Expr
    dummy_parameter_value: float
    dummy_fixed_parameter: float

    def __init__(
        self,
        dummy_parameter: Expr,
        dummy_parameter_value: float,
        dummy_fixed_parameter: float = 4.0,
    ) -> None:

        self.dummy_parameter = dummy_parameter
        self.dummy_parameter_value = dummy_parameter_value
        self.dummy_fixed_parameter = dummy_fixed_parameter

    def get_terminal_parameters(self) -> dict[str, float]:
        """
        straightforward definition since this dummy parameter class only contains a single
        invertible parameter.
        """

        return {"dummy_parameter": self.dummy_parameter_value}

    def get_parameter_expressions(self) -> dict[str, Expr]:
        """
        straightforward definition since this dummy parameter class only contains a single
        invertible parameter.
        """

        return {"dummy_parameter": self.dummy_parameter}


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
        * (parameters["Earth_radius"] / norm(vector=position(state_vector=state_vector)))
        ** parameter_test_force_parameters.dummy_fixed_parameter
        * position(state_vector=state_vector)
        / norm(vector=position(state_vector=state_vector))
    )


class TimeTestForceParameters(ForceParameters):
    """
    Dummy force to test time-dependent parameters.
    """

    time_dependent_parameter: TimeDependentParameter

    def __init__(self, time_dependent_parameter: TimeDependentParameter) -> None:

        self.time_dependent_parameter = time_dependent_parameter

    def get_terminal_parameters(self) -> dict[str, float]:
        """
        Straightforward definition since this dummy parameter class only contains a single
        time-dependent parameter.
        """

        return self.time_dependent_parameter.get_terminal_parameters()

    def get_parameter_expressions(self) -> dict[str, Expr]:
        """
        Straightforward definition since this dummy parameter class only contains a single
        time-dependent parameter.
        """

        return self.time_dependent_parameter.get_parameter_expressions()


def time_test_force(
    state_vector: MutableDenseMatrix,
    parameters: dict[str, Expr],
    time_test_force_parameters: TimeTestForceParameters,
) -> MutableDenseMatrix:
    """
    Virtual force to test the simulation's architecture. Constant and radial. Has a time-dependent
    scale parameter.
    """

    return (
        piecewise_lagrange(
            t=parameters["t"],
            t_syms=time_test_force_parameters.time_dependent_parameter.time_sampling_expressions,
            y_syms=time_test_force_parameters.time_dependent_parameter.parameter_value_expressions,
            order=time_test_force_parameters.time_dependent_parameter.interpolation_order,
        )
        * (parameters["Earth_radius"] / norm(vector=position(state_vector=state_vector))) ** 2
        * position(state_vector=state_vector)
        / norm(vector=position(state_vector=state_vector))
    )
