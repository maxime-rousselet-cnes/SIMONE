"""
Dummy accelerations to test the architecture.
"""

from sympy import Expr, MutableDenseMatrix, simplify, srepr

from .parameters import Parameters, TimeDependentParameter
from .utils import norm, position


class ParameterizedTestForceParameters(Parameters):
    """
    Force that has a non-invertible parameter (to represent typically, maximum degree of spherical
    harmonics) and an invertible parameter to scale a planetary radiation force.
    """

    dummy_parameter: Expr
    dummy_parameter_value: float
    dummy_fixed_parameter: float

    def __init__(
        self,
        dummy_parameter: Expr | str,
        dummy_parameter_value: float,
        dummy_fixed_parameter: float = 4.0,
    ) -> None:

        self.dummy_parameter = simplify(expr=dummy_parameter)
        self.dummy_parameter_value = dummy_parameter_value
        self.dummy_fixed_parameter = dummy_fixed_parameter

    def get_terminal_parameters(self) -> dict[str, float]:
        """
        Straightforward definition since this dummy parameter class only contains a single
        invertible parameter.
        """

        return {r"p_{dummy\ parameter}": self.dummy_parameter_value}

    def get_parameter_expressions(self) -> dict[str, Expr]:
        """
        Straightforward definition since this dummy parameter class only contains a single
        invertible parameter.
        """

        return {r"p_{dummy\ parameter}": self.dummy_parameter}

    def to_serializable(self) -> dict[str, float | str]:
        """
        To (.JSON) files.
        """

        return {
            "dummy_parameter": srepr(self.dummy_parameter),
            "dummy_parameter_value": self.dummy_parameter_value,
            "dummy_fixed_parameter": self.dummy_fixed_parameter,
        }


def parameterized_test_force(
    state_vector: MutableDenseMatrix,
    parameter_expressions: dict[str, Expr],
    parameter_test_force_parameters: ParameterizedTestForceParameters,
) -> MutableDenseMatrix:
    """
    Virtual force to test the simulation's architecture. Analog to a constant Earth radiation
    pressure. Has a scale parameter.
    """

    return (
        parameter_test_force_parameters.dummy_parameter
        * (
            parameter_expressions[r"R_{Earth\ radius}"]
            / norm(vector=position(state_vector=state_vector))
        )
        ** parameter_test_force_parameters.dummy_fixed_parameter
        * position(state_vector=state_vector)
        / norm(vector=position(state_vector=state_vector))
    )


class TimeTestForceParameters(Parameters):
    """
    Dummy force to test time-dependent parameters.
    """

    time_dependent_parameter: TimeDependentParameter

    def __init__(self, time_dependent_parameter: TimeDependentParameter | dict) -> None:

        self.time_dependent_parameter = (
            time_dependent_parameter
            if isinstance(time_dependent_parameter, TimeDependentParameter)
            else TimeDependentParameter(**time_dependent_parameter)
        )
        self.time_dependent_parameter.time_sampling_expressions = [
            simplify(expr=expression)
            for expression in self.time_dependent_parameter.time_sampling_expressions
        ]
        self.time_dependent_parameter.parameter_expressions = [
            simplify(expr=expression)
            for expression in self.time_dependent_parameter.parameter_expressions
        ]

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

    def to_serializable(self) -> dict[str, float | str]:
        """
        Straightforward definition since this dummy parameter class only contains a single
        time-dependent parameter.
        """

        return {"time_dependent_parameter": self.time_dependent_parameter.to_serializable()}


def time_test_force(
    state_vector: MutableDenseMatrix,
    parameter_expressions: dict[str, Expr],
    time_test_force_parameters: TimeTestForceParameters,
) -> MutableDenseMatrix:
    """
    Virtual force to test the simulation's architecture. Constant and radial. Has a time-dependent
    scale parameter.
    """

    return (
        time_test_force_parameters.time_dependent_parameter.piecewise_lagrange(
            t=parameter_expressions[r"t"]
        )
        * (
            parameter_expressions[r"R_{Earth\ radius}"]
            / norm(vector=position(state_vector=state_vector))
        )
        ** 2
        * position(state_vector=state_vector)
        / norm(vector=position(state_vector=state_vector))
    )
