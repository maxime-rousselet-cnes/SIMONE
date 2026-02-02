"""
Defines General parameter-like classes. Force-specific parameter classes yet to be defined with the
said force.
"""

from dataclasses import dataclass
from datetime import datetime

from numpy import ndarray
from sympy import Expr, Matrix, MutableDenseMatrix, Piecewise, Symbol, srepr

from .utils import piecewise_lagrange


def datetime_differences(t_1: datetime, t_2: datetime) -> float:
    """
    Returns the difference between two datetime instances in seconds.
    """

    return (t_2 - t_1).total_seconds()


class Parameters:
    """
    Abstract class from which every force-specific has to inherit.
    A force-specific parameters class has to redefine properly these 3 methods and a constructor
    ensuring self-consistence.
    """

    def get_terminal_parameters(self) -> dict[str, float]:
        """
        Returns terminal (numeric) parameters required by the force.

        Default implementation returns an empty dict. Subclasses should override to provide a
        mapping of symbol names to their numeric values.
        """

        return {}

    def get_parameter_expressions(self) -> dict[str, Expr]:
        """
        Returns parameter expressions required by the force.

        Default implementation returns an empty dict. Subclasses should override to provide a
        mapping of symbol names to their symbolic expressions.
        """

        return {}

    def to_serializable(self) -> dict[str, float | str]:
        """
        To save in (.JSON) files.
        """

        return {}


@dataclass
class ArcParameters:
    """
    Time characteristics of an orbit arc.
    """

    time_step: float
    arc_start_datetime: datetime
    arc_length: float
    lagrange_interpolation_order: int
    arc_id: str

    # Defines whether the arc is computed for initial simulated measurement generation purposes or
    # not.
    is_initial: bool

    def generate_lagrange_kernels(self, t: Expr) -> MutableDenseMatrix:
        """
        For minimizing complexity use.
        """

        return Matrix(
            [
                [
                    piecewise_lagrange(
                        t=t,
                        t_syms=[
                            Symbol(rf"t_{i}") for i in range(2 * self.lagrange_interpolation_order)
                        ],
                        # k-th component.
                        y_syms=[
                            Symbol(rf"\theta^{j}_{i}")
                            for i in range(2 * self.lagrange_interpolation_order)
                        ],
                        order=self.lagrange_interpolation_order,
                    )
                ]
                for j in range(6)
            ]
        )


@dataclass
class TimeDependentParameter(Parameters):
    """
    General description of a time-dependent parameter to be interpolated by when defining the force
    that uses it.
    """

    interpolation_order: int
    time_sampling_expressions: list[Expr]
    parameter_expressions: list[Expr]
    time_sampling_values: list[float]
    parameter_values: list[float]

    def piecewise_lagrange(self, t: Expr) -> Piecewise:
        """
        Applies Lagrange interpolation of wanted order.
        """

        return piecewise_lagrange(
            t=t,
            t_syms=self.time_sampling_expressions,
            y_syms=self.parameter_expressions,
            order=self.interpolation_order,
        )

    def get_terminal_parameters(self) -> dict[str, float]:
        """
        Makes sure to consider the time-dependent parameter values as well as its time sampling
        values.
        """

        return {
            str(symbol): value
            for symbol, value in zip(self.time_sampling_expressions, self.time_sampling_values)
        } | {
            str(symbol): value
            for symbol, value in zip(self.parameter_expressions, self.parameter_values)
        }

    def get_parameter_expressions(self) -> dict[str, Expr]:
        """
        Makes sure to consider the time-dependent parameter symbols as well as its time sampling
        symbols.
        """

        return {str(symbol): symbol for symbol in self.time_sampling_expressions} | {
            str(symbol): symbol for symbol in self.parameter_expressions
        }

    def to_serializable(self) -> dict[str, float | str]:
        """
        Gets needed information to build back the instance from the constructor.
        """

        return {
            "interpolation_order": self.interpolation_order,
            "time_sampling_expressions": [
                srepr(expression) for expression in self.time_sampling_expressions
            ],
            "parameter_expressions": [
                srepr(expression) for expression in self.parameter_expressions
            ],
            "time_sampling_values": list(self.time_sampling_values),
            "parameter_values": list(self.parameter_values),
        }


def generate_time_dependent_parameter(
    symbol: str,
    arc_start_datetime: datetime,
    datetime_sampling_values: list[datetime],
    parameter_values: list[float] | ndarray[float],
    interpolation_order: int = 4,
) -> TimeDependentParameter:
    """
    Formally builds an instance of TimeDependentParameter. Different of the base constructor, which
    assumes already correctly defined attributes.
    """

    assert len(parameter_values) == len(datetime_sampling_values)

    return TimeDependentParameter(
        interpolation_order=interpolation_order,
        time_sampling_expressions=[
            Symbol(rf"t^{symbol}" + "{" + rf"{i_timestamp}" + "}")
            for i_timestamp, _ in enumerate(datetime_sampling_values)
        ],
        parameter_expressions=[
            Symbol(rf"{symbol}_" + "{" + rf"{i_timestamp}" + "}")
            for i_timestamp, _ in enumerate(datetime_sampling_values)
        ],
        time_sampling_values=[
            datetime_differences(t_1=arc_start_datetime, t_2=datetime_sample)
            for datetime_sample in datetime_sampling_values
        ],
        parameter_values=list(parameter_values),
    )
