"""
Defines General parameter-like classes. Force-specific parameter classes yet to be defined with the
said force.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sympy import Expr, Matrix, MutableDenseMatrix, Symbol

from .utils import piecewise_lagrange


def datetime_differences(t_1: datetime, t_2: datetime) -> float:
    """
    Returns the difference between two datetime instances in seconds.
    """

    return (t_2 - t_1).total_seconds()


class Parameters:
    """
    Abstract class from which every force-specific has to inherit.
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

    def generate_lagrange_kernels(self) -> MutableDenseMatrix:
        """
        For minimizing complexity use.
        """

        return Matrix(
            [
                [
                    piecewise_lagrange(
                        t=Symbol("t"),
                        t_syms=[
                            Symbol(f"t_{i}") for i in range(2 * self.lagrange_interpolation_order)
                        ],
                        # k-th component.
                        y_syms=[
                            Symbol(f"theta^{j}_{i}")
                            for i in range(2 * self.lagrange_interpolation_order)
                        ],
                        order=self.lagrange_interpolation_order,
                    )
                ]
                for j in range(6)
            ]
        )


class SimulationParameters:
    """
    All parameters required for a forward simulation.
    """

    arc_parameters: ArcParameters
    simulated_forces: dict[str, Optional[Parameters]]
    lagrange_kernels: MutableDenseMatrix
    parameter_expressions: dict[str, Expr]
    terminal_parameter_values: dict[str, float]

    def __init__(
        self,
        arc_parameters: ArcParameters,
        simulated_forces: dict[str, Optional[Parameters]],
        parameter_expressions: dict[str, Expr],
        terminal_parameter_values: dict[str, float],
    ) -> None:

        self.arc_parameters = arc_parameters
        self.simulated_forces = simulated_forces
        self.lagrange_kernels = self.arc_parameters.generate_lagrange_kernels()
        self.parameter_expressions = parameter_expressions
        self.terminal_parameter_values = terminal_parameter_values

        for _, force_parameters in simulated_forces.items():

            if force_parameters:

                self.update_expressions(
                    new_expressions=force_parameters.get_parameter_expressions()
                )
                self.update_terminal_parameter_values(
                    new_expressions=force_parameters.get_terminal_parameters()
                )

    def update_expressions(self, new_expressions: dict[str, Expr]) -> None:
        """
        Updates the dictionary of parameter expressions. Need to update the dictionary of terminal
        parameter values too.
        """

        self.parameter_expressions.update(new_expressions)

    def update_terminal_parameter_values(self, new_expressions: dict[str, Expr]) -> None:
        """
        Updates the dictionary of terminal parameter values. Need to update the dictionary of
        parameter expressions.
        """

        self.terminal_parameter_values.update(new_expressions)


@dataclass
class ParameterSampling:
    """
    Sampling definition for a time-dependent parameter.
    """

    datetime_sampling_values: list[datetime]
    parameter_values: list[float]


class TimeDependentParameter(Parameters):
    """
    General description of a time-dependent parameter to be interpolated by when defining the force
    that uses it.
    """

    time_sampling_expressions: list[Expr]
    parameter_value_expressions: list[Expr]
    interpolation_order: int
    parameter_sampling: ParameterSampling

    def __init__(
        self,
        symbol: str,
        arc_start_datetime: datetime,
        parameter_sampling: ParameterSampling,
        interpolation_order: int = 4,
    ) -> None:

        self.time_sampling_expressions: list[Expr] = [
            Symbol(f"t^{symbol}_{i_timestamp}")
            for i_timestamp, _ in enumerate(parameter_sampling.datetime_sampling_values)
        ]
        self.parameter_value_expressions = [
            Symbol(f"{symbol}_{i_timestamp}")
            for i_timestamp, _ in enumerate(parameter_sampling.parameter_values)
        ]
        self.interpolation_order = interpolation_order
        self.time_sampling_values = [
            datetime_differences(t_1=arc_start_datetime, t_2=datetime_sample)
            for datetime_sample in parameter_sampling.datetime_sampling_values
        ]
        self.parameter_values = parameter_sampling.parameter_values

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
            for symbol, value in zip(self.parameter_value_expressions, self.parameter_values)
        }

    def get_parameter_expressions(self) -> dict[str, Expr]:
        """
        Makes sure to consider the time-dependent parameter symbols as well as its time sampling
        symbols.
        """

        return {str(symbol): symbol for symbol in self.time_sampling_expressions} | {
            str(symbol): symbol for symbol in self.parameter_value_expressions
        }
