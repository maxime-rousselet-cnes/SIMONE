"""
Defines General parameter-like classes. Force-specific parameter classes yet to be defined with the
said force.
"""

from datetime import datetime
from typing import Optional

from sympy import Expr, Symbol


def datetime_differences(t_1: datetime, t_2: datetime) -> float:
    """
    Returns the difference between two datetime instances in seconds.
    """

    return (t_2 - t_1).total_seconds()


class ForceParameters:
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


class SimulationParameters:

    time_step: float
    satellite: str
    simulated_forces: dict[str, Optional[ForceParameters]]
    arc_start_datetime: datetime
    arc_length: float

    def __init__(
        self,
        time_step: float,
        satellite: str,
        simulated_forces: dict[str, Optional[ForceParameters]],
        arc_start_datetime: datetime,
        arc_length: float,
    ) -> None:
        """ """

        self.time_step = time_step
        self.satellite = satellite
        self.simulated_forces = simulated_forces
        self.arc_start_datetime = arc_start_datetime
        self.arc_length = arc_length


class TimeDependentParameter(ForceParameters):

    time_sampling_expressions: list[Expr]
    parameter_value_expressions: list[Expr]
    interpolation_order: int
    time_sampling_values: list[datetime]
    parameter_values: list[float]

    def __init__(
        self,
        symbol: str,
        arc_start_datetime: datetime,
        datetime_sampling_values: list[datetime],
        parameter_values: list[float],
        interpolation_order: int = 4,
    ) -> None:
        """ """

        self.time_sampling_expressions: list[Expr] = [
            Symbol(f"t^{symbol}_{i_timestamp}")
            for i_timestamp, _ in enumerate(datetime_sampling_values)
        ]
        self.parameter_value_expressions = [
            Symbol(f"{symbol}_{i_timestamp}") for i_timestamp, _ in enumerate(parameter_values)
        ]
        self.interpolation_order = interpolation_order
        self.time_sampling_values = [
            datetime_differences(t_1=arc_start_datetime, t_2=datetime_sample)
            for datetime_sample in datetime_sampling_values
        ]
        self.parameter_values = parameter_values

    def get_terminal_parameters(self) -> dict[str, float]:
        """ """

        return {
            str(symbol): value
            for symbol, value in zip(self.time_sampling_expressions, self.time_sampling_values)
        } | {
            str(symbol): value
            for symbol, value in zip(self.parameter_value_expressions, self.parameter_values)
        }

    def get_parameter_expressions(self) -> dict[str, Expr]:
        """ """

        return {str(symbol): symbol for symbol in self.time_sampling_expressions} | {
            str(symbol): symbol for symbol in self.parameter_value_expressions
        }
