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

    pass


class SimulationParameters:

    time_step: float
    maximum_degree: int
    satellite: str
    simulated_forces: dict[str, Optional[ForceParameters]]
    initial_datetime: datetime
    arc_length: float


class SingleParameter:
    expression: Expr
    value: float


class TimeDependentParameter:

    arc_time_sampling: list[SingleParameter]
    values: list[SingleParameter]
    symbol: str
    order: int

    def __init__(
        self,
        symbol: str,
        arc_start_datetime: datetime,
        time_sampling: list[datetime],
        values: list[float],
        order: int,
    ) -> None:
        """ """

        self.symbol = symbol
        self.arc_time_sampling = [
            SingleParameter(
                expression=Symbol(f"t^{symbol}_{i_timestamp}"),
                value=datetime_differences(t_1=arc_start_datetime, t_2=timestamp_datetime),
            )
            for i_timestamp, timestamp_datetime in enumerate(time_sampling)
        ]
        self.values = [
            SingleParameter(expression=Symbol(f"{symbol}_{i_timestamp}"), value=value)
            for i_timestamp, value in enumerate(values)
        ]
        self.order = order
