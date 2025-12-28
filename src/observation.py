"""
Describes observation types for a general observation framework.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ObservationType(Enum):
    """
    Lists the different types of observations that are handled.
    """

    STATION_TO_SATELLITE_LASER_RANGING = "station to satellite laser ranging"


@dataclass
class ObservationDescription:
    """
    Abstract description of an observation. To be inherited by specific observation types.
    """

    id: str
    date_time: datetime
    observation_type: ObservationType


@dataclass
class MeasurementDescription(ObservationDescription):
    """
    A measurement is the result of an observation.
    """

    value: float
