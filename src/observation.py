from datetime import datetime
from typing import Enum


class ObservationType(Enum):

    STATION_TO_SATELLITE_LASER_RANGING = "station to satellite laser ranging"


class ObservationDescription:

    id: str
    date_time: datetime
    observation_type: ObservationType


class MeasurementDescription(ObservationDescription):

    value: float
