"""
Defines all numerical values and objects for testing architecture and robustness.
"""

from datetime import datetime, timedelta
from math import asin, degrees
from pathlib import Path
from random import uniform
from typing import Optional

from numpy import pi
from sympy import Symbol

from .parameters import (
    ArcParameters,
    Parameters,
    ParameterSampling,
    SimulationParameters,
    TimeDependentParameter,
)
from .station import Station, StationPosition
from .test_forces import ParameterizedTestForceParameters, TimeTestForceParameters

TEST_OUTPUT_PATH = Path("test")
TEST_ARC_LENGTH = 10000.0
TEST_TIME_STEP = 10.0
TEST_ARC_START_DATETIME = datetime(
    year=2000, month=1, day=1, hour=0, minute=0, second=0, microsecond=0
)
TEST_ARC_PARAMETERS = ArcParameters(
    time_step=TEST_TIME_STEP,
    arc_start_datetime=TEST_ARC_START_DATETIME,
    arc_length=TEST_ARC_LENGTH,
    lagrange_interpolation_order=4,
    arc_id="test_arc_id",
)
TEST_SIMULATED_FORCES: dict[str, Optional[Parameters]] = {
    "central_body_attraction": None,
    "parameterized_test_force": ParameterizedTestForceParameters(
        dummy_parameter=Symbol("dummy_parameter"), dummy_parameter_value=1e-5
    ),
    "time_test_force": TimeTestForceParameters(
        time_dependent_parameter=TimeDependentParameter(
            symbol="time_dependent_parameter",
            arc_start_datetime=TEST_ARC_START_DATETIME,
            parameter_sampling=ParameterSampling(
                datetime_sampling_values=[
                    TEST_ARC_START_DATETIME + timedelta(seconds=seconds)
                    for seconds in range(0, int(TEST_ARC_LENGTH + 1), 1000)
                ],
                parameter_values=[
                    1e-5 * delta_value * TEST_TIME_STEP
                    for delta_value in range(int(TEST_ARC_LENGTH // TEST_TIME_STEP) + 1)
                ],
            ),
        )
    ),
}
TEST_SIMULATION_PARAMETERS = SimulationParameters(
    arc_parameters=TEST_ARC_PARAMETERS,
    simulated_forces=TEST_SIMULATED_FORCES,
    parameter_expressions={
        "Earth_radius": Symbol("Earth_radius"),
        "gravitational_parameter": Symbol("gravitational_parameter"),
        "t": Symbol("t"),
        "semi_major_axis": Symbol("semi_major_axis"),
        "eccentricity": Symbol("eccentricity"),
        "inclination": Symbol("inclination"),
        "right_ascension_ascending_node": Symbol("right_ascension_ascending_node"),
        "argument_of_periapsis": Symbol("argument_of_periapsis"),
        "true_anomaly": Symbol("true_anomaly"),
        "arc_start_Earth_rotation_angle": Symbol("arc_start_Earth_rotation_angle"),
        "Earth_rotation_angular_speed": Symbol("Earth_rotation_angular_speed"),
    },
    terminal_parameter_values={
        "gravitational_parameter": 3.986004418e14,
        "Earth_radius": 6.371e6,
        "semi_major_axis": 7e6,
        "eccentricity": 0.05,
        "inclination": 45.0,
        "right_ascension_ascending_node": 0.0,
        "argument_of_periapsis": 0.0,
        "true_anomaly": 0.0,
        "arc_start_Earth_rotation_angle": 0.0,
        "Earth_rotation_angular_speed": 2 * pi / 86164,
    },
)
TEST_STATION_QUANTITY = 100
TEST_STATIONS = {
    "station_test_id_"
    + str(k): Station(
        name="station_test_id_" + str(k),
        station_position=StationPosition(
            latitude=degrees(asin(uniform(a=-1, b=1))),
            longitude=uniform(-180, 180),
            altitude=0.0,
        ),
    )
    for k in range(TEST_STATION_QUANTITY)
}

for station_id, station in TEST_STATIONS.items():

    TEST_SIMULATION_PARAMETERS.update_expressions(
        new_expressions=station.get_parameter_expressions()
    )
    TEST_SIMULATION_PARAMETERS.update_terminal_parameter_values(
        new_expressions=station.get_terminal_parameters()
    )
