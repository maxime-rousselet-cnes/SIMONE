"""
Describes a station object and its parameters.
"""

from dataclasses import dataclass
from pathlib import Path
from random import uniform
from typing import Optional

import numpy
from numpy import arcsin, array, asin, dot, ndarray
from numpy.linalg import norm
from pandas import read_csv
from sympy import Expr, Matrix, MutableDenseMatrix, Symbol, cos, sin

from .base_constants import EARTH_GROUND_MASK, EARTH_RADIUS, TEST_OUTPUT_PATH, degrees, radians
from .dynamics import ecef_to_eci
from .parameters import Parameters
from .utils import rotation_matrix


class StationPosition:
    """
    Describes the reference position of a ground station.
    """

    latitude: float
    longitude: float
    altitude: float

    def __init__(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        altitude: float = 0.0,
        earth_ground_mask: Optional[ndarray[bool]] = EARTH_GROUND_MASK,
    ) -> None:

        if latitude and longitude:

            self.latitude = latitude
            self.longitude = longitude
            self.altitude = altitude

        # Creates random station coordinates uniformly on ground.
        else:

            self.altitude = 0.0
            condition = False

            while not condition:

                self.generate_random_station_coordinates()
                condition = self.is_on_ground(earth_ground_mask=earth_ground_mask)

    def generate_random_station_coordinates(self) -> None:
        """
        Uniform on the sphere.
        """

        self.latitude = degrees(asin(uniform(a=-1, b=1)))
        self.longitude = uniform(-180, 180)

    def is_on_ground(self, earth_ground_mask: Optional[ndarray[bool]] = EARTH_GROUND_MASK) -> bool:
        """
        Verifies if the station is on the ground or not (ocean/ice cap).
        """

        return (
            True
            if earth_ground_mask is None
            else earth_ground_mask[
                int((self.latitude + 90) / 180 * len(earth_ground_mask)),
                int((self.longitude + 180) / 360 * len(earth_ground_mask[0])),
            ]
        )

    def cartesian_ecef(self) -> tuple[float, float, float]:
        """
        Gets x, y and z coordinates of the station in the Earth-centered reference frame.
        """

        return (
            (EARTH_RADIUS * cos(radians(self.latitude)) * cos(radians(self.longitude))),
            (EARTH_RADIUS * cos(radians(self.latitude)) * sin(radians(self.longitude))),
            EARTH_RADIUS * sin(radians(self.latitude)),
        )


@dataclass
class StationPositionBias:
    """
    Describes the constant bias in position for a ground station.
    """

    eastward: float = 0.0
    northward: float = 0.0
    vertical: float = 0.0


@dataclass
class StationParameters:
    """
    Describes the invertible parameters of a ground station.
    """

    eastward_speed: float = 0.0
    northward_speed: float = 0.0
    vertical_speed: float = 0.0
    range_bias: float = 0.0


@dataclass
class StationSimulation:
    """
    The non-invertible parameters needed to generate simulated measurements.
    """

    delta_timestamp: float = 10.0
    random_delay: float = 10.0
    sigma_noise: float = 1e-2


class Station(Parameters):
    """
    Describes a ground station invertible parameters and constants.
    """

    name: str
    station_position: StationPosition
    station_parameters: StationParameters
    station_position_bias: StationPositionBias = StationPositionBias()
    minimal_elevation_angle: float
    station_simulation: StationSimulation = StationSimulation()

    def __init__(
        self,
        name: str,
        station_position: StationPosition,
        station_parameters: StationParameters = StationParameters(),
        minimal_elevation_angle: float = 30.0,
    ) -> None:

        self.name = name
        self.station_position = station_position
        self.station_parameters = station_parameters
        self.minimal_elevation_angle = minimal_elevation_angle
        self.station_position_bias = StationPositionBias(eastward=0.0, northward=0.0, vertical=0.0)

    def get_terminal_parameters(self) -> dict[str, float]:
        """
        Dictionnary containing the numerical values of both invertible and non-invertible
        parameters.
        """

        return {
            parameter_name.replace("station", self.name): parameter_value
            for parameter_name, parameter_value in zip(
                [
                    r"\phi_{station\ latitude}",
                    r"\lambda_{station\ longitude}",
                    r"h_{station\ altitude}",
                    r"b^e_{station\ eastward\ bias}",
                    r"b^n_{station\ northward\ bias}",
                    r"b^v_{station\ vertical\ bias}",
                    r"v^e_{station\ eastward\ speed}",
                    r"v^n_{station\ northward\ speed}",
                    r"v^v_{station\ vertical\ speed}",
                    r"\Delta r_{station}",
                ],
                [
                    self.station_position.latitude,
                    self.station_position.longitude,
                    self.station_position.altitude,
                    self.station_position_bias.eastward,
                    self.station_position_bias.northward,
                    self.station_position_bias.vertical,
                    self.station_parameters.eastward_speed,
                    self.station_parameters.northward_speed,
                    self.station_parameters.vertical_speed,
                    self.station_parameters.range_bias,
                ],
            )
        }

    def get_parameter_expressions(self) -> dict[str, Expr]:
        """
        Dictionnary containing all parameter expressions for a station.
        """

        return {
            parameter_name.replace("station", self.name): Symbol(
                parameter_name.replace("station", self.name)
            )
            for parameter_name in [
                r"\phi_{station\ latitude}",
                r"\lambda_{station\ longitude}",
                r"h_{station\ altitude}",
                r"b^e_{station\ eastward\ bias}",
                r"b^n_{station\ northward\ bias}",
                r"b^v_{station\ vertical\ bias}",
                r"v^e_{station\ eastward\ speed}",
                r"v^n_{station\ northward\ speed}",
                r"v^v_{station\ vertical\ speed}",
                r"\Delta r_{station}",
            ]
        }

    def visibility(
        self, state_vector: ndarray[float], terminal_parameter_values: dict[str, float]
    ) -> bool:
        """
        Verifies numerically if a given satellite is visible from the station.
        """

        # Station ECEF position (spherical Earth assumption)
        r_station = array(
            object=[
                (terminal_parameter_values[r"R_{Earth\ radius}"] + self.station_position.altitude)
                * numpy.cos(self.station_position.latitude)
                * numpy.cos(self.station_position.longitude),
                (terminal_parameter_values[r"R_{Earth\ radius}"] + self.station_position.altitude)
                * numpy.cos(self.station_position.latitude)
                * numpy.sin(self.station_position.longitude),
                (terminal_parameter_values[r"R_{Earth\ radius}"] + self.station_position.altitude)
                * numpy.sin(self.station_position.latitude),
            ]
        )
        rho = state_vector[:3] - r_station
        elevation = arcsin(dot(rho / norm(rho), r_station / norm(r_station)))

        return degrees(elevation) >= self.minimal_elevation_angle

    def apply_to_station(self, expression: Expr) -> Expr:
        """
        Particularizes a station-dependent expression to a particular instance.
        """

        # Collect symbols to replace
        mapping: dict[Symbol, Symbol] = {}
        s: Symbol

        for s in expression.free_symbols:

            if "station" in s.name:

                new_name = s.name.replace("station", f"{self.name}", 1)

                mapping[s] = Symbol(new_name)

        return expression.xreplace(mapping)


def station_state_vector(parameter_expressions: dict[str, Expr]) -> MutableDenseMatrix:
    """
    Returns the symbolic 6D Cartesian state vector (position + velocity) of a station in the
    inertial frame.
    """

    latitude = Symbol(r"\phi_{station\ latitude}")
    longitude = Symbol(r"\lambda_{station\ longitude}")
    altitude = Symbol(r"h_{station\ altitude}")
    station_eastward_speed = Symbol(r"v^e_{station\ eastward\ speed}")
    station_northward_speed = Symbol(r"v^n_{station\ northward\ speed}")
    station_vertical_speed = Symbol(r"v^v_{station\ vertical\ speed}")

    # Nominal ECEF position assuming spherical Earth.
    r_ecef = MutableDenseMatrix(
        [
            (parameter_expressions[r"R_{Earth\ radius}"] + altitude)
            * cos(latitude)
            * cos(longitude),
            (parameter_expressions[r"R_{Earth\ radius}"] + altitude)
            * cos(latitude)
            * sin(longitude),
            (parameter_expressions[r"R_{Earth\ radius}"] + altitude) * sin(latitude),
        ]
    )
    enu_offset = MutableDenseMatrix(
        [
            Symbol(r"b^e_{station\ eastward\ bias}")
            + station_eastward_speed * parameter_expressions[r"t"],
            Symbol(r"b^n_{station\ northward\ bias}")
            + station_northward_speed * parameter_expressions[r"t"],
            Symbol(r"b^v_{station\ vertical\ bias}")
            + station_vertical_speed * parameter_expressions[r"t"],
        ]
    )
    enu_velocity = MutableDenseMatrix(
        [station_eastward_speed, station_northward_speed, station_vertical_speed]
    )
    enu_to_ecef = rotation_matrix(
        angle=longitude, unit_vector=MutableDenseMatrix([0, 0, 1])
    ) @ rotation_matrix(angle=latitude, unit_vector=MutableDenseMatrix([0, 1, 0]))
    r_ecef_total = r_ecef + enu_to_ecef @ enu_offset
    v_ecef_local = Matrix(enu_to_ecef @ enu_velocity)
    r_eci = Matrix(ecef_to_eci(parameter_expressions=parameter_expressions) @ r_ecef_total)

    # Inertial velocity: omega * r + rotated local velocity.
    return MutableDenseMatrix.vstack(
        Matrix(r_eci),
        Matrix(
            ecef_to_eci(parameter_expressions=parameter_expressions) @ v_ecef_local
            + MutableDenseMatrix(
                [0, 0, parameter_expressions[r"\omega_{Earth\ rotation\ angular\ speed}"]]
            ).cross(r_eci)
        ),
    )


def get_stations(
    stations_path: Path = TEST_OUTPUT_PATH, station_file_name: str = "stations"
) -> dict[str, Station]:
    """
    Reads a (.CSV) file to get all station informations.
    """

    stations = {}
    dataframe = read_csv(filepath_or_buffer=stations_path.joinpath(station_file_name + ".csv"))

    for (
        station_id,
        latitude,
        longitude,
        altitude,
        eastward_speed,
        northward_speed,
        vertical_speed,
        range_bias,
        minimal_elevation_angle,
    ) in zip(
        dataframe["station_id"].to_list(),
        dataframe["latitude"].to_list(),
        dataframe["longitude"].to_list(),
        dataframe["altitude"].to_list(),
        dataframe["eastward_speed"].to_list(),
        dataframe["northward_speed"].to_list(),
        dataframe["vertical_speed"].to_list(),
        dataframe["range_bias"].to_list(),
        dataframe["minimal_elevation_angle"].to_list(),
    ):

        stations[station_id] = Station(
            name=station_id,
            station_position=StationPosition(
                latitude=latitude, longitude=longitude, altitude=altitude
            ),
            station_parameters=StationParameters(
                eastward_speed=eastward_speed,
                northward_speed=northward_speed,
                vertical_speed=vertical_speed,
                range_bias=range_bias,
            ),
            minimal_elevation_angle=minimal_elevation_angle,
        )

    return stations
