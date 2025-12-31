"""
Describes a station object and its parameters.
"""

from dataclasses import dataclass

import numpy
from numpy import ndarray
from sympy import Expr, Matrix, MutableDenseMatrix, Symbol, cos, sin

from .parameters import Parameters
from .utils import rotation_matrix


@dataclass
class StationPosition:
    """
    Describes the reference position of a ground station.
    """

    latitude: float
    longitude: float
    altitude: float


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
        minimal_elevation_angle: float = 60.0,
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
            self.name + "_" + parameter_name: parameter_value
            for parameter_name, parameter_value in zip(
                [
                    "latitude",
                    "longitude",
                    "altitude",
                    "eastward_bias",
                    "northward_bias",
                    "vertical_bias",
                    "eastward_speed",
                    "northward_speed",
                    "vertical_speed",
                    "range_bias",
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
            self.name + "_" + parameter_name: Symbol(self.name + "_" + parameter_name)
            for parameter_name in [
                "latitude",
                "longitude",
                "altitude",
                "eastward_bias",
                "northward_bias",
                "vertical_bias",
                "eastward_speed",
                "northward_speed",
                "vertical_speed",
                "range_bias",
            ]
        }

    def station_state_vector(self, parameter_expressions: dict[str, Expr]) -> MutableDenseMatrix:
        """
        Returns the symbolic 6D Cartesian state vector (position + velocity)
        of the station in the inertial frame.
        """

        latitude = Symbol(f"{self.name}_latitude")
        longitude = Symbol(f"{self.name}_longitude")
        altitude = Symbol(f"{self.name}_altitude")

        # Nominal ECEF position assuming spherical Earth.
        r_ecef = MutableDenseMatrix(
            [
                (parameter_expressions["Earth_radius"] + altitude) * cos(latitude) * cos(longitude),
                (parameter_expressions["Earth_radius"] + altitude) * cos(latitude) * sin(longitude),
                (parameter_expressions["Earth_radius"] + altitude) * sin(latitude),
            ]
        )
        enu_offset = MutableDenseMatrix(
            [
                parameter_expressions[f"{self.name}_eastward_bias"]
                + parameter_expressions[f"{self.name}_eastward_speed"] * parameter_expressions["t"],
                parameter_expressions[f"{self.name}_northward_bias"]
                + parameter_expressions[f"{self.name}_northward_speed"]
                * parameter_expressions["t"],
                parameter_expressions[f"{self.name}_vertical_bias"]
                + parameter_expressions[f"{self.name}_vertical_speed"] * parameter_expressions["t"],
            ]
        )
        enu_velocity = MutableDenseMatrix(
            [
                parameter_expressions[f"{self.name}_eastward_speed"],
                parameter_expressions[f"{self.name}_northward_speed"],
                parameter_expressions[f"{self.name}_vertical_speed"],
            ]
        )
        enu_to_ecef = rotation_matrix(
            angle=longitude, unit_vector=MutableDenseMatrix([0, 0, 1])
        ) @ rotation_matrix(angle=latitude, unit_vector=MutableDenseMatrix([0, 1, 0]))
        r_ecef_total = r_ecef + enu_to_ecef @ enu_offset
        v_ecef_local = Matrix(enu_to_ecef @ enu_velocity)
        r_eci = Matrix(
            rotation_matrix(
                angle=parameter_expressions["arc_start_Earth_rotation_angle"]
                + parameter_expressions["Earth_rotation_angular_speed"] * parameter_expressions["t"]
            )
            @ r_ecef_total
        )

        # Inertial velocity: omega * r + rotated local velocity.

        return MutableDenseMatrix.vstack(
            Matrix(r_eci),
            Matrix(
                rotation_matrix(
                    angle=parameter_expressions["arc_start_Earth_rotation_angle"]
                    + parameter_expressions["Earth_rotation_angular_speed"]
                    * parameter_expressions["t"]
                )
                @ v_ecef_local
                + MutableDenseMatrix(
                    [0, 0, parameter_expressions["Earth_rotation_angular_speed"]]
                ).cross(r_eci)
            ),
        )

    def visibility(
        self, state_vector: ndarray[float], terminal_parameter_values: dict[str, float]
    ) -> bool:
        """
        Verifies numerically if a given satellite is visible from the station.
        """

        # Station ECEF position (spherical Earth assumption)
        r_station = numpy.array(
            object=[
                (terminal_parameter_values["Earth_radius"] + self.station_position.altitude)
                * numpy.cos(self.station_position.latitude)
                * numpy.cos(self.station_position.longitude),
                (terminal_parameter_values["Earth_radius"] + self.station_position.altitude)
                * numpy.cos(self.station_position.latitude)
                * numpy.sin(self.station_position.longitude),
                (terminal_parameter_values["Earth_radius"] + self.station_position.altitude)
                * numpy.sin(self.station_position.latitude),
            ]
        )
        rho = state_vector[:3] - r_station
        elevation = numpy.arcsin(
            numpy.dot(rho / numpy.linalg.norm(rho), r_station / numpy.linalg.norm(r_station))
        )

        return numpy.degrees(elevation) >= self.minimal_elevation_angle
