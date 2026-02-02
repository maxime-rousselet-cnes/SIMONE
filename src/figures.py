"""
Defines all plot functions for teaching purposes.
"""

# TODO: Range shown the two frames around the observation timestep.
# TODO: Point.
from pathlib import Path
from typing import Optional

from numpy import cos, cross, linspace, meshgrid, ndarray, pi, reshape, roll, sin, zeros
from numpy.linalg import norm
from plotly.graph_objects import Figure, Frame, Scatter3d, Surface

from .base_constants import (
    DEFAULT_MU,
    EARTH_COLOR_SCALE,
    EARTH_RADIUS,
    EARTH_SURFACE_COLOR,
    GRAY_EARTH_IMAGE,
    TEST_OUTPUT_PATH,
)
from .dynamics import ecef_to_eci, eci_to_ecef
from .observation import ArcOutput, load_arc_output
from .station import Station, get_stations
from .utils import evaluate_terminal_parameters

TRAJECTORY_COLOR = "purple"
TRAJECTORY_LINE_WIDTH = 20
CLOSEST_ORBIT_COLOR = "red"
CLOSEST_ORBIT_LINE_WIDTH = 10
RANGE_MEASUREMENT_COLOR = "red"
RANGE_MEASUREMENT_WIDTH = 10
STATION_COLOR = "red"
STATION_DOT_SIZE = 20
DEFAULT_FRAME_DURATION = 40  # (ms).
DEFAULT_SNAKE_LENGTH = 20
DEFAULT_ELLIPSE_NUMBER_OF_POINTS = 200
SPEED_UP_FACTOR = 20


def elliptical_orbit_points(
    y: ndarray[float], n_points: int = DEFAULT_ELLIPSE_NUMBER_OF_POINTS, mu: float = DEFAULT_MU
) -> ndarray[float]:
    """
    Computes a satellite's elliptical reference orbit in the inertial frame for a single timestep.
    """

    r = y[:3]
    v = y[3:]
    h = cross(a=r, b=v)
    eccentricity_vector = cross(a=v, b=h) / mu - r / norm(x=r)
    eccentricity = norm(x=eccentricity_vector)
    semi_major_axis = 1 / (2 / norm(x=r) - norm(x=v) ** 2 / mu)
    ellipse = zeros(shape=(n_points, 3))

    # Unit vectors in orbital plane
    p_hat = eccentricity_vector / eccentricity
    q_hat = cross(a=h / norm(x=h), b=p_hat)

    for i, nu_i in enumerate(linspace(0, 2 * pi, n_points)):

        r_i = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * cos(nu_i))
        ellipse[i] = r_i * (cos(nu_i) * p_hat + sin(nu_i) * q_hat)

    return ellipse


def rotate_columns(arr: ndarray[float], x: float) -> ndarray:
    """
    Performs Earth rotation for the background Earth representation.
    """

    return roll(arr, shift=int(round(x * arr.shape[1])) % arr.shape[1], axis=1)


def display_earth_figure(angle: float) -> Figure:
    """
    Generates a figure with a 3D spherical representation of the Earth.
    """
    nlat, nlon = GRAY_EARTH_IMAGE.shape
    lat = linspace(-pi / 2, pi / 2, nlat)
    lon = linspace(-pi, pi, nlon)
    lon, lat = meshgrid(lon, lat)
    x = EARTH_RADIUS * cos(lat) * cos(lon)
    y = EARTH_RADIUS * cos(lat) * sin(lon)
    z = EARTH_RADIUS * sin(lat)
    figure = Figure(
        Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=rotate_columns(arr=EARTH_SURFACE_COLOR, x=angle / (2 * pi)),
            colorscale=EARTH_COLOR_SCALE,
            showscale=False,
        )
    )
    figure.update_layout(
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
            "bgcolor": "black",
        },
        paper_bgcolor="black",
    )

    return figure


class AnimationParameters:
    """
    Needed class to display animations.
    """

    reference_frame: str = "ECI"  # Assumes "ECEF" otherwise.
    display_range_measurements: bool = True
    display_stations: bool = True
    display_ellipse: bool = True
    time_indices_for_range_measurements: dict[int, list[str]] = {}
    satellite_positions: dict[str, ndarray[float]] = {}
    station_positions: dict[str, list[ndarray[float]]] = {}

    def display_arc_animation(self, arc_output: ArcOutput, stations: dict[str, Station]) -> None:
        """
        Allows to vizualise the output of a forward simulation, i.e. the simulated trajectory of
        a spacecraft for a chosen arc, either in Earth-fixed reference frame or inertial reference
        frame, and eventually the range measurements and the closest ellipse at any time.
        """

        parameters = arc_output.simulation_parameters
        arc_output.t = arc_output.t[::SPEED_UP_FACTOR]
        arc_output.y = arc_output.y[::SPEED_UP_FACTOR]
        ellipses = (
            None
            if not self.display_ellipse
            else [
                elliptical_orbit_points(
                    y=y_k,
                    n_points=DEFAULT_ELLIPSE_NUMBER_OF_POINTS,
                    mu=parameters.terminal_parameter_values[r"\mu_{gravitational\ parameter}"],
                )
                for y_k in arc_output.y
            ]
        )

        self.time_indices_for_range_measurements: dict[int, list[str]] = {}

        if self.display_range_measurements:

            for station_id, timestamps in arc_output.observation_timestamps.items():

                time_index = 0

                for timestamp in timestamps:

                    while arc_output.t[time_index] < timestamp:

                        time_index += 1

                    if not time_index in self.time_indices_for_range_measurements:

                        self.time_indices_for_range_measurements[time_index] = []

                    self.time_indices_for_range_measurements[time_index] += [station_id]

        self.satellite_positions = (
            arc_output.y[:, :3]
            if self.reference_frame == "ECI"
            else [
                evaluate_terminal_parameters(
                    expression=eci_to_ecef(
                        parameter_expressions=parameters.parameter_expressions,
                    ),
                    parameter_expressions=parameters.parameter_expressions,
                    terminal_parameter_values=parameters.terminal_parameter_values,
                ).xreplace(rule={parameters.parameter_expressions[r"t"]: t_k})
                @ reshape(y_k, shape=(3, 1))
                for t_k, y_k in zip(arc_output.t, arc_output.y[:, :3])
            ]
        )

        self.station_positions: dict[str, list[ndarray[float]]] = {}

        if self.display_range_measurements or self.display_stations:

            for station_id in arc_output.observation_timestamps:

                self.station_positions[station_id] = (
                    [stations[station_id].station_position.cartesian_ecef()] * len(arc_output.t)
                    if self.reference_frame == "ECEF"
                    else [
                        evaluate_terminal_parameters(
                            expression=ecef_to_eci(
                                parameter_expressions=parameters.parameter_expressions,
                            ),
                            parameter_expressions=parameters.parameter_expressions,
                            terminal_parameter_values=parameters.terminal_parameter_values,
                        ).xreplace(rule={parameters.parameter_expressions[r"t"]: t_k})
                        @ reshape(
                            stations[station_id].station_position.cartesian_ecef(), shape=(3, 1)
                        )
                        for t_k in arc_output.t
                    ]
                )

        # Initial frame.
        base_figure = self.generate_figure(
            ellipses=ellipses,
            angle=parameters.terminal_parameter_values[
                r"\theta_{arc\ start\ Earth\ rotation\ angle}"
            ],
            time_index=0,
        )

        # Animation figure.
        animation = Figure(
            data=base_figure.data,
            layout=base_figure.layout,
        )
        animation.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": DEFAULT_FRAME_DURATION,
                                        "redraw": True,
                                    },
                                    "fromcurrent": True,
                                },
                            ],
                        }
                    ],
                }
            ]
        )

        # Frames.
        animation.frames = [
            Frame(
                data=fig.data,
                layout=fig.layout,
                name=str(time_index),
            )
            for time_index, fig in enumerate(
                self.generate_figure(
                    ellipses=ellipses,
                    angle=parameters.terminal_parameter_values[
                        r"\theta_{arc\ start\ Earth\ rotation\ angle}"
                    ]
                    + parameters.terminal_parameter_values[
                        r"\omega_{Earth\ rotation\ angular\ speed}"
                    ]
                    * arc_output.t[time_index],
                    time_index=time_index,
                )
                for time_index in range(len(arc_output.t))
            )
        ]

        animation.show()

    def generate_figure(
        self,
        ellipses: Optional[ndarray[float]],
        angle: float,
        time_index: int = 0,
    ) -> Figure:
        """
        Generate a static figure to be a frame in an animation.
        """

        figure = display_earth_figure(angle=angle)

        if ellipses:

            figure.add_trace(
                trace=Scatter3d(
                    x=ellipses[time_index][:, 0],
                    y=ellipses[time_index][:, 1],
                    z=ellipses[time_index][:, 2],
                    mode="lines",
                    line={"color": CLOSEST_ORBIT_COLOR, "width": CLOSEST_ORBIT_LINE_WIDTH},
                )
            )

        figure.add_trace(
            trace=Scatter3d(
                x=self.satellite_positions[
                    max(0, time_index - DEFAULT_SNAKE_LENGTH) : time_index + 1, 0
                ],
                y=self.satellite_positions[
                    max(0, time_index - DEFAULT_SNAKE_LENGTH) : time_index + 1, 1
                ],
                z=self.satellite_positions[
                    max(0, time_index - DEFAULT_SNAKE_LENGTH) : time_index + 1, 2
                ],
                mode="lines",
                line={"color": TRAJECTORY_COLOR, "width": TRAJECTORY_LINE_WIDTH},
            )
        )

        return figure


def display_test_animation(
    output_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = "stations",
    arc_id: str = "test_arc_id",
) -> None:
    """
    Displays the test arc with all options.
    """

    AnimationParameters().display_arc_animation(
        arc_output=load_arc_output(output_path=output_path, arc_id=arc_id),
        stations=get_stations(stations_path=output_path, station_file_name=station_file_name),
    )


def plot_station(figure: Figure, station: Station) -> None:
    """
    Adds a 3D static representation of a point of interest on the surface of the Earth.
    """

    x, y, z = station.station_position.cartesian_ecef()
    figure.add_trace(
        Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode="markers",
            marker={"color": STATION_COLOR, "size": STATION_DOT_SIZE},
            name=station.name,
        )
    )
