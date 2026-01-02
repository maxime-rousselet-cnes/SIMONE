"""
Describes observation types for a general observation framework.
"""

import random
from pathlib import Path
from shutil import rmtree

from numpy import array, ndarray
from numpy.random import normal
from pandas import DataFrame, read_csv
from sympy import Expr, Symbol

from .forward_simulation import propagate_ephemeris
from .interpolation import apply_lagrange_kernel
from .parameters import SimulationParameters
from .station import Station, station_state_vector
from .test_constants import TEST_OUTPUT_PATH, TEST_SIMULATION_PARAMETERS, TEST_STATIONS
from .utils import distance


def generate_measurements(
    t: ndarray[float],
    y: ndarray[float],
    stations: dict[str, Station],
    observation_timestamps: dict[str, list[float]],
    simulation_parameters: SimulationParameters,
) -> tuple[dict[str, list[float]], dict[str, Expr]]:
    """
    Produces theoretical measurement values from integrated state vectors.
    """

    station_theoretical_measurements: dict[str, list[float]] = {}
    station_observations: dict[str, Expr] = {}
    general_station_state_vector = station_state_vector(
        parameter_expressions=simulation_parameters.parameter_expressions
    )

    for station_id, station in stations.items():

        # Symbolic expression to differentiate later to produce dQ/dgamma and nabla Q.
        station_observations[station_id] = distance(
            vector_1=station.apply_to_station(expression=general_station_state_vector),
            vector_2=simulation_parameters.lagrange_kernels,
        )

        time_index = 0
        station_theoretical_measurements[station_id] = []

        for observation_timestamp in observation_timestamps[station_id]:

            # For linear complexity in quadrature length.
            while t[time_index] < observation_timestamp:

                time_index += 1

            station_theoretical_measurements[station_id] += [
                apply_lagrange_kernel(
                    expression=station_observations[station_id],
                    simulation_parameters=simulation_parameters,
                    t=t,
                    y=y,
                    time_index=time_index,
                ).xreplace(rule={Symbol("t"): observation_timestamp})
            ]

    return station_theoretical_measurements, station_observations


def get_visibilities(
    t: ndarray[float],
    y: ndarray[float],
    stations: dict[str, Station],
    simulation_parameters: SimulationParameters,
) -> dict[str, list[float]]:
    """
    To simulate measurements.
    """

    observation_timestamps = {}

    for station_id, station in stations.items():

        observation_timestamps[station_id] = []
        window_start = 0.0
        is_visible = False

        for current_quadrature_time, state_vector in zip(t, y):

            if station.visibility(
                state_vector=state_vector,
                terminal_parameter_values=simulation_parameters.terminal_parameter_values,
            ):

                if not is_visible:

                    is_visible = True
                    window_start = current_quadrature_time

            elif is_visible:

                is_visible = False
                first_timestamp = (
                    window_start + station.station_simulation.random_delay * random.random()
                )
                observation_timestamps[station_id] += [
                    first_timestamp + k * station.station_simulation.delta_timestamp
                    for k in range(
                        int(
                            (current_quadrature_time - first_timestamp)
                            // station.station_simulation.delta_timestamp
                        )
                    )
                ]

    return observation_timestamps


def simulate_measurements(
    t: ndarray[float],
    y: ndarray[float],
    stations: dict[str, Station],
    simulation_parameters: SimulationParameters,
    output_path: Path = TEST_OUTPUT_PATH,
) -> None:
    """
    Simulate a serie of measurements whenever the satellite is visible from a station.
    """

    observation_timestamps = get_visibilities(
        t=t, y=y, stations=stations, simulation_parameters=simulation_parameters
    )
    station_theoretical_measurements, _ = generate_measurements(
        t=t,
        y=y,
        stations=stations,
        observation_timestamps=observation_timestamps,
        simulation_parameters=simulation_parameters,
    )
    save_measurements(
        stations=stations,
        observation_timestamps=observation_timestamps,
        station_theoretical_measurements=station_theoretical_measurements,
        simulation_parameters=simulation_parameters,
        output_path=output_path,
    )


def save_measurements(
    stations: dict[str, Station],
    observation_timestamps: dict[str, list[float]],
    station_theoretical_measurements: dict[str, list[float]],
    simulation_parameters: SimulationParameters,
    output_path: Path,
) -> None:
    """
    Writes a line per measurement in a (.CSV) file.
    """

    dataframe = DataFrame(data={"station_id": [], "timestamp": [], "value": []})

    for station_id, station in stations.items():

        dataframe.add(
            other={
                "station_id": len(observation_timestamps[station_id]) * [station_id],
                "timestamp": observation_timestamps[station_id],
                "value": list(
                    array(object=station_theoretical_measurements[station_id], dtype=float)
                    + normal(
                        loc=0,
                        scale=station.station_simulation.sigma_noise,
                        size=len(observation_timestamps[station_id]),
                    )
                ),
            }
        )

    save_path = output_path.joinpath(simulation_parameters.arc_parameters.arc_id)
    save_path.mkdir(exist_ok=True, parents=True)
    dataframe.to_csv(
        path_or_buf=save_path.joinpath("measurements.csv"),
        index=False,
    )


def get_measurements(
    measurements_path: Path = TEST_OUTPUT_PATH,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Gets measurement timestamps and values per station from (.CSV) file.
    """

    dataframe = read_csv(
        filepath_or_buffer=output_path.joinpath(
            simulation_parameters.arc_parameters.arc_id
        ).joinpath("measurements.csv")
    )
    # TODO.

    return observation_timestamps, measurement_values


def test_observations(
    simulation_parameters: SimulationParameters = TEST_SIMULATION_PARAMETERS,
) -> None:
    """
    Verifies if the measurements are correctly created in a forward simulation.
    """

    t, y, _ = propagate_ephemeris(
        simulation_parameters=simulation_parameters,
    )

    if TEST_OUTPUT_PATH.exists():

        rmtree(TEST_OUTPUT_PATH)

    simulate_measurements(
        t=t, y=y, stations=TEST_STATIONS, simulation_parameters=simulation_parameters
    )

    assert (
        TEST_OUTPUT_PATH.joinpath(simulation_parameters.arc_parameters.arc_id)
        .joinpath("measurements.csv")
        .exists()
    )
