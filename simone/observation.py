"""
Describes observation types for a general observation framework.
"""

from pathlib import Path
from random import random
from typing import Optional

from base_models import evaluate_terminal_parameters, load_base_model, save_base_model
from numpy import array, ndarray
from numpy.random import normal
from pandas import DataFrame, read_csv
from sympy import Expr, Symbol

from .base_constants import (
    DEFAULT_MEASUREMENT_DIRECTORY_NAME,
    DEFAULT_RESIDUALS_FILE_NAME,
    DEFAULT_SIMULATED_MEASUREMENTS_FILE_NAME,
    DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
    TEST_ARC_ID,
    TEST_MEASUREMENTS_PATH,
    TEST_NO_ITERATIONS_PATH,
)
from .simulation_parameters import SimulationParameters, load_simulation_parameters
from .station import Station, station_state_vector
from .utils import STATE_VECTOR_LINE, STATE_VECTOR_MATRIX, distance


def save_measurements(
    observation_timestamps: dict[str, list[float]],
    measurements: dict[str, list[float]],
    stations: Optional[dict[str, Station]] = None,
    path: Path = TEST_MEASUREMENTS_PATH,
    file_name: str = DEFAULT_RESIDUALS_FILE_NAME,
) -> None:
    """
    Writes a line per measurement in a (.CSV) file.
    In case stations are provided, adds a noise to the measurements according to the station
    parameters.
    """

    station_ids, timestamps, values = [], [], []

    for station_id, measurement_list in measurements.items():

        station_ids += len(observation_timestamps[station_id]) * [station_id]
        timestamps += observation_timestamps[station_id]
        values += list(
            array(object=measurement_list, dtype=float)
            + (
                0
                if stations is None
                else normal(
                    loc=0,
                    # Adds a noise for simulations initially generated simulations.
                    scale=stations[station_id].station_simulation.sigma_noise,
                    size=len(observation_timestamps[station_id]),
                )
            )
        )

    path.mkdir(exist_ok=True, parents=True)
    DataFrame(data={"station_id": station_ids, "timestamp": timestamps, "value": values}).to_csv(
        path_or_buf=path.joinpath(file_name + ".csv"),
        index=False,
    )


class ArcOutput:
    """
    To be used for post-process or plot purposes.
    """

    simulation_parameters: SimulationParameters
    t: ndarray
    y: ndarray
    observation_timestamps: dict[str, list[float]] = {}
    simulated_measurements: dict[str, list[float]] = {}
    residuals: dict[str, list[float]] = {}

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        t: ndarray,
        y: ndarray,
    ) -> None:

        self.simulation_parameters = simulation_parameters
        self.t = t
        self.y = y

    def update_observations(
        self,
        observation_timestamps: dict[str, list[float]],
        theoretical_measurements: dict[str, list[float]],
        real_measurements: dict[str, list[float]],
    ) -> None:
        """
        Difference.
        """

        self.simulation_parameters.arc_parameters.is_initial = False
        self.observation_timestamps = observation_timestamps
        self.simulated_measurements = theoretical_measurements
        self.residuals = compute_residuals(
            theoretical_measurements=theoretical_measurements,
            real_measurements=real_measurements,
        )

    def save(
        self,
        output_path: Path = TEST_NO_ITERATIONS_PATH,
        simulated_measurements_file_name: str = DEFAULT_SIMULATED_MEASUREMENTS_FILE_NAME,
        residuals_file_name: str = DEFAULT_RESIDUALS_FILE_NAME,
        simulation_parameters_file_name: str = DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
    ) -> None:
        """
        In (.JSON) file.
        """

        path = output_path.joinpath(self.simulation_parameters.arc_parameters.arc_id).joinpath(
            "arc_output"
        )
        path.mkdir(exist_ok=True, parents=True)
        self.simulation_parameters.save(path=path)
        save_base_model(
            obj=self.t,
            name="t",
            path=path,
        )
        save_base_model(
            obj=self.y,
            name="y",
            path=path,
        )
        save_measurements(
            observation_timestamps=self.observation_timestamps,
            measurements=self.simulated_measurements,
            path=path,
            file_name=simulated_measurements_file_name,
        )
        save_measurements(
            observation_timestamps=self.observation_timestamps,
            measurements=self.residuals,
            path=path,
            file_name=residuals_file_name,
        )
        self.simulation_parameters.save(path=path, name=simulation_parameters_file_name)


def apply_lagrange_kernel(
    expression: Expr,
    arc_output: ArcOutput,
    y: ndarray,
    to_interpolate: list[Expr],
    time_index: int,
) -> Expr:
    """
    Evaluates an expression containing a Lagrange kernel.
    """

    simulation_parameters = arc_output.simulation_parameters

    return evaluate_terminal_parameters(
        expression=expression.xreplace(
            rule=dict(zip(to_interpolate, simulation_parameters.lagrange_kernels))
        ),
        parameter_expressions=simulation_parameters.parameter_expressions,
        terminal_parameter_values=simulation_parameters.terminal_parameter_values,
    ).xreplace(
        rule={
            Symbol(rf"t_{i}"): t_i
            for i, t_i in enumerate(
                arc_output.t[
                    time_index
                    - simulation_parameters.arc_parameters.lagrange_interpolation_order : time_index
                    + simulation_parameters.arc_parameters.lagrange_interpolation_order
                ]
            )
        }
        | {
            Symbol(rf"\theta^{j}_{i}"): y_i
            for j in range(6)
            for i, y_i in enumerate(
                y[
                    time_index
                    - simulation_parameters.arc_parameters.lagrange_interpolation_order : time_index
                    + simulation_parameters.arc_parameters.lagrange_interpolation_order,
                    j,
                ]
            )
        }
    )


def interpolate_function_of_position(
    expression: Expr,
    arc_output: ArcOutput,
    observation_timestamps: list[float],
    partials: Optional[list[Expr]] = None,
    partial_numerical_values: Optional[ndarray] = None,
) -> list[float | ndarray]:
    """
    Applies numerically any function that depends on the interpolation of the state vector.
    """

    time_index = 0
    results = []

    for observation_timestamp in observation_timestamps:

        # For linear complexity in quadrature length.
        while arc_output.t[time_index] < observation_timestamp:

            time_index += 1

        time_dependent_expression: Expr = apply_lagrange_kernel(
            expression=expression,
            arc_output=arc_output,
            y=arc_output.y,
            to_interpolate=STATE_VECTOR_LINE,
            time_index=time_index,
        )

        if partials:

            time_dependent_expression = apply_lagrange_kernel(
                expression=time_dependent_expression,
                arc_output=arc_output,
                y=partial_numerical_values,
                to_interpolate=partials,
                time_index=time_index,
            )

        results += [
            time_dependent_expression.xreplace(
                rule={
                    arc_output.simulation_parameters.parameter_expressions[
                        r"t"
                    ]: observation_timestamp
                }
            )
        ]

    return results


def generate_measurements(
    arc_output: ArcOutput,
    stations: dict[str, Station],
    observation_timestamps: dict[str, list[float]],
) -> tuple[dict[str, list[float]], Expr]:
    """
    Produces theoretical measurement values from integrated state vectors.
    """

    station_theoretical_measurements: dict[str, list[float]] = {}
    general_observation_expression = distance(
        vector_1=station_state_vector(
            parameter_expressions=arc_output.simulation_parameters.parameter_expressions
        ),
        vector_2=STATE_VECTOR_MATRIX,
    ) + Symbol(r"\Delta r_{station}")

    for station_id, timestamps in observation_timestamps.items():

        if not timestamps:
            continue

        # Symbolic expression to differentiate later to produce dQ/dgamma and nabla Q.
        station_theoretical_measurements[station_id] = interpolate_function_of_position(
            expression=stations[station_id].apply_to_station(
                expression=general_observation_expression
            ),
            arc_output=arc_output,
            observation_timestamps=observation_timestamps[station_id],
        )

    return station_theoretical_measurements, general_observation_expression


def get_visibilities(
    t: ndarray,
    y: ndarray,
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

        for i_time, (current_quadrature_time, state_vector) in enumerate(zip(t, y)):

            if (
                i_time > simulation_parameters.arc_parameters.lagrange_interpolation_order
                and station.visibility(
                    state_vector=state_vector,
                    terminal_parameter_values=simulation_parameters.terminal_parameter_values,
                )
            ):

                if not is_visible:

                    is_visible = True
                    window_start = current_quadrature_time

            elif is_visible:

                is_visible = False
                first_timestamp = window_start + station.station_simulation.random_delay * random()
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
    t: ndarray,
    y: ndarray,
    stations: dict[str, Station],
    simulation_parameters: SimulationParameters,
    path: Path = TEST_NO_ITERATIONS_PATH,
) -> dict[str, list[float]]:
    """
    Simulate a serie of measurements whenever the satellite is visible from a station.
    """

    observation_timestamps = get_visibilities(
        t=t, y=y, stations=stations, simulation_parameters=simulation_parameters
    )
    arc_output = ArcOutput(simulation_parameters=simulation_parameters, t=t, y=y)
    measurements, _ = generate_measurements(
        arc_output=arc_output,
        stations=stations,
        observation_timestamps=observation_timestamps,
    )
    save_measurements(
        observation_timestamps=observation_timestamps,
        measurements=measurements,
        stations=stations if simulation_parameters.arc_parameters.is_initial else None,
        path=path.joinpath(DEFAULT_MEASUREMENT_DIRECTORY_NAME),
        file_name=simulation_parameters.arc_parameters.arc_id,
    )

    return measurements


def get_measurements(
    path: Path, name: str = DEFAULT_RESIDUALS_FILE_NAME
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Gets measurement timestamps and values per station from (.CSV) file.
    """

    observation_timestamps: dict[str, list[float]] = {}
    measurement_values: dict[str, list[float]] = {}
    dataframe = read_csv(filepath_or_buffer=path.joinpath(name + ".csv"))

    for station_id, timestamp, value in zip(
        dataframe["station_id"].to_list(),
        dataframe["timestamp"].to_list(),
        dataframe["value"].to_list(),
    ):

        if station_id not in observation_timestamps:

            observation_timestamps[station_id] = []
            measurement_values[station_id] = []

        observation_timestamps[station_id] += [float(timestamp)]
        measurement_values[station_id] += [float(value)]

    return observation_timestamps, measurement_values


def compute_residuals(
    theoretical_measurements: dict[str, list[float]], real_measurements: dict[str, list[float]]
) -> dict[str, list[float]]:
    """
    Difference.
    """

    return {
        station_id: array(object=real_measurement)
        - array(object=theoretical_measurements[station_id])
        for station_id, real_measurement in real_measurements.items()
    }


def load_arc_output(
    iteration_path: Path = TEST_NO_ITERATIONS_PATH,
    arc_id: str = TEST_ARC_ID,
    simulated_measurements_file_name: str = DEFAULT_SIMULATED_MEASUREMENTS_FILE_NAME,
    residuals_file_name: str = DEFAULT_RESIDUALS_FILE_NAME,
    simulation_parameters_file_name: str = DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
) -> ArcOutput:
    """
    Gets the numerical outputs of an arc for plot purposes.
    """

    path = iteration_path.joinpath(arc_id).joinpath("arc_output")
    arc_output = ArcOutput(
        simulation_parameters=load_simulation_parameters(
            path=path, name=simulation_parameters_file_name
        ),
        t=array(object=load_base_model(name="t", path=path)),
        y=array(object=load_base_model(name="y", path=path)),
    )
    observation_timestamps, residuals = get_measurements(path=path, name=residuals_file_name)
    _, simulated_measurements = get_measurements(path=path, name=simulated_measurements_file_name)
    arc_output.observation_timestamps = observation_timestamps
    arc_output.simulated_measurements = simulated_measurements
    arc_output.residuals = residuals

    return arc_output
