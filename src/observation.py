"""
Describes observation types for a general observation framework.
"""

from pathlib import Path
from random import random
from typing import Optional

from numpy import array, ndarray
from numpy.random import normal
from pandas import DataFrame, read_csv
from sympy import Expr, Symbol

from .base_constants import TEST_OUTPUT_PATH
from .simulation_parameters import SimulationParameters, load_simulation_parameters
from .station import Station, station_state_vector
from .utils import (
    STATE_VECTOR_LINE,
    STATE_VECTOR_MATRIX,
    distance,
    evaluate_terminal_parameters,
    load_base_model,
    save_base_model,
)


def apply_lagrange_kernel(
    expression: Expr,
    arc_output: ArcOutput,
    y: ndarray[float],
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
    partial_numerical_values: Optional[ndarray[float]] = None,
) -> list[float | ndarray[float]]:
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


def get_station_range(parameter_expressions: dict[str, Expr], station_id: str) -> Expr:
    """
    Gets the wanted symbol for station range bias.
    """

    for parameter in parameter_expressions:

        if station_id in parameter and "Delta" in parameter:

            return parameter_expressions[parameter]

    return Expr(0)


def generate_measurements(
    arc_output: ArcOutput,
    stations: dict[str, Station],
    observation_timestamps: dict[str, list[float]],
) -> tuple[dict[str, list[float]], dict[str, Expr]]:
    """
    Produces theoretical measurement values from integrated state vectors.
    """

    station_theoretical_measurements: dict[str, list[float]] = {}
    station_observations: dict[str, Expr] = {}
    general_station_state_vector = station_state_vector(
        parameter_expressions=arc_output.simulation_parameters.parameter_expressions
    )

    for station_id, timestamps in observation_timestamps.items():

        if not timestamps:
            continue

        # Symbolic expression to differentiate later to produce dQ/dgamma and nabla Q.
        station_observations[station_id] = distance(
            vector_1=stations[station_id].apply_to_station(expression=general_station_state_vector),
            vector_2=STATE_VECTOR_MATRIX,
        ) + get_station_range(
            parameter_expressions=arc_output.simulation_parameters.parameter_expressions,
            station_id=station_id,
        )
        station_theoretical_measurements[station_id] = interpolate_function_of_position(
            expression=station_observations[station_id],
            arc_output=arc_output,
            observation_timestamps=observation_timestamps[station_id],
        )

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
    arc_output = ArcOutput(simulation_parameters=simulation_parameters, t=t, y=y)
    station_theoretical_measurements, _ = generate_measurements(
        arc_output=arc_output,
        stations=stations,
        observation_timestamps=observation_timestamps,
    )
    save_measurements(
        argument=stations,
        observation_timestamps=observation_timestamps,
        station_theoretical_measurements=station_theoretical_measurements,
        simulation_parameters=simulation_parameters,
        output_path=output_path,
    )


def save_measurements(
    observation_timestamps: dict[str, list[float]],
    station_theoretical_measurements: dict[str, list[float]],
    simulation_parameters: SimulationParameters,
    argument: dict[str, Station] | str = "measurements",
    output_path: Path = TEST_OUTPUT_PATH,
) -> None:
    """
    Writes a line per measurement in a (.CSV) file.
    """

    if isinstance(argument, str):

        stations: dict[str, Station] = {}
        simulation_parameters.arc_parameters.is_initial = False

    else:

        stations = argument
        argument = "measurements"

    station_ids, timestamps, values = [], [], []

    for station_id, measurements in station_theoretical_measurements.items():

        station_ids += len(observation_timestamps[station_id]) * [station_id]
        timestamps += observation_timestamps[station_id]
        values += list(
            array(object=measurements, dtype=float)
            + (
                0
                if not simulation_parameters.arc_parameters.is_initial
                else normal(
                    loc=0,
                    # Adds a noise for simulations initially generated simulations.
                    scale=stations[station_id].station_simulation.sigma_noise,
                    size=len(observation_timestamps[station_id]),
                )
            )
        )

    save_path = output_path.joinpath(simulation_parameters.arc_parameters.arc_id)
    save_path.mkdir(exist_ok=True, parents=True)
    DataFrame(data={"station_id": station_ids, "timestamp": timestamps, "value": values}).to_csv(
        path_or_buf=save_path.joinpath(argument + ".csv"),
        index=False,
    )


def get_measurements(
    simulation_parameters: SimulationParameters,
    measurements_path: Path = TEST_OUTPUT_PATH,
    name: str = "measurements",
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Gets measurement timestamps and values per station from (.CSV) file.
    """

    observation_timestamps: dict[str, list[float]] = {}
    measurement_values: dict[str, list[float]] = {}
    dataframe = read_csv(
        filepath_or_buffer=measurements_path.joinpath(
            simulation_parameters.arc_parameters.arc_id
        ).joinpath(name + ".csv")
    )

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


class ArcOutput:
    """
    To be used for post-process or plot purposes.
    """

    simulation_parameters: SimulationParameters
    t: ndarray[float]
    y: ndarray[float]
    observation_timestamps: dict[str, list[float]] = {}
    simulated_measurements: dict[str, list[float]] = {}
    residuals: dict[str, list[float]] = {}

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        t: ndarray[float],
        y: ndarray[float],
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

    def save(self, output_path: Path = TEST_OUTPUT_PATH, iteration: int = 0) -> None:
        """
        In (.JSON) file.
        """

        path = (
            output_path.joinpath(self.simulation_parameters.arc_parameters.arc_id)
            .joinpath(str(iteration))
            .joinpath("arc_output")
        )
        path.mkdir(exist_ok=True, parents=True)
        self.simulation_parameters.save(output_path=path)
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
            argument=str(iteration) + "/arc_output/simulated_measurements",
            observation_timestamps=self.observation_timestamps,
            station_theoretical_measurements=self.simulated_measurements,
            simulation_parameters=self.simulation_parameters,
        )
        save_measurements(
            argument=str(iteration) + "/arc_output/residuals",
            observation_timestamps=self.observation_timestamps,
            station_theoretical_measurements=self.residuals,
            simulation_parameters=self.simulation_parameters,
        )


def load_arc_output(
    output_path: Path = TEST_OUTPUT_PATH, arc_id: str = "test_arc_id", iteration: int = 0
) -> ArcOutput:
    """
    Gets the numerical outputs of an arc for plot purposes.
    """

    path = output_path.joinpath(arc_id).joinpath(str(iteration)).joinpath("arc_output")
    arc_output = ArcOutput(
        simulation_parameters=load_simulation_parameters(
            output_path=output_path,
            arc_id=arc_id,
            name=str(iteration) + "/arc_output/simulation_parameters",
        ),
        t=array(object=load_base_model(name="t", path=path)),
        y=array(object=load_base_model(name="y", path=path)),
    )
    observation_timestamps, simulated_measurements = get_measurements(
        simulation_parameters=arc_output.simulation_parameters,
        name=str(iteration) + "/arc_output/simulated_measurements",
    )
    _, residuals = get_measurements(
        simulation_parameters=arc_output.simulation_parameters,
        name=str(iteration) + "/arc_output/residuals",
    )
    arc_output.observation_timestamps = observation_timestamps
    arc_output.simulated_measurements = simulated_measurements
    arc_output.residuals = residuals

    return arc_output
