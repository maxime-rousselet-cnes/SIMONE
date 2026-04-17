"""
Solves the normal equations and extrapolates the resulting orbit iteratively.
"""

from multiprocessing import Pool
from pathlib import Path
from typing import Optional

from base_models import load_base_model
from numpy import array, concatenate, diag, inf, matmul, mean, ndarray, zeros, zeros_like
from numpy.linalg import cholesky, inv

from .base_constants import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_STATIONS_FILE_NAME,
    DFAULT_CONVERGENCE_THRESHOLD,
    TEST_INVERSION_NAME,
    TEST_NO_ITERATIONS_PATH,
    TEST_OUTPUT_PATH,
)
from .quadrature import build_normal_equations, propagate_partials_and_save, save_normal_equations
from .simulation_parameters import SimulationParameters
from .station import get_stations

PARAMETER_KINDS = ["dynamic", "station"]


def solve_normal_equations(n_matrix: ndarray, s_second_member: ndarray) -> tuple[ndarray, ndarray]:
    """
    Uses the Cholesky method to invert the square system.
    """

    l_matrix = cholesky(n_matrix)
    l_matrix_inverse = inv(a=l_matrix)
    l_transpose_matrix_inverse = inv(a=l_matrix.T)
    z_second_member = matmul(l_matrix_inverse, s_second_member)
    n_inverse_matrix = matmul(l_transpose_matrix_inverse, l_matrix_inverse)
    correlation_matrix = zeros_like(a=n_inverse_matrix)

    for i in range(len(correlation_matrix)):

        for j in range(len(correlation_matrix)):

            correlation_matrix[i, j] = (
                n_inverse_matrix[i, j] / (n_inverse_matrix[i, i] * n_inverse_matrix[j, j]) ** 0.5
            )

    return matmul(l_transpose_matrix_inverse, z_second_member), correlation_matrix


def run_single_arc(
    simulation_parameters: SimulationParameters,
    stations_path: Path = TEST_NO_ITERATIONS_PATH,
    station_file_name: str = DEFAULT_STATIONS_FILE_NAME,
    path: Path = TEST_NO_ITERATIONS_PATH,
    parameters_initial_guess: Optional[
        dict[str, float]
    ] = None,  # To be updated at every iteration.
) -> Path:
    """
    Provides the arc normal equations.
    """

    if parameters_initial_guess is None:

        parameters_initial_guess = {}

    parameters_initial_guess_dict = {
        "dynamic": {
            parameter: initial_guess
            for parameter, initial_guess in parameters_initial_guess.items()
            if "station" not in parameter
        },
        "station": {
            parameter: initial_guess
            for parameter, initial_guess in parameters_initial_guess.items()
            if "station" in parameter
        },
    }

    for initial_guesses in parameters_initial_guess_dict.values():

        for parameter, initial_guess in initial_guesses.items():

            simulation_parameters.terminal_parameter_values[parameter] = initial_guess

    stations = get_stations(path=stations_path, station_file_name=station_file_name)
    simulation_parameters.update_for_stations(stations=stations)
    simulation_parameters.arc_parameters.is_initial = False
    arc_output, observation_partials, arc_parameters_to_invert, station_id_per_observation = (
        propagate_partials_and_save(
            stations=stations,
            simulation_parameters=simulation_parameters,
            parameters_to_invert={
                key: None if initial_guesses is None else list(initial_guesses.keys())
                for key, initial_guesses in parameters_initial_guess_dict.items()
            },
            path=path,
        )
    )

    return save_normal_equations(
        products=tuple(
            list(
                build_normal_equations(
                    observation_partials=observation_partials, arc_output=arc_output
                )
            )
            + [
                station_id_per_observation,
                arc_parameters_to_invert,
            ]
        ),
        simulation_parameters=simulation_parameters,
        output_path=path,
    )


def load_normal_equations(
    path: Path,
    stations_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = DEFAULT_STATIONS_FILE_NAME,
) -> tuple[ndarray, ndarray, ndarray, dict[str, Optional[list[str]]]]:
    """
    Loads the normal equations for a single arc and builds the corresponding a priori weight matrix.
    """

    a_matrix = array(object=load_base_model(name="a_matrix", path=path), dtype=float)
    b_second_member = array(object=load_base_model(name="b_second_member", path=path), dtype=float)
    station_id_per_observation = load_base_model(name="station_id_per_observation", path=path)
    arc_parameters_to_invert = load_base_model(name="parameters", path=path)
    stations = get_stations(path=stations_path, station_file_name=station_file_name)
    weight_matrix = diag(
        v=[
            stations[station_id].station_simulation.sigma_noise
            for station_id in station_id_per_observation
        ]
    )

    return a_matrix, b_second_member, weight_matrix, arc_parameters_to_invert


def build_arc_parameter_indices(
    overall_index: int,
    arc_parameters_to_invert: dict[str, Optional[list[str]]],
    parameters_to_cumulate_indices: dict[str, int],
    arc_parameter_indices: list[dict[str, dict[str, int]]],
    i_arc: int,
) -> tuple[int, list[dict[str, dict[str, int]]]]:
    """
    Builds the indices of the parameters to invert for a single arc, and updates the overall index.
    """

    arc_parameter_index = 0

    for parameter_kind in PARAMETER_KINDS:

        for parameter in arc_parameters_to_invert[parameter_kind]:

            if parameter in parameters_to_cumulate_indices:

                if parameters_to_cumulate_indices[parameter] == -1:  # Not stored already.

                    parameters_to_cumulate_indices[parameter] = overall_index
                    overall_index += 1

                arc_parameter_indices[i_arc][parameter_kind][parameter] = (
                    parameters_to_cumulate_indices[parameter]
                )

            else:

                arc_parameter_indices[i_arc][parameter_kind][parameter] = overall_index
                overall_index += 1

            arc_parameter_index += 1
    return overall_index, arc_parameter_indices


def gather_normal_equations(
    normal_equation_path_per_arc: list[Path],
    inversion_path: Path,
    station_file_name: str,
    parameters_to_cumulate_indices: dict[str, int],
) -> tuple[int, list[ndarray], list[ndarray], list[ndarray], list[dict[str, dict[str, int]]]]:
    """
    Loads the normal equations of every arc and builds the corresponding indices needed for
    cumulating.
    """

    b_second_member_per_arc = []
    overall_index = 0
    arc_parameter_indices: list[dict[str, dict[str, int]]] = [
        {parameter_kind: {} for parameter_kind in PARAMETER_KINDS}
        for _ in normal_equation_path_per_arc
    ]
    all_n_matrices = []
    all_s_second_members = []

    for i_arc, _ in enumerate(normal_equation_path_per_arc):

        a_matrix, b_second_member, weight_matrix, arc_parameters_to_invert = load_normal_equations(
            path=_,
            stations_path=inversion_path.parent,
            station_file_name=station_file_name,
        )
        b_second_member_per_arc += [b_second_member]
        atw_matrix = matmul(a_matrix.T, weight_matrix)
        all_n_matrices += [array(object=matmul(atw_matrix, a_matrix), dtype=float)]
        all_s_second_members += [array(object=matmul(atw_matrix, b_second_member), dtype=float)]
        overall_index, arc_parameter_indices = build_arc_parameter_indices(
            overall_index=overall_index,
            arc_parameters_to_invert=arc_parameters_to_invert,
            parameters_to_cumulate_indices=parameters_to_cumulate_indices,
            arc_parameter_indices=arc_parameter_indices,
            i_arc=i_arc,
        )

    return (
        overall_index,
        b_second_member_per_arc,
        all_n_matrices,
        all_s_second_members,
        arc_parameter_indices,
    )


def cumulate(
    overall_index: int,
    arc_parameter_indices: list[dict[str, dict[str, int]]],
    all_n_matrices: list[ndarray],
    all_s_second_members: list[ndarray],
) -> tuple[ndarray, ndarray]:
    """
    Cumulates common parameter between the arcs.
    """

    n_cumulated_matrix = zeros(shape=(overall_index, overall_index))
    s_cumulated_second_member = zeros(shape=(overall_index, 1))

    for i_arc, (n_matrix_arc, s_vector_arc) in enumerate(zip(all_n_matrices, all_s_second_members)):

        local_to_global = []

        for parameter_kind in PARAMETER_KINDS:

            local_to_global += list(arc_parameter_indices[i_arc][parameter_kind].values())

        for i_local, i_global in enumerate(local_to_global):

            s_cumulated_second_member[i_global][0] += s_vector_arc[i_local][0]

            for j_local, j_global in enumerate(local_to_global):

                n_cumulated_matrix[i_global, j_global] += n_matrix_arc[i_local, j_local]

    return n_cumulated_matrix, s_cumulated_second_member


def cumulate_and_solve_normal_equations(
    mean_residual_history: list[float],
    normal_equation_path_per_arc: list[Path],
    inversion_path: Path,
    station_file_name: str,
    parameters_to_cumulate: Optional[list[str]] = None,
) -> tuple[ndarray, ndarray, list[dict[str, dict[str, int]]], bool, list[float]]:
    """
    Uses the normal equations for every arc to cumulate and solve globally
    """

    if parameters_to_cumulate is None:

        parameters_to_cumulate = []

    (
        overall_index,
        b_second_member_per_arc,
        all_n_matrices,
        all_s_second_members,
        arc_parameter_indices,
    ) = gather_normal_equations(
        normal_equation_path_per_arc=normal_equation_path_per_arc,
        inversion_path=inversion_path,
        station_file_name=station_file_name,
        parameters_to_cumulate_indices={parameter: -1 for parameter in parameters_to_cumulate},
    )

    mean_residual_history += [mean(abs(concatenate(b_second_member_per_arc)).flatten())]

    if abs(mean_residual_history[-2] - mean_residual_history[-1]) < DFAULT_CONVERGENCE_THRESHOLD:

        return array(object=()), array(object=()), [], True, mean_residual_history

    n_cumulated_matrix, s_cumulated_second_member = cumulate(
        overall_index=overall_index,
        arc_parameter_indices=arc_parameter_indices,
        all_n_matrices=all_n_matrices,
        all_s_second_members=all_s_second_members,
    )
    delta_x_solution, n_matrix_inverse = solve_normal_equations(
        n_matrix=n_cumulated_matrix, s_second_member=s_cumulated_second_member
    )

    return (
        delta_x_solution.flatten(),
        n_matrix_inverse,
        arc_parameter_indices,
        False,
        mean_residual_history,
    )


def update_data_structures(
    arc_parameter_indices: list[dict[str, dict[str, int]]],
    simulation_parameters_per_arc: list[SimulationParameters],
    delta_x_solution: ndarray,
) -> list[Optional[dict[str, float]]]:
    """
    Updates all data structures for next iteration, for every arc.
    """

    parameter_values_per_arc = []

    for i_arc, parameter_indices_per_kind in enumerate(arc_parameter_indices):

        for parameter_indices in parameter_indices_per_kind.values():

            for parameter, index in parameter_indices.items():

                if parameter not in simulation_parameters_per_arc[i_arc].terminal_parameter_values:

                    simulation_parameters_per_arc[i_arc].terminal_parameter_values[parameter] = 0

                simulation_parameters_per_arc[i_arc].terminal_parameter_values[
                    parameter
                ] += delta_x_solution[index]

            parameter_values_per_arc += [
                {
                    parameter: simulation_parameters_per_arc[i_arc].terminal_parameter_values[
                        parameter
                    ]
                    for parameter in parameter_indices
                }
            ]

    return parameter_values_per_arc


def solve_precise_orbit_determination(
    simulation_parameters_per_arc: list[SimulationParameters],
    inversion_path: Path = TEST_OUTPUT_PATH.joinpath(TEST_INVERSION_NAME),
    parameters_values_initial_guess_per_arc: Optional[list[Optional[dict[str, float]]]] = None,
    station_file_name: str = DEFAULT_STATIONS_FILE_NAME,
    parameters_to_cumulate: Optional[list[str]] = None,
) -> tuple[list[dict[str, float]], list[ndarray], list[dict[str, dict[str, int]]]]:
    """
    Performs a loop on:
        - Forward simulation of the different arcs and partial derivatives.
        - Solving of the normal equations.
    """

    if parameters_values_initial_guess_per_arc is None:

        parameters_values_initial_guess_per_arc = [None] * len(simulation_parameters_per_arc)

    mean_residual_history = [inf]
    parameter_values_per_iteration_per_arc: list[list[Optional[dict[str, float]]]] = [
        parameters_values_initial_guess_per_arc
    ]
    correlations_per_iterations: list[ndarray] = []

    for iteration in range(DEFAULT_MAX_ITERATIONS):

        print("Iteration", iteration)

        with Pool() as p:

            normal_equation_path_per_arc = p.starmap(
                run_single_arc,
                [
                    (
                        simulation_parameters,
                        inversion_path.parent,
                        station_file_name,
                        inversion_path.joinpath(str(iteration)),
                        parameter_values,
                    )
                    for simulation_parameters, parameter_values in zip(
                        simulation_parameters_per_arc, parameter_values_per_iteration_per_arc[-1]
                    )
                ],
            )

        arc_parameter_indices: list[dict[str, dict[str, int]]]
        (
            delta_x_solution,
            n_matrix_inverse,
            arc_parameter_indices,
            breaker_flag,
            mean_residual_history,
        ) = cumulate_and_solve_normal_equations(
            mean_residual_history=mean_residual_history,
            normal_equation_path_per_arc=normal_equation_path_per_arc,
            inversion_path=inversion_path,
            station_file_name=station_file_name,
            parameters_to_cumulate=parameters_to_cumulate,
        )

        if breaker_flag:

            break

        parameter_values_per_iteration_per_arc += [
            update_data_structures(
                arc_parameter_indices=arc_parameter_indices,
                simulation_parameters_per_arc=simulation_parameters_per_arc,
                delta_x_solution=delta_x_solution,
            )
        ]
        correlations_per_iterations += [n_matrix_inverse]

    return (
        parameter_values_per_iteration_per_arc,
        correlations_per_iterations,
        arc_parameter_indices,
    )
