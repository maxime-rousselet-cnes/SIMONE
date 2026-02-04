"""
Solves the normal equations and extrapolates the resulting orbit iteratively.
"""

# TODO: Cumulates.

from pathlib import Path

from numpy import array, concatenate, inf, matmul, mean, ndarray, pi, zeros_like
from numpy.linalg import cholesky, inv

from src.base_constants import (
    DEFAULT_MAX_ITERATIONS,
    DFAULT_CONVERGENCE_THRESHOLD,
    TEST_OUTPUT_PATH,
)
from src.quadrature import propagate_partials_and_save, save_normal_equations
from src.simulation_parameters import load_simulation_parameters
from src.station import get_stations
from src.test import TEST_ARC_PARAMETERS
from src.utils import save_base_model


def solve_normal_equations(
    n_matrix: ndarray[float], s_second_member: ndarray[float]
) -> tuple[ndarray[float], ndarray[float]]:
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


def solve_iteratively_precise_orbit_determination(
    output_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = "stations",
    simulation_parameters_file_name: str = "simulation_parameters",
    arc_id: str = TEST_ARC_PARAMETERS.arc_id,
    parameters_initial_guess: dict[str, float] = {
        r"J_2": 9e-3,
        r"\omega_{Earth\ rotation\ angular\ speed}": 2 * pi / 86100,
    },
) -> tuple[list[dict[str, float]], list[ndarray[float]]]:
    """
    Performs a loop on:
        - Forward simulation of the orbit and partial derivatives.
        - Solving of the normal equations.
    """

    simulation_parameters = load_simulation_parameters(
        output_path=output_path,
        arc_id=arc_id,
        name=simulation_parameters_file_name,
    )
    simulation_parameters.arc_parameters.is_initial = False
    stations = get_stations(stations_path=output_path, station_file_name=station_file_name)
    simulation_parameters.update_for_stations(stations=stations)
    observation_partials: dict[str, ndarray[float]]
    mean_residuals_history = [inf]
    parameter_values_per_iterations: list[dict[str, float]] = [parameters_initial_guess]
    correlations_per_iterations: list[ndarray[float]] = []

    for parameter, initial_guess in parameters_initial_guess.items():

        simulation_parameters.terminal_parameter_values[parameter] = initial_guess

    for iteration in range(DEFAULT_MAX_ITERATIONS):

        print("Iteration", iteration)

        arc_output, observation_partials, all_parameters_to_invert = propagate_partials_and_save(
            stations=stations,
            simulation_parameters=simulation_parameters,
            iteration=iteration,
            parameters_to_invert=parameters_initial_guess.keys(),
        )
        mean_residuals = mean(abs(concatenate(list(arc_output.residuals.values()))))

        if abs(mean_residuals - mean_residuals_history[-1]) < DFAULT_CONVERGENCE_THRESHOLD:

            break

        mean_residuals_history += [mean_residuals]

        a_matrix = array(object=list(observation_partials.values())).T
        b_second_member = concatenate(list(arc_output.residuals.values()))[:, None]
        n_matrix = array(object=matmul(a_matrix.T, a_matrix), dtype=float)
        s_second_member = array(object=matmul(a_matrix.T, b_second_member), dtype=float)
        path = save_normal_equations(
            n_matrix=n_matrix,
            s_second_member=s_second_member,
            simulation_parameters=simulation_parameters,
            iteration=iteration,
        )
        save_base_model(obj=all_parameters_to_invert, name="parameters", path=path)
        delta_x_solution, n_matrix_inverse = solve_normal_equations(
            n_matrix=n_matrix, s_second_member=s_second_member
        )

        for delta_x, parameter in zip(delta_x_solution.flatten(), all_parameters_to_invert):

            if parameter in simulation_parameters.terminal_parameter_values:

                simulation_parameters.terminal_parameter_values[parameter] += delta_x

        parameter_values_per_iterations += [
            {
                parameter: simulation_parameters.terminal_parameter_values[parameter]
                for parameter in parameters_initial_guess
            }
        ]
        correlations_per_iterations += [n_matrix_inverse]

    return parameter_values_per_iterations, correlations_per_iterations
