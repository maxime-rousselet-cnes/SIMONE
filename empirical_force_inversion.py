"""
Adapted from test.py: adjust simultaneously J_2 and empirical accelerations for synthetic
observability discussion.
"""

from pathlib import Path
from test import (
    TEST_ARC_PARAMETERS,
    simulate_observations,
    test_clear_test_folder,
    test_generate_stations,
)
from time import time
from typing import Optional

from base_models import load_base_model, save_base_model
from matplotlib.pyplot import GridSpec, figure, tight_layout
from sympy import Symbol

from simone import (
    DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
    DEFAULT_STATIONS_FILE_NAME,
    TEST_NO_ITERATIONS_NAME,
    EmpiricalForceParameters,
    Parameters,
    SimulationParameters,
    solve_precise_orbit_determination,
)

TEST_EMPIRICAL_SIMULATED_FORCES: dict[str, Optional[Parameters]] = {
    "central_body_attraction": None,
    "j2_attraction": None,
    "empirical_force": EmpiricalForceParameters(
        b_parameters=[Symbol(r"b_r"), Symbol(r"b_t"), Symbol(r"b_n")],
        periodic_parameters=[
            Symbol(r"c_r"),
            Symbol(r"c_t"),
            Symbol(r"c_n"),
            Symbol(r"s_r"),
            Symbol(r"s_t"),
            Symbol(r"s_n"),
        ],
    ),
}
NUMERICAL_TOLERANCE = 1e-9
TEST_EMPIRICAL_FORCE_OUTPUT_PATH = Path("empirical_force_tests")
TEST_EMPIRICAL_FORCE_NO_ITERATIONS_PATH = TEST_EMPIRICAL_FORCE_OUTPUT_PATH.joinpath(
    TEST_NO_ITERATIONS_NAME
)
TEST_EMPIRICAL_FORCE_INVERSION_PATH = TEST_EMPIRICAL_FORCE_OUTPUT_PATH.joinpath("Inverting both")
TRUE_J_2_MODEL = 1e-3
WRONG_J_2_MODEL = TRUE_J_2_MODEL - TRUE_J_2_MODEL / 1e5
if __name__ == "__main__":
    """
    TODO.
    """

    # Initialization.
    t_0 = time()
    TEST_ARC_PARAMETERS.arc_length = 86400.0
    test_clear_test_folder(path=TEST_EMPIRICAL_FORCE_OUTPUT_PATH)
    simulation_parameters = SimulationParameters(
        arc_parameters=TEST_ARC_PARAMETERS,
        simulated_forces=TEST_EMPIRICAL_SIMULATED_FORCES,
        terminal_parameter_values={r"J_2": TRUE_J_2_MODEL},
    )
    simulation_parameters.save(
        path=TEST_EMPIRICAL_FORCE_NO_ITERATIONS_PATH, name=DEFAULT_SIMULATION_PARAMETERS_FILE_NAME
    )
    test_generate_stations(
        output_path=TEST_EMPIRICAL_FORCE_OUTPUT_PATH,
        simulation_parameters_path=TEST_EMPIRICAL_FORCE_NO_ITERATIONS_PATH,
    )

    # Fixed J_2 with error.
    print("Fixed J_2")
    simulation_parameters, _, _, _ = simulate_observations(
        stations_path=TEST_EMPIRICAL_FORCE_OUTPUT_PATH,
        save_path=TEST_EMPIRICAL_FORCE_OUTPUT_PATH.joinpath("Fixed J_2"),
        station_file_name=DEFAULT_STATIONS_FILE_NAME,
        simulation_parameters_path=TEST_EMPIRICAL_FORCE_NO_ITERATIONS_PATH,
        simulation_parameters_file_name=DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
    )
    simulation_parameters.terminal_parameter_values[r"J_2"] = WRONG_J_2_MODEL
    solve_precise_orbit_determination(
        simulation_parameters_per_arc=[simulation_parameters],
        inversion_path=TEST_EMPIRICAL_FORCE_OUTPUT_PATH.joinpath("Fixed J_2"),
        parameters_values_initial_guess_per_arc=[None],
        station_file_name=DEFAULT_STATIONS_FILE_NAME,
    )
    print()

    # Inverted J_2.
    print("Inverted J_2")
    simulation_parameters, _, _, _ = simulate_observations(
        stations_path=TEST_EMPIRICAL_FORCE_OUTPUT_PATH,
        save_path=TEST_EMPIRICAL_FORCE_OUTPUT_PATH.joinpath("Inverted J_2"),
        station_file_name=DEFAULT_STATIONS_FILE_NAME,
        simulation_parameters_path=TEST_EMPIRICAL_FORCE_NO_ITERATIONS_PATH,
        simulation_parameters_file_name=DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
    )
    solve_precise_orbit_determination(
        simulation_parameters_per_arc=[simulation_parameters],
        inversion_path=TEST_EMPIRICAL_FORCE_OUTPUT_PATH.joinpath("Inverted J_2"),
        parameters_values_initial_guess_per_arc=[{r"J_2": WRONG_J_2_MODEL}],
        station_file_name=DEFAULT_STATIONS_FILE_NAME,
    )
    print()

    # Inverted empirical acceleration.
    print("Inverted empirical acceleration")
    simulation_parameters, _, _, _ = simulate_observations(
        stations_path=TEST_EMPIRICAL_FORCE_OUTPUT_PATH,
        save_path=TEST_EMPIRICAL_FORCE_OUTPUT_PATH.joinpath("Inverted empirical acceleration"),
        station_file_name=DEFAULT_STATIONS_FILE_NAME,
        simulation_parameters_path=TEST_EMPIRICAL_FORCE_NO_ITERATIONS_PATH,
        simulation_parameters_file_name=DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
    )
    simulation_parameters.terminal_parameter_values[r"J_2"] = WRONG_J_2_MODEL
    solve_precise_orbit_determination(
        simulation_parameters_per_arc=[simulation_parameters],
        inversion_path=TEST_EMPIRICAL_FORCE_OUTPUT_PATH.joinpath("Inverted empirical acceleration"),
        parameters_values_initial_guess_per_arc=[{r"b_n": 0.0, r"c_n": 0.0, r"s_n": 0.0}],
        station_file_name=DEFAULT_STATIONS_FILE_NAME,
    )
    print()

    # Inverting both and showing correlation.
    print("Inverting both")
    simulation_parameters, _, _, _ = simulate_observations(
        stations_path=TEST_EMPIRICAL_FORCE_OUTPUT_PATH,
        save_path=TEST_EMPIRICAL_FORCE_INVERSION_PATH,
        station_file_name=DEFAULT_STATIONS_FILE_NAME,
        simulation_parameters_path=TEST_EMPIRICAL_FORCE_NO_ITERATIONS_PATH,
        simulation_parameters_file_name=DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
    )
    parameter_values_per_iteration_per_arc, correlations_per_iterations, arc_parameter_indices = (
        solve_precise_orbit_determination(
            simulation_parameters_per_arc=[simulation_parameters],
            inversion_path=TEST_EMPIRICAL_FORCE_INVERSION_PATH,
            parameters_values_initial_guess_per_arc=[
                {r"J_2": WRONG_J_2_MODEL, r"b_n": 0.0, r"c_n": 0.0, r"s_n": 0.0}
            ],
            station_file_name=DEFAULT_STATIONS_FILE_NAME,
        )
    )
    save_base_model(
        obj={
            "parameter_values_per_iteration_per_arc": parameter_values_per_iteration_per_arc,
            "correlations_per_iterations": correlations_per_iterations,
            "arc_parameter_indices": arc_parameter_indices,
        },
        name="free_result",
        path=TEST_EMPIRICAL_FORCE_INVERSION_PATH,
    )

    # Figure generation.
    result = load_base_model(name="free_result", path=TEST_EMPIRICAL_FORCE_INVERSION_PATH)
    correlations = result["correlations_per_iterations"][-1]
    parameter_indices: dict = result["arc_parameter_indices"][0]["dynamic"]
    fig = figure(figsize=(8, 6))
    grid = GridSpec(nrows=1, ncols=1, figure=fig)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_title("Toy-model arc correlations", fontweight="bold")
    im_solution = ax.imshow(correlations, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    labels = [r"$" + parameter + "$" for parameter in parameter_indices.keys()]
    ax.set_xticks(ticks=range(len(parameter_indices)), labels=labels)
    ax.set_yticks(ticks=range(len(parameter_indices)), labels=labels)
    cbar = ax.figure.colorbar(im_solution, ax=ax)
    cbar.set_ticks([-1.0, 0.0, 1.0])
    cbar.set_ticklabels(["-100 %: Co-linear", "0 %: De-correlated", "100 %: Co-linear"])

    for i_parameter, parameter_i in enumerate(parameter_indices.keys()):

        for j_parameter, parameter_j in enumerate(parameter_indices.keys()):

            ax.text(
                j_parameter,
                i_parameter,
                f"{correlations[i_parameter][j_parameter]:.2g}",
                ha="center",
                va="center",
                color="black" if abs(correlations[i_parameter][j_parameter]) < 0.7 else "white",
            )

    tight_layout()
    fig.savefig(
        fname=TEST_EMPIRICAL_FORCE_INVERSION_PATH.joinpath("toy_model_correlations.pdf"),
        bbox_inches="tight",
    )
    print(time() - t_0)
