"""
Integrates the partial derivatives.
"""

from pathlib import Path
from typing import Optional

from base_models import (
    evaluate_terminal_parameters,
    fixed_timestep_integrator,
    partial_symbols,
    save_base_model,
    variation_equation,
    vector_variation_equation,
)
from numpy import concatenate, ndarray, zeros, zeros_like
from sympy import Expr, MutableDenseMatrix, Symbol, flatten, lambdify, symbols
from sympy.core.numbers import Zero

from .base_constants import TEST_OUTPUT_PATH
from .forward_simulation import propagate_ephemeris
from .observation import (
    ArcOutput,
    generate_measurements,
    get_measurements,
    interpolate_function_of_position,
)
from .simulation_parameters import SimulationParameters
from .station import Station
from .utils import STATE_VECTOR_LINE


def integrate_variation_equations(
    arc_output: ArcOutput,
    simulation_parameters: SimulationParameters,
    generalized_symbolic_propagator: MutableDenseMatrix,
    parameters_to_invert: Optional[list[str]],
) -> tuple[dict[str, tuple[list[Expr], ndarray]], list[str]]:
    """
    Integrates through time the partial derivative of every state parameter with respect to every
    invertible parameter.
    """

    parameters_to_invert: list[Expr] = list(
        symbols(r"x_0 y_0 z_0 \dot{x}_0 \dot{y}_0 \dot{z}_0")
    ) + (
        []
        if not parameters_to_invert
        else [
            simulation_parameters.parameter_expressions[parameter]
            for parameter in parameters_to_invert
        ]
    )
    partials: dict[str, tuple[list[Expr], ndarray]] = {}

    for i_parameter, parameter in enumerate(parameters_to_invert):

        partial_expressions, partials_matrix_for_parameter = partial_symbols(
            parameter=parameter, state_vector_line=STATE_VECTOR_LINE
        )
        variation_equations = evaluate_terminal_parameters(
            expression=vector_variation_equation(
                dynamic=generalized_symbolic_propagator,
                parameter=parameter,
                partials=partials_matrix_for_parameter,
                state_vector_line=STATE_VECTOR_LINE,
            ),
            parameter_expressions=simulation_parameters.parameter_expressions,
            terminal_parameter_values=simulation_parameters.terminal_parameter_values,
        )
        partial_numerical_values = fixed_timestep_integrator(
            fun=lambdify(
                args=[
                    simulation_parameters.parameter_expressions[r"t"],
                    STATE_VECTOR_LINE,
                    partial_expressions,
                ],
                expr=flatten(variation_equations),
            ),
            t=arc_output.t,
            y=arc_output.y,
            i_parameter_initial_conditions=i_parameter,
        )
        partials[str(parameter)] = (partial_expressions, partial_numerical_values)

    return partials, [str(parameter) for parameter in parameters_to_invert]


def integrate_observation_partials(
    arc_output: ArcOutput,
    state_vector_partials_per_parameter: dict[str, tuple[list[Expr], ndarray]],
    stations: dict[str, Station],
    measurement_expression: Expr,
    observation_timestamps: dict[str, list[float]],
) -> dict[str, ndarray]:  # Per station, per parameter.
    """
    Uses the Leibniz formula to numerically derive the partial derivative of observations to invert
    from the numerical values of the partial derivative of the state parameters obtained via the
    variation equations.
    """

    obsevation_partials: dict[str, ndarray] = {}

    for parameter, (partials, numerical_partials) in state_vector_partials_per_parameter.items():

        # DQ/Dgamma = <(nabla Q)|(dx/dgamma)> + dQ/dgamma.
        observation_variation_equation = variation_equation(
            expression=measurement_expression,
            parameter=Symbol(parameter),
            partials=partials,
            state_vector_line=STATE_VECTOR_LINE,
        )
        obsevation_partials[parameter] = zeros(shape=0)

        for station_id, timestamps in observation_timestamps.items():

            # Takes advantage of the independence of stations relative to each other.
            obsevation_partials[
                (
                    parameter
                    if "station" not in parameter
                    else parameter.replace("station", station_id)
                )
            ] = concatenate(
                (
                    obsevation_partials[parameter],
                    interpolate_function_of_position(
                        expression=stations[station_id].apply_to_station(
                            expression=observation_variation_equation
                        ),
                        arc_output=arc_output,
                        observation_timestamps=timestamps,
                        partials=partials,
                        partial_numerical_values=numerical_partials,
                    ),
                )
            )

    return obsevation_partials


def save_normal_equations(
    n_matrix: ndarray,
    s_second_member: ndarray,
    simulation_parameters: SimulationParameters,
    iteration: int = 0,
    output_path: Path = TEST_OUTPUT_PATH,
) -> Path:
    """
    Saves normal equations in the previous iteration's folder.
    """

    path = (
        output_path.joinpath(simulation_parameters.arc_parameters.arc_id)
        .joinpath(str(iteration))
        .joinpath("normal_equations")
    )
    save_base_model(obj=n_matrix, name="n_matrix", path=path)
    save_base_model(obj=s_second_member, name="s_second_member", path=path)

    return path


def update_for_station_dependent_parameters(
    stations: dict[str, Station],
    y: ndarray,
    partials: dict[str, tuple[list[Expr], ndarray]],
    parameters_to_invert: dict[str, Optional[list[str]]],
) -> tuple[dict[str, tuple[list[Expr], ndarray]], dict[str, Optional[list[str]]]]:
    """
    Expands the 2 considered data structure by adding an unkon per station and per
    station_dependent unknown.
    """

    if parameters_to_invert["station"] is not None:

        for parameter in parameters_to_invert["station"]:

            partials[parameter] = (
                [Zero() for _ in range(6)],
                zeros_like(a=y),
            )

            for station_id in stations:

                parameters_to_invert["dynamic"] += [parameter.replace("station", station_id)]

    return partials, parameters_to_invert


def propagate_partials_and_save(
    stations: dict[str, Station],
    simulation_parameters: SimulationParameters,
    parameters_to_invert: dict[str, Optional[list[str]]],
    iteration: int = 0,
    output_path: Path = TEST_OUTPUT_PATH,
) -> tuple[ArcOutput, dict[str, ndarray], list[str]]:
    """
    Performs a forward simulation including partial derivatives computing that allows to build the A
    matrix. Also saves the arc output.
    """

    t, y, generalized_symbolic_propagator = propagate_ephemeris(
        simulation_parameters=simulation_parameters,
    )
    observation_timestamps, real_measurement_values = get_measurements(
        simulation_parameters=simulation_parameters
    )
    arc_output = ArcOutput(simulation_parameters=simulation_parameters, t=t, y=y)
    theoretical_measurement_values, measurement_expression = generate_measurements(
        arc_output=arc_output,
        stations=stations,
        observation_timestamps=observation_timestamps,
    )
    arc_output.update_observations(
        observation_timestamps=observation_timestamps,
        theoretical_measurements=theoretical_measurement_values,
        real_measurements=real_measurement_values,
    )
    arc_output.save(output_path=output_path, iteration=iteration)
    partials, parameters_to_invert["dynamic"] = integrate_variation_equations(
        arc_output=arc_output,
        simulation_parameters=simulation_parameters,
        generalized_symbolic_propagator=generalized_symbolic_propagator,
        parameters_to_invert=parameters_to_invert["dynamic"],
    )
    partials, parameters_to_invert = update_for_station_dependent_parameters(
        stations=stations, y=y, partials=partials, parameters_to_invert=parameters_to_invert
    )

    return (
        arc_output,
        integrate_observation_partials(
            arc_output=arc_output,
            state_vector_partials_per_parameter=partials,
            stations=stations,
            measurement_expression=measurement_expression,
            observation_timestamps=observation_timestamps,
        ),
        parameters_to_invert["dynamic"],
    )
