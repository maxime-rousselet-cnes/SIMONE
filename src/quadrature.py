"""
Integrates the partial derivatives.
"""

from pathlib import Path
from typing import Callable, Optional

from numpy import array, concatenate, diff, ndarray, zeros
from sympy import Expr, Matrix, MutableDenseMatrix, Symbol, flatten, lambdify, symbols

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
from .utils import STATE_VECTOR_LINE, evaluate_terminal_parameters, save_base_model


def vector_variation_equation(
    dynamic: MutableDenseMatrix, parameter: Expr, partials: MutableDenseMatrix
) -> MutableDenseMatrix:
    """
    Applies the variation method to algebraically derive the time-dependent behavior of a partial
    derivative to integrate on the satellite's dynamic quadrature points.
    """

    return Matrix(
        [
            [variation_equation(expression=expression, parameter=parameter, partials=partials)]
            for expression in dynamic.flat()
        ]
    )


def variation_equation(expression: Expr, parameter: Expr, partials: MutableDenseMatrix) -> Expr:
    """
    Applies the variation method to algebraically derive a partial derivative expression with
    respect to a parameter.
    """

    return MutableDenseMatrix(
        [expression.diff(state_parameter) for state_parameter in STATE_VECTOR_LINE]
    ).dot(b=partials) + expression.diff(parameter)


def fixed_timestep_integrator(
    fun: Callable, t: ndarray[float], y: ndarray[float], i_parameter: int
) -> ndarray[float]:
    """
    Performs the numerical quadrature of partial derivatives over the orbit timesteps.
    """

    numerical_partials = [zeros(shape=6)]
    dt_array = diff(a=t)

    if i_parameter < 6:

        # Because dx^i/dx^i_0(t=0) := 1.
        numerical_partials[0][i_parameter] = 1

    for timestep, dt, state_vector in zip(t[1:], dt_array, y[:-1]):

        numerical_partials += [
            numerical_partials[-1]
            + dt * array(object=fun(timestep, state_vector, numerical_partials[-1]))
        ]

    return numerical_partials


def integrate_variation_equations(
    arc_output: ArcOutput,
    simulation_parameters: SimulationParameters,
    generalized_symbolic_propagator: MutableDenseMatrix,
    parameters_to_invert: Optional[list[str]],
) -> tuple[dict[str, ndarray[float]], dict[str, list[Expr]], list[str]]:
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
    partials: dict[str, list[Expr]] = {}
    numerical_partials: dict[str, ndarray[float]] = {}

    for i_parameter, parameter in enumerate(parameters_to_invert):

        partials[parameter] = [
            Symbol(
                r"\frac{\partial state_parameter}{\partial parameter}".replace(
                    "state_parameter", str(state_parameter)
                ).replace("parameter", str(parameter))
            )
            for state_parameter in STATE_VECTOR_LINE
        ]
        partials_matrix_for_parameter = MutableDenseMatrix(
            [[partial] for partial in partials[parameter]]
        )
        variation_equations = evaluate_terminal_parameters(
            expression=vector_variation_equation(
                dynamic=generalized_symbolic_propagator,
                parameter=parameter,
                partials=partials_matrix_for_parameter,
            ),
            parameter_expressions=simulation_parameters.parameter_expressions,
            terminal_parameter_values=simulation_parameters.terminal_parameter_values,
        )
        numerical_partials[parameter] = array(
            object=fixed_timestep_integrator(
                fun=lambdify(
                    args=[simulation_parameters.parameter_expressions[r"t"]]
                    + [STATE_VECTOR_LINE]
                    + [partials[parameter]],
                    expr=flatten(variation_equations),
                ),
                t=arc_output.t,
                y=arc_output.y,
                i_parameter=i_parameter,
            )
        )

    return numerical_partials, partials, [str(parameter) for parameter in parameters_to_invert]


def integrate_observation_partials(
    arc_output: ArcOutput,
    state_vector_partials_per_parameter_expressions: dict[str, list[Expr]],
    state_vector_partials_per_parameter_values: dict[str, ndarray[float]],
    measurement_expressions: dict[str, Expr],
    observation_timestamps: dict[str, list[float]],
) -> dict[str, ndarray[float]]:  # Per station, per parameter.
    """
    Uses the Leibniz formula to numerically derive the partial derivative of observations to invert
    from the numerical values of the partial derivative of the state parameters obtained via the
    variation equations.
    """

    obsevation_partials: dict[str, ndarray[float]] = {}

    for parameter, partials in state_vector_partials_per_parameter_expressions.items():

        obsevation_partials[parameter] = zeros(shape=0)

        for station_id, timestamps in observation_timestamps.items():

            # DQ/Dgamma = <(nabla Q)|(dx/dgamma)> + dQ/dgamma.
            obsevation_partials[parameter] = concatenate(
                (
                    obsevation_partials[parameter],
                    interpolate_function_of_position(
                        expression=variation_equation(
                            expression=measurement_expressions[station_id],
                            parameter=parameter,
                            partials=partials,
                        ),
                        arc_output=arc_output,
                        observation_timestamps=timestamps,
                        partials=partials,
                        partial_numerical_values=state_vector_partials_per_parameter_values[
                            parameter
                        ],
                    ),
                )
            )

    return obsevation_partials


def save_normal_equations(
    n_matrix: ndarray[float],
    s_second_member: ndarray[float],
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


def propagate_partials_and_save(
    stations: dict[str, Station],
    simulation_parameters: SimulationParameters,
    iteration: int = 0,
    parameters_to_invert: Optional[list[str]] = None,
    output_path: Path = TEST_OUTPUT_PATH,
) -> tuple[ArcOutput, dict[str, ndarray[float]], list[str]]:
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
    theoretical_measurement_values, measurement_expressions = generate_measurements(
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
    numerical_partials, partials, parameters_to_invert = integrate_variation_equations(
        arc_output=arc_output,
        simulation_parameters=simulation_parameters,
        generalized_symbolic_propagator=generalized_symbolic_propagator,
        parameters_to_invert=parameters_to_invert,
    )

    return (
        arc_output,
        integrate_observation_partials(
            arc_output=arc_output,
            state_vector_partials_per_parameter_expressions=partials,
            state_vector_partials_per_parameter_values=numerical_partials,
            measurement_expressions=measurement_expressions,
            observation_timestamps=observation_timestamps,
        ),
        parameters_to_invert,
    )
