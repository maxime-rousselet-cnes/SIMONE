"""
Describes a forward simulation of an arc for a satellite
"""

from base_models import adaptive_runge_kutta_45, evaluate_terminal_parameters
from numpy import array, ndarray
from sympy import MutableDenseMatrix, evaluate, flatten, lambdify

from .dynamics import symbolic_propagator
from .ephemeris import OrbitalParameters
from .simulation_parameters import SimulationParameters
from .utils import STATE_VECTOR_LINE, STATE_VECTOR_MATRIX


def generate_numerically_initial_condition(
    simulation_parameters: SimulationParameters,
    initial_conditions: OrbitalParameters = OrbitalParameters(),
) -> ndarray:
    """
    Initial conditions numerically evaluated if parameterized.
    """

    return array(
        object=evaluate_terminal_parameters(
            expression=initial_conditions.to_cartesian_state(
                gravitational_parameter=simulation_parameters.parameter_expressions[
                    r"\mu_{gravitational\ parameter}"
                ],
            ),
            parameter_expressions=simulation_parameters.parameter_expressions,
            terminal_parameter_values=simulation_parameters.terminal_parameter_values,
        ),
        dtype=float,
    ).flatten()


def propagate_ephemeris(
    simulation_parameters: SimulationParameters,
    initial_conditions: OrbitalParameters = OrbitalParameters(),
) -> tuple[
    list[float],
    list[list[float]],
    MutableDenseMatrix,
]:
    """
    Integration of motion.
    """

    with evaluate(False):

        generalized_symbolic_propagator: MutableDenseMatrix = symbolic_propagator(
            state_vector=STATE_VECTOR_MATRIX,
            simulation_parameters=simulation_parameters,
        )

    t, y = adaptive_runge_kutta_45(
        # The numerical function to integrate takes line as input: time and 6 for cartesian state.
        fun=lambdify(
            args=[simulation_parameters.parameter_expressions[r"t"], STATE_VECTOR_LINE],
            expr=flatten(
                evaluate_terminal_parameters(
                    expression=generalized_symbolic_propagator,
                    parameter_expressions=simulation_parameters.parameter_expressions,
                    terminal_parameter_values=simulation_parameters.terminal_parameter_values,
                ),
            ),
        ),
        t_bounds=(
            0.0,
            simulation_parameters.arc_parameters.arc_length,
            simulation_parameters.arc_parameters.time_step,
        ),
        y_0=generate_numerically_initial_condition(
            simulation_parameters=simulation_parameters, initial_conditions=initial_conditions
        ),
    )

    # Times and integrated state vector at all times.
    return (
        t,
        y,
        # To be differentiated with respect to invertible parameters.
        generalized_symbolic_propagator,
    )
