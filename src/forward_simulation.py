"""
Describes a forward simulation of an arc for a satellite
"""

from numpy import array
from numpy.linalg import norm
from scipy.integrate import RK45
from sympy import MutableDenseMatrix, evaluate, flatten, lambdify, symbols

from .dynamics import symbolic_propagator
from .ephemeris import OrbitalParameters
from .parameters import SimulationParameters
from .test_constants import TEST_ARC_LENGTH, TEST_SIMULATION_PARAMETERS, TEST_TIME_STEP
from .utils import STATE_VECTOR_LINE, STATE_VECTOR_MATRIX, evaluate_terminal_parameters


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

    rk45_integrator = RK45(
        # The numerical function to integrate takes line as input: time and 6 for cartesian state.
        fun=lambdify(
            args=[symbols("t")] + [STATE_VECTOR_LINE],
            expr=flatten(
                evaluate_terminal_parameters(
                    expression=generalized_symbolic_propagator,
                    parameter_expressions=simulation_parameters.parameter_expressions,
                    terminal_parameter_values=simulation_parameters.terminal_parameter_values,
                ),
            ),
        ),
        t0=0.0,
        # Initial conditions numerically evaluated if parameterized.
        y0=flatten(
            evaluate_terminal_parameters(
                expression=initial_conditions.to_cartesian_state(
                    gravitational_parameter=simulation_parameters.parameter_expressions[
                        "gravitational_parameter"
                    ],
                ),
                parameter_expressions=simulation_parameters.parameter_expressions,
                terminal_parameter_values=simulation_parameters.terminal_parameter_values,
            )
        ),
        t_bound=simulation_parameters.arc_parameters.arc_length,
        max_step=simulation_parameters.arc_parameters.time_step,
        vectorized=True,
    )
    t = []
    y = []

    while rk45_integrator.status == "running":

        t.append(rk45_integrator.t)
        y.append(rk45_integrator.y)

        # Manages surface crash.
        if norm(y[-1][:3]) <= simulation_parameters.terminal_parameter_values["Earth_radius"]:

            break

        rk45_integrator.step()

    # Times and integrated state vector at all times.
    return (
        array(object=t, dtype=float),
        array(object=y, dtype=float),
        # To be differentiated with respect to invertible parameters.
        generalized_symbolic_propagator,
    )


def test_forward_simulation(
    simulation_parameters: SimulationParameters = TEST_SIMULATION_PARAMETERS,
) -> None:
    """
    Checks if the forward simulation of orbit determination runs for dummy forces.
    """

    t, y, _ = propagate_ephemeris(
        simulation_parameters=simulation_parameters,
    )

    assert len(t) >= TEST_ARC_LENGTH // TEST_TIME_STEP
    assert len(y) == len(t)
