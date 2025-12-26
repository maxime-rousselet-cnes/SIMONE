from typing import Callable

from scipy.integrate import RK45
from sympy import Expr, MutableDenseMatrix, evaluate, lambdify, symbols

from src import (
    OrbitalParameters,
    SimulationParameters,
    orbital_parameters_to_cartesian_state,
    symbolic_propagator,
)


def propagate_ephemeris(
    # TODO: Generate all Force Parameter instances and call their get_terminal_parameters.
    # TODO: Add Force Parameter symbols to parameter_expressions.
    simulation_parameters: SimulationParameters,
    parameter_expressions: dict[str, Expr],
    terminal_parameter_values: dict[str, float],
    initial_conditions: OrbitalParameters,
) -> tuple[
    list[float],
    list[list[float]],
    Callable[[MutableDenseMatrix, dict[str, Expr]], MutableDenseMatrix],
]:
    """
    Integration of motion.
    """

    with evaluate(False):

        generalized_symbolic_propagator: MutableDenseMatrix = symbolic_propagator(
            simulation_parameters=simulation_parameters
        )

    rk45_integrator = RK45(
        fun=lambdify(
            args=list(symbols("x y z v_x v_y v_z")) + [symbols("t")],
            expr=generalized_symbolic_propagator.xreplace(
                rule={
                    parameter_expressions[parameter_name]: value
                    for parameter_name, value in terminal_parameter_values.items()
                }
            ),
        ),
        t0=0.0,
        y0=orbital_parameters_to_cartesian_state(
            orbital_parameters=initial_conditions,
            gravitational_parameter=parameter_expressions["gravitational_parameter"],
        ),
        t_bound=simulation_parameters.arc_length,
        max_step=simulation_parameters.time_step,
        # TODO: Vector atol/rtol.
        vectorized=True,
    )
    t = []
    y = []

    # Integrates for the arc duration or until the altitude limit is reached.
    while rk45_integrator.status == "running":

        t.append(rk45_integrator.t)
        y.append(rk45_integrator.y)

        # TODO: Manage surface crash.
        if False:

            break

        rk45_integrator.step()

    # Times and integrated state vector at all times.
    return t, y
