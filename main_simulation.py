import numpy
from scipy.integrate import RK45
from sympy import Expr, Matrix, MutableDenseMatrix, evaluate, lambdify, symbols

from src import SimulationParameters, symbolic_propagator


def propagate_ephemeris(
    simulation_parameters: SimulationParameters,
    parameters: dict[str, Expr],
    terminal_parameter_values: dict[str, float],
):
    """
    Integration of motion.
    """

    state_vector = Matrix([[symbol] for symbol in symbols("x y z v_x v_y v_z")])
    t = symbols("t")

    with evaluate(False):

        ephemeris_flow_expression: MutableDenseMatrix = symbolic_propagator(
            simulation_parameters=simulation_parameters
        )

    lambda_propagator = lambdify(
        args=list(parameter for parameter in state_parameters.flat()) + [t],
        expr=ephemeris_flow_expression.xreplace(
            rule={
                parameters[parameter_name]: value
                for parameter_name, value in terminal_parameter_values.items()
            }
            | {"delta_t": delta_t}
        ),
    )

    rk45_integrator = RK45(
        fun=lambda_propagator,
        t0=0.0,
        y0=y0,
        t_bound=integration_parameters["arc_duration"],
        max_step=integration_parameters["max_step"],
        rtol=[EPSILON] * 3 + [INF] * (3 + len(dr_dgamma_0)),
        atol=[integration_parameters["dR_tol_max"]] * 3 + [INF] * (3 + len(dr_dgamma_0)),
        vectorized=True,
        first_step=EPSILON,
    )
    t = []
    y = []

    # Integrates for the arc duration or until the altitude limit is reached.
    while rk45_integrator.status == "running":
        t.append(rk45_integrator.t)
        y.append(rk45_integrator.y)
        if (integration_parameters["altitude_limit"] >= 0) and (
            norm(R=y[-1]) < parameters["R_T"] + integration_parameters["altitude_limit"]
        ):
            break
        rk45_integrator.step()

    # Times and integrated vector containing position, speed and eventually dr_dgamma and dr_dot_dgamma.
    return t, y
