"""
Describes a forward simulation of an arc for a satellite
"""

from datetime import datetime, timedelta

from numpy import array
from numpy.linalg import norm
from scipy.integrate import RK45
from sympy import Expr, Matrix, MutableDenseMatrix, Symbol, evaluate, flatten, lambdify, symbols

from .ephemeris import OrbitalParameters
from .forces import symbolic_propagator
from .parameters import ParameterSampling, SimulationParameters, TimeDependentParameter
from .test_forces import ParameterizedTestForceParameters, TimeTestForceParameters
from .utils import evaluate_terminal_parameters


def propagate_ephemeris(
    simulation_parameters: SimulationParameters,
    parameter_expressions: dict[str, Expr],
    terminal_parameter_values: dict[str, float],
    initial_conditions: OrbitalParameters,
) -> tuple[
    list[float],
    list[list[float]],
    MutableDenseMatrix,
]:
    """
    Integration of motion.
    """

    state_vector_line = list(symbols("x y z v_x v_y v_z"))

    with evaluate(False):

        generalized_symbolic_propagator: MutableDenseMatrix = symbolic_propagator(
            state_vector=Matrix(state_vector_line).T,
            parameters=parameter_expressions,
            simulation_parameters=simulation_parameters,
        )

    rk45_integrator = RK45(
        # The numerical function to integrate takes line as input: time and 6 for cartesian state.
        fun=lambdify(
            args=[symbols("t")] + [state_vector_line],
            expr=flatten(
                evaluate_terminal_parameters(
                    expression=generalized_symbolic_propagator,
                    parameter_expressions=parameter_expressions,
                    terminal_parameter_values=terminal_parameter_values,
                ),
            ),
        ),
        t0=0.0,
        # Initial conditions numerically evaluated if parameterized.
        y0=flatten(
            evaluate_terminal_parameters(
                expression=initial_conditions.to_cartesian_state(
                    gravitational_parameter=parameter_expressions["gravitational_parameter"],
                ),
                parameter_expressions=parameter_expressions,
                terminal_parameter_values=terminal_parameter_values,
            )
        ),
        t_bound=simulation_parameters.arc_length,
        max_step=simulation_parameters.time_step,
        vectorized=True,
    )
    t = []
    y = []

    while rk45_integrator.status == "running":

        t.append(rk45_integrator.t)
        y.append(rk45_integrator.y)

        # Manages surface crash.
        if norm(y[-1][:3]) <= terminal_parameter_values["Earth_radius"]:

            break

        rk45_integrator.step()

    # Times and integrated state vector at all times.
    return (
        array(object=t, dtype=float),
        array(object=y, dtype=float),
        # To be differentiated with respect to invertible parameters.
        generalized_symbolic_propagator,
    )


def test_forward_orbit():
    """
    Checks if the forward simulation of orbit determination runs for dummy forces.
    """

    arc_length = 10000.0
    time_step = 10.0
    arc_start_datetime = datetime(
        year=2000, month=1, day=1, hour=0, minute=0, second=0, microsecond=0
    )
    simulation_parameters = SimulationParameters(
        time_step=time_step,
        simulated_forces={
            "central_body_attraction": None,
            "parameterized_test_force": ParameterizedTestForceParameters(
                dummy_parameter=Symbol("dummy_parameter"), dummy_parameter_value=1e-5
            ),
            "time_test_force": TimeTestForceParameters(
                time_dependent_parameter=TimeDependentParameter(
                    symbol="time_dependent_parameter",
                    arc_start_datetime=arc_start_datetime,
                    parameter_sampling=ParameterSampling(
                        datetime_sampling_values=[
                            arc_start_datetime + timedelta(seconds=seconds)
                            for seconds in range(0, int(arc_length + 1), 1000)
                        ],
                        parameter_values=[
                            1e-5 * delta_value * time_step
                            for delta_value in range(int(arc_length // time_step) + 1)
                        ],
                    ),
                )
            ),
        },
        arc_start_datetime=arc_start_datetime,
        arc_length=arc_length,
    )
    parameter_expressions = {
        "Earth_radius": Symbol("Earth_radius"),
        "gravitational_parameter": Symbol("gravitational_parameter"),
        "t": Symbol("t"),
        "semi_major_axis": Symbol("semi_major_axis"),
        "eccentricity": Symbol("eccentricity"),
        "inclination": Symbol("inclination"),
        "right_ascension_ascending_node": Symbol("right_ascension_ascending_node"),
        "argument_of_periapsis": Symbol("argument_of_periapsis"),
        "true_anomaly": Symbol("true_anomaly"),
    }
    terminal_parameter_values = {
        "gravitational_parameter": 3.986004418e14,
        "Earth_radius": 6.371e6,
        "semi_major_axis": 7e6,
        "eccentricity": 0.05,
        "inclination": 45.0,
        "right_ascension_ascending_node": 0.0,
        "argument_of_periapsis": 0.0,
        "true_anomaly": 0.0,
    }

    for _, force_parameters in simulation_parameters.simulated_forces.items():

        if force_parameters:

            parameter_expressions.update(force_parameters.get_parameter_expressions())
            terminal_parameter_values.update(force_parameters.get_terminal_parameters())

    initial_conditions = OrbitalParameters()

    t, y, _ = propagate_ephemeris(
        simulation_parameters=simulation_parameters,
        parameter_expressions=parameter_expressions,
        terminal_parameter_values=terminal_parameter_values,
        initial_conditions=initial_conditions,
    )

    assert len(t) >= arc_length // time_step
    assert len(y) == len(t)
