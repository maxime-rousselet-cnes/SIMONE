"""
Regroups all forces defined in the different force modules.
"""

from sympy import Expr, Matrix, MutableDenseMatrix

from .simulation_parameters import ALL_FORCES, SimulationParameters
from .utils import rotation_matrix, speed


def ecef_to_eci(parameter_expressions: dict[str, Expr]) -> MutableDenseMatrix:
    """
    Returns the rotation matrix from Earth-fixed frame to intertial frame.
    """

    return rotation_matrix(
        angle=parameter_expressions[r"\theta_{arc\ start\ Earth\ rotation\ angle}"]
        + parameter_expressions[r"\omega_{Earth\ rotation\ angular\ speed}"]
        * parameter_expressions[r"t"]
    )


def eci_to_ecef(parameter_expressions: dict[str, Expr]) -> MutableDenseMatrix:
    """
    Returns the rotation matrix from intertial frame to Earth-fixed frame.
    """

    return ecef_to_eci(parameter_expressions=parameter_expressions).T


def symbolic_propagator(
    state_vector: MutableDenseMatrix,
    simulation_parameters: SimulationParameters,
) -> MutableDenseMatrix:
    """
    Builds a simulation-specific symbolic propagator. Yet to be evaluated for parameter expressions
    before being integrated on state and time.
    """

    return Matrix.vstack(
        speed(state_vector=state_vector),
        sum(
            (
                (
                    ALL_FORCES[force_name](
                        state_vector, simulation_parameters.parameter_expressions, force_parameters
                    )
                    if force_parameters
                    else ALL_FORCES[force_name](
                        state_vector, simulation_parameters.parameter_expressions
                    )
                )
                for force_name, force_parameters in simulation_parameters.simulated_forces.items()
            ),
            start=MutableDenseMatrix.zeros(rows=3, cols=1),
        ),
    )
