"""
Describes all Newtonian accelerations based on gravity.
"""

from sympy import Expr, Matrix, MutableDenseMatrix

from .utils import norm, position


def central_body_attraction(
    state_vector: MutableDenseMatrix, parameter_expressions: dict[str, Expr]
) -> MutableDenseMatrix:
    """
    Computes the acceleration due to the central body's gravitational attraction.
    """

    return (
        -parameter_expressions[r"\mu_{gravitational\ parameter}"]
        / norm(vector=position(state_vector=state_vector)) ** 3
        * Matrix(position(state_vector=state_vector))
    )


def j2_attraction(
    state_vector: MutableDenseMatrix, parameter_expressions: dict[str, Expr]
) -> MutableDenseMatrix:
    """
    Effect of the Earth's flattening on gravity.
    """

    r = position(state_vector=state_vector)
    z = r[2]
    expr: Expr = 5 * z**2 / norm(vector=r) ** 2 - 1

    return (
        3
        / 2
        * parameter_expressions[r"\mu_{gravitational\ parameter}"]
        * parameter_expressions[r"R_{Earth\ radius}"] ** 2
        * parameter_expressions[r"J_2"]
        * MutableDenseMatrix([[r[0] * expr], [r[1] * expr], [z * (expr - 2)]])
        / norm(vector=r) ** 5
    )
