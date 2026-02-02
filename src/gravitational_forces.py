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
