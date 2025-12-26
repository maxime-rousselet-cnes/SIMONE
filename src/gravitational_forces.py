from sympy import Expr, Matrix, MutableDenseMatrix

from utils import norm, position


def central_body_attraction(
    state_vector: MutableDenseMatrix, parameters: dict[str, Expr]
) -> MutableDenseMatrix:
    """
    Computes the acceleration due to the central body's gravitational attraction.
    """

    return (
        -parameters["gravitational_parameter"]
        / norm(vector=position(state_vector=state_vector)) ** 3
        * Matrix(position(state_vector=state_vector))
    )
