"""
Independent utility functions.
"""

from math import atan2

from sympy import Expr, Identity, Matrix, MutableDenseMatrix, Piecewise, cos, sin, sqrt, symbols

from .base_constants import degrees, radians

STATE_VECTOR_LINE: list[Expr] = list(symbols(r"x y z \dot{x} \dot{y} \dot{z}"))
STATE_VECTOR_MATRIX: MutableDenseMatrix = Matrix(STATE_VECTOR_LINE)


def position(state_vector: MutableDenseMatrix) -> MutableDenseMatrix:
    """
    The position is the projection of the state vector onto the first three components.
    """

    return Matrix(state_vector[:3])


def speed(state_vector: MutableDenseMatrix) -> MutableDenseMatrix:
    """
    The speed is the projection of the state vector onto the three next components, after the three
    components related to position.
    """

    return Matrix(state_vector[3:6])


from sympy import atan2, cos, sin, sqrt


def simplify_atan2_trig(expr: Expr):
    """
    Takes advantages of trigonometric simplifications.
    """

    if expr.args[0].func == atan2:

        y, x = expr.args[0].args

        if expr.func == cos:

            return x / sqrt(x**2 + y**2)

        if expr.func == sin:

            return y / sqrt(x**2 + y**2)

    return expr


def rotation_matrix(
    angle: Expr, unit_vector: MutableDenseMatrix = Matrix([[0], [0], [1]])
) -> MutableDenseMatrix:
    """
    General expression of a rotation matrix on any axis using Rodrigues rotation formula. Assumes
    the given axis direction is a unit vector.
    """

    outer_product_matrix = Matrix(
        [[term_1 * term_2 for term_2 in unit_vector.flat()] for term_1 in unit_vector.flat()]
    )
    cross_product_matrix = Matrix(
        [
            [0, -unit_vector[2], unit_vector[1]],
            [unit_vector[2], 0, -unit_vector[0]],
            [-unit_vector[1], unit_vector[0], 0],
        ]
    )
    cos_angle: Expr = simplify_atan2_trig(expr=cos(angle))

    return (
        cos_angle * Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        + (1 - cos_angle) * outer_product_matrix
        + simplify_atan2_trig(expr=sin(angle)) * cross_product_matrix
    )


def norm(vector: MutableDenseMatrix) -> MutableDenseMatrix:
    """
    Computes the Euclidean norm of a vector.
    """

    return sqrt(sum(component**2 for component in vector.flat()))


def distance(vector_1: MutableDenseMatrix, vector_2: MutableDenseMatrix) -> Expr:
    """
    Euclidian distance.
    """

    return norm(vector=position(state_vector=vector_2 - vector_1))


def lagrange_polynomial_interpolation(t: Expr, t_points: list[Expr], y_points: list[Expr]) -> Expr:
    """
    Builds the k-th order Lagrange polynomial symbolically.
    t_points, y_points: lists of length k + 1.
    """

    l = 0

    for i, _ in enumerate(t_points):

        term = y_points[i]

        for j, _ in enumerate(t_points):

            if j != i:

                term *= (t - t_points[j]) / (t_points[i] - t_points[j])

        l += term

    return l


def piecewise_lagrange(t: Expr, t_syms: list[Expr], y_syms: list[Expr], order: int):
    """
    General symbolic piecewise Lagrange interpolator. Interpolates the given (t_syms, y_syms) data
    at time t.
    """

    n = len(t_syms)
    pieces = []

    for i in range(n):

        half_order = order // 2
        left = max(0, i - half_order)
        right = left + order + 1

        if right > n:

            right = n
            left = max(0, right - (order + 1))

        t_slice = t_syms[left:right]
        y_slice = y_syms[left:right]
        poly = lagrange_polynomial_interpolation(t, t_slice, y_slice)
        condition = True if i == n - 1 else t < t_syms[i + 1]
        pieces.append((poly, condition))

    return Piecewise(*pieces)


def geographic_coordinates_from_cartesian(r: list[float]) -> tuple[float, float]:
    """
    Assumes spehrical Earth.
    """

    return degrees(angle=atan2(r[2], (r[0] ** 2 + r[1] ** 2) ** 0.5)), degrees(
        angle=atan2(r[1], r[0])
    )


def ecef_position(
    latitude: Expr, longitude: Expr, altitude: Expr, parameter_expressions: dict[str, Expr]
) -> MutableDenseMatrix:
    """
    Nominal ECEF position assuming spherical Earth.
    """

    return MutableDenseMatrix(
        [
            (parameter_expressions[r"R_{Earth\ radius}"] + altitude)
            * cos(radians(angle=latitude))
            * cos(radians(angle=longitude)),
            (parameter_expressions[r"R_{Earth\ radius}"] + altitude)
            * cos(radians(angle=latitude))
            * sin(radians(angle=longitude)),
            (parameter_expressions[r"R_{Earth\ radius}"] + altitude) * sin(radians(angle=latitude)),
        ]
    )
