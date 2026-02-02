"""
Independent utility functions.
"""

from json import JSONEncoder, dump, load
from json.decoder import JSONDecodeError
from pathlib import Path
from time import sleep
from typing import Any, Optional

from numpy import ndarray
from pydantic import BaseModel
from sympy import Expr, Identity, Matrix, MutableDenseMatrix, Piecewise, cos, sin, symbols

STATE_VECTOR_LINE = list(symbols(r"x y z v_x v_y v_z"))
STATE_VECTOR_MATRIX: MutableDenseMatrix = Matrix(STATE_VECTOR_LINE).T


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

    return (
        cos(angle) * Identity(3)
        + (1 - cos(angle)) * outer_product_matrix
        + sin(angle) * cross_product_matrix
    )


def norm(vector: MutableDenseMatrix) -> MutableDenseMatrix:
    """
    Computes the Euclidean norm of a vector.
    """

    return sum(component**2 for component in vector.flat()) ** 0.5


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


def evaluate_terminal_parameters(
    expression: Expr,
    parameter_expressions: dict[str, Expr],
    terminal_parameter_values: dict[str, float],
) -> Expr:
    """
    Substitudes terminal parameter expression into their values.
    """

    return expression.xreplace(
        rule={
            parameter_expressions[parameter_name]: value
            for parameter_name, value in terminal_parameter_values.items()
        }
    )


class JSONSerialize(JSONEncoder):
    """
    Handmade JSON encoder that correctly encodes special structures.
    """

    def default(self, o):

        if isinstance(o, ndarray):

            return o.tolist()

        if isinstance(o, BaseModel):

            return o.__dict__

        return JSONEncoder().default(o)


def save_base_model(obj: Any, name: str, path: Path):
    """
    Saves a JSON serializable type.
    """

    # Eventually considers subpath.
    while len(name.split("/")) > 1:

        path = path.joinpath(name.split("/")[0])
        name = "".join(name.split("/")[1:])

    # May create the directory.
    path.mkdir(exist_ok=True, parents=True)

    # Saves the object.
    with open(path.joinpath(name + ".json"), "w", encoding="utf-8") as file:

        dump(obj, fp=file, cls=JSONSerialize, indent=4)


def load_base_model(
    name: str,
    path: Path,
    base_model_type: Optional[Any] = None,
) -> Any:
    """
    Loads a JSON serializable type.
    """

    filepath = path.joinpath(name + ("" if ".json" in name else ".json"))

    try:

        with open(filepath, "r", encoding="utf-8") as file:

            loaded_content = load(fp=file)

    except JSONDecodeError:

        # Waits to avoid concurrent reading/writing.
        sleep(1e-3)

        # Then retries.
        return load_base_model(name=name, path=path, base_model_type=base_model_type)

    return loaded_content if not base_model_type else base_model_type(**loaded_content)
