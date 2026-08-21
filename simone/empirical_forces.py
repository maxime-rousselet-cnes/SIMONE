"""
Empirical accelerations to eventually absorb model errors.
"""

from sympy import Expr, Matrix, MutableDenseMatrix, atan2, cos, simplify, sin, sqrt, srepr
from sympy.vector import dot

from .parameters import Parameters
from .utils import norm, position, rotation_matrix, speed


class EmpiricalForceParameters(Parameters):
    """
    Models constant bias and once per revolution sinusoidal accelerations in every direction of the
    orbital frame.
    """

    b_r: Expr
    b_t: Expr
    b_b: Expr
    c_r: Expr
    c_t: Expr
    c_b: Expr
    s_r: Expr
    s_t: Expr
    s_b: Expr
    b_r_value: float
    b_t_value: float
    b_n_value: float
    c_r_value: float
    c_t_value: float
    c_n_value: float
    s_r_value: float
    s_t_value: float
    s_n_value: float

    def __init__(
        self,
        b_parameters: list[Expr | str],
        periodic_parameters: list[Expr | str],
        b_parameter_values: list[float] = [0.0, 0.0, 0.0],
        periodic_parameter_values: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ) -> None:

        self.b_r = simplify(expr=b_parameters[0])
        self.b_t = simplify(expr=b_parameters[1])
        self.b_n = simplify(expr=b_parameters[2])

        self.c_r = simplify(expr=periodic_parameters[0])
        self.c_t = simplify(expr=periodic_parameters[1])
        self.c_n = simplify(expr=periodic_parameters[2])
        self.s_r = simplify(expr=periodic_parameters[3])
        self.s_t = simplify(expr=periodic_parameters[4])
        self.s_n = simplify(expr=periodic_parameters[5])

        self.b_r_value = b_parameter_values[0]
        self.b_t_value = b_parameter_values[1]
        self.b_n_value = b_parameter_values[2]

        self.c_r_value = periodic_parameter_values[0]
        self.c_t_value = periodic_parameter_values[1]
        self.c_n_value = periodic_parameter_values[2]
        self.s_r_value = periodic_parameter_values[3]
        self.s_t_value = periodic_parameter_values[4]
        self.s_n_value = periodic_parameter_values[5]

    def get_terminal_parameters(self) -> dict[str, float]:
        """
        Straightforward definition.
        """

        return {
            r"b_r": self.b_r_value,
            r"b_t": self.b_t_value,
            r"b_n": self.b_n_value,
            r"c_r": self.c_r_value,
            r"c_t": self.c_t_value,
            r"c_n": self.c_n_value,
            r"s_r": self.s_r_value,
            r"s_t": self.s_t_value,
            r"s_n": self.s_n_value,
        }

    def get_parameter_expressions(self) -> dict[str, Expr]:
        """
        Straightforward definition.
        """

        return {
            r"b_r": self.b_r,
            r"b_t": self.b_t,
            r"b_n": self.b_n,
            r"c_r": self.c_r,
            r"c_t": self.c_t,
            r"c_n": self.c_n,
            r"s_r": self.s_r,
            r"s_t": self.s_t,
            r"s_n": self.s_n,
        }

    def to_serializable(self) -> dict[str, float | str]:
        """
        To (.JSON) files.
        """

        return {
            "b_parameters": [srepr(self.b_r), srepr(self.b_t), srepr(self.b_n)],
            "periodic_parameters": [
                srepr(self.c_r),
                srepr(self.c_t),
                srepr(self.c_n),
                srepr(self.s_r),
                srepr(self.s_t),
                srepr(self.s_n),
            ],
            "b_parameter_values": [self.b_r_value, self.b_t_value, self.b_n_value],
            "periodic_parameter_values": [
                self.c_r_value,
                self.c_t_value,
                self.c_n_value,
                self.s_r_value,
                self.s_t_value,
                self.s_n_value,
            ],
        }


def empirical_force(
    state_vector: MutableDenseMatrix,
    parameter_expressions: dict[str, Expr],
    empirical_force_parameters: EmpiricalForceParameters,
) -> MutableDenseMatrix:
    """
    A constant bias and a once per revolution periodic term in every direction of the orbital frame.
    """

    a, i, Omega, omega, nu = orbit_angles_from_cartesian(
        state_vector=state_vector, parameter_expressions=parameter_expressions
    )
    two_pi_over_orbital_period = sqrt(
        parameter_expressions[r"\mu_{gravitational\ parameter}"] / a**3
    )

    # RTN to XYZ rotations.

    return Matrix(
        rotation_matrix(angle=Omega)
        @ rotation_matrix(angle=i, unit_vector=MutableDenseMatrix([[1.0], [0.0], [0.0]]))
        @ rotation_matrix(angle=omega)
        @ rotation_matrix(angle=nu)
        @ Matrix(
            [
                [
                    empirical_force_parameters.b_r
                    + empirical_force_parameters.c_r
                    * cos(parameter_expressions[r"t"] * two_pi_over_orbital_period)
                    + empirical_force_parameters.s_r
                    * sin(parameter_expressions[r"t"] * two_pi_over_orbital_period)
                ],
                [
                    empirical_force_parameters.b_t
                    + empirical_force_parameters.c_t
                    * cos(parameter_expressions[r"t"] * two_pi_over_orbital_period)
                    + empirical_force_parameters.s_t
                    * sin(parameter_expressions[r"t"] * two_pi_over_orbital_period)
                ],
                [
                    empirical_force_parameters.b_n
                    + empirical_force_parameters.c_n
                    * cos(parameter_expressions[r"t"] * two_pi_over_orbital_period)
                    + empirical_force_parameters.s_n
                    * sin(parameter_expressions[r"t"] * two_pi_over_orbital_period)
                ],
            ]
        )
    )


def orbit_angles_from_cartesian(
    state_vector: MutableDenseMatrix, parameter_expressions: dict[str, Expr]
) -> tuple[Expr, Expr, Expr, Expr, Expr]:
    """
    Gets the angular orbital parameters in radians from the cartesian state vector.
    """

    r_vector: MutableDenseMatrix = position(state_vector=state_vector)
    r = norm(vector=r_vector)
    v_vector: MutableDenseMatrix = speed(state_vector=state_vector)
    v = norm(vector=v_vector)
    h_vector: MutableDenseMatrix = r_vector.cross(v_vector)
    h = norm(vector=h_vector)
    n_vector: MutableDenseMatrix = MutableDenseMatrix([[0.0], [0.0], [1.0]]).cross(h_vector)
    n = norm(vector=n_vector)
    e_vector: MutableDenseMatrix = (
        v_vector.cross(h_vector) / parameter_expressions[r"\mu_{gravitational\ parameter}"]
        - r_vector / r
    )
    e = norm(vector=e_vector)
    a = 1 / (2 / r - v**2 / parameter_expressions[r"\mu_{gravitational\ parameter}"])
    i: Expr = atan2(sqrt(h_vector[0, 0] ** 2 + h_vector[1, 0] ** 2), h_vector[2, 0])
    Omega: Expr = atan2(n_vector[1, 0], n_vector[0, 0])
    omega: Expr = atan2(
        Matrix(n_vector.cross(e_vector)).dot(h_vector) / (n * e * h),
        n_vector.dot(e_vector) / (n * e),
    )
    nu: Expr = atan2(
        Matrix(e_vector.cross(r_vector)).dot(h_vector) / (r * e * h),
        e_vector.dot(r_vector) / (r * e),
    )

    return (
        a,
        i,
        Omega,
        omega,
        nu,
    )
