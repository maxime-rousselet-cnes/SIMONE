"""
Provides base spatial dynamics formulas.
"""

from sympy import Expr, Matrix, MutableDenseMatrix, Symbol, cos, pi, sin, sqrt

from .utils import rotation_matrix


class OrbitalParameters:
    """
    Symbolic expressions for classical Keplerian elements.
    """

    semi_major_axis: Expr
    eccentricity: Expr
    inclination: Expr
    right_ascension_ascending_node: Expr
    argument_of_periapsis: Expr
    true_anomaly: Expr

    def __init__(self) -> None:

        self.semi_major_axis = Symbol("semi_major_axis")
        self.eccentricity = Symbol("eccentricity")
        self.inclination = Symbol("inclination")
        self.right_ascension_ascending_node = Symbol("right_ascension_ascending_node")
        self.argument_of_periapsis = Symbol("argument_of_periapsis")
        self.true_anomaly = Symbol("true_anomaly")

    def to_cartesian_state(
        self,
        gravitational_parameter: Expr,
    ) -> MutableDenseMatrix:
        """
        Converts to cartesian state vector.
        """

        rotation_orbital_plane_to_inertial = rotation_matrix(
            angle=pi / 180 * self.right_ascension_ascending_node
        ) @ (
            rotation_matrix(
                angle=pi / 180 * self.inclination,
                unit_vector=MutableDenseMatrix([[1], [0], [0]]),
            )
            @ rotation_matrix(angle=pi / 180 * self.argument_of_periapsis)
        )
        impact_parameter = self.semi_major_axis * (1 - self.eccentricity**2)
        radius = impact_parameter / (1 + self.eccentricity * cos(pi / 180 * self.true_anomaly))
        orbital_plane_position = MutableDenseMatrix(
            [
                [radius * cos(pi / 180 * self.true_anomaly)],
                [radius * sin(pi / 180 * self.true_anomaly)],
                [0],
            ]
        )
        velocity_proxy = sqrt(gravitational_parameter / impact_parameter)
        orbital_plane_speed = MutableDenseMatrix(
            [
                [-velocity_proxy * sin(pi / 180 * self.true_anomaly)],
                [velocity_proxy * (self.eccentricity + cos(pi / 180 * self.true_anomaly))],
                [0],
            ]
        )
        return Matrix.vstack(
            Matrix(rotation_orbital_plane_to_inertial @ orbital_plane_position),
            Matrix(rotation_orbital_plane_to_inertial @ orbital_plane_speed),
        )

    def periapsis(self) -> Expr:
        """
        Computes periapsis distance for minimal altitude check.
        """

        return self.semi_major_axis * (1 - self.eccentricity)
