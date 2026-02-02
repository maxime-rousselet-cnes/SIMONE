"""
Provides base spatial dynamics formulas.
"""

from sympy import Expr, Matrix, MutableDenseMatrix, Symbol, cos, sin, sqrt

from .base_constants import radians
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

        self.semi_major_axis = Symbol(r"a_{semi-major\ axis}")
        self.eccentricity = Symbol(r"e_{eccentricity}")
        self.inclination = Symbol(r"i_{inclination}")
        self.right_ascension_ascending_node = Symbol(r"\Omega_{right\ ascension\ ascending\ node}")
        self.argument_of_periapsis = Symbol(r"\omega_{argument\ of\ periapsis}")
        self.true_anomaly = Symbol(r"\nu_{true\ anomaly}")

    def to_cartesian_state(
        self,
        gravitational_parameter: Expr,
    ) -> MutableDenseMatrix:
        """
        Converts to cartesian state vector.
        """

        rotation_orbital_plane_to_inertial = rotation_matrix(
            angle=radians(self.right_ascension_ascending_node)
        ) @ (
            rotation_matrix(
                angle=radians(self.inclination),
                unit_vector=MutableDenseMatrix([[1], [0], [0]]),
            )
            @ rotation_matrix(angle=radians(self.argument_of_periapsis))
        )
        impact_parameter = self.semi_major_axis * (1 - self.eccentricity**2)
        radius = impact_parameter / (1 + self.eccentricity * cos(radians(self.true_anomaly)))
        orbital_plane_position = MutableDenseMatrix(
            [
                [radius * cos(radians(self.true_anomaly))],
                [radius * sin(radians(self.true_anomaly))],
                [0],
            ]
        )
        velocity_proxy = sqrt(gravitational_parameter / impact_parameter)
        orbital_plane_speed = MutableDenseMatrix(
            [
                [-velocity_proxy * sin(radians(self.true_anomaly))],
                [velocity_proxy * (self.eccentricity + cos(radians(self.true_anomaly)))],
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
