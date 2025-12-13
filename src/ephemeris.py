"""
Provides base spatial dynamics formulas.
"""

from sympy import Expr, Matrix, MutableDenseMatrix, cos, sin, sqrt

from .utils import rotation_matrix


class OrbitalParameters:

    semi_major_axis: Expr
    eccentricity: Expr
    inclination: Expr
    right_ascension_ascending_node: Expr
    argument_of_periapsis: Expr
    true_anomaly: Expr


def orbital_parameters_to_cartesian_state(
    orbital_parameters: OrbitalParameters,
    gravitational_parameter: Expr,
) -> MutableDenseMatrix:
    """ """

    rotation_orbital_plane_to_inertial = rotation_matrix(
        angle=orbital_parameters.right_ascension_ascending_node
    ) @ (
        rotation_matrix(
            angle=orbital_parameters.inclination,
            unit_vector=MutableDenseMatrix([[1.0], [0.0], [0.0]]),
        )
        @ rotation_matrix(angle=orbital_parameters.argument_of_periapsis)
    )
    impact_parameter = orbital_parameters.semi_major_axis * (1 - orbital_parameters.eccentricity**2)
    radius = impact_parameter / (
        1 + orbital_parameters.eccentricity * cos(orbital_parameters.true_anomaly)
    )
    orbital_plane_position = MutableDenseMatrix(
        [
            [radius * cos(orbital_parameters.true_anomaly)],
            [radius * sin(orbital_parameters.true_anomaly)],
            [0.0],
        ]
    )
    velocity_proxy = sqrt(gravitational_parameter / impact_parameter)
    orbital_plane_speed = MutableDenseMatrix(
        [
            [-velocity_proxy * sin(orbital_parameters.true_anomaly)],
            [
                velocity_proxy
                * (orbital_parameters.eccentricity + cos(orbital_parameters.true_anomaly))
            ],
            [0.0],
        ]
    )
    return Matrix.vstack(
        rotation_orbital_plane_to_inertial @ orbital_plane_position,
        rotation_orbital_plane_to_inertial @ orbital_plane_speed,
    )
