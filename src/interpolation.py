"""
Independent utility functions.
"""

from numpy import ndarray
from sympy import Expr, Symbol

from .simulation_parameters import SimulationParameters
from .utils import evaluate_terminal_parameters


def apply_lagrange_kernel(
    expression: Expr,
    simulation_parameters: SimulationParameters,
    t: ndarray[float],
    y: ndarray[float],
    time_index: int,
) -> Expr:
    """
    Evaluates an expression containing a Lagrange kernel.
    """

    return evaluate_terminal_parameters(
        expression=expression,
        parameter_expressions=simulation_parameters.parameter_expressions,
        terminal_parameter_values=simulation_parameters.terminal_parameter_values,
    ).xreplace(
        rule={
            Symbol(rf"t_{i}"): t_i
            for i, t_i in enumerate(
                t[
                    time_index
                    - simulation_parameters.arc_parameters.lagrange_interpolation_order : time_index
                    + simulation_parameters.arc_parameters.lagrange_interpolation_order
                ]
            )
        }
        | {
            Symbol(rf"\theta^{j}_{i}"): y_i
            for j in range(6)
            for i, y_i in enumerate(
                y[
                    time_index
                    - simulation_parameters.arc_parameters.lagrange_interpolation_order : time_index
                    + simulation_parameters.arc_parameters.lagrange_interpolation_order,
                    j,
                ]
            )
        }
    )
