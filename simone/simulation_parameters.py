"""
Defines the class needed to describe a simulation globally. Built from Parameter classes.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from base_models import load_base_model, save_base_model
from sympy import Expr, MutableDenseMatrix, Symbol, simplify, srepr

from .base_constants import (
    DATETIME_FORMAT,
    DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
    DEFAULT_TERMINAL_PARAMETER_VALUES,
    TEST_NO_ITERATIONS_PATH,
)
from .gravitational_forces import central_body_attraction, j2_attraction
from .parameters import ArcParameters, Parameters
from .test_forces import (
    ParameterizedTestForceParameters,
    TimeTestForceParameters,
    parameterized_test_force,
    time_test_force,
)

ALL_FORCES = {
    "central_body_attraction": central_body_attraction,
    "j2_attraction": j2_attraction,
    "parameterized_test_force": parameterized_test_force,
    "time_test_force": time_test_force,
}
ALL_FORCES_PARAMETERS = {
    "central_body_attraction": None,
    "j2_attraction": None,
    "parameterized_test_force": ParameterizedTestForceParameters,
    "time_test_force": TimeTestForceParameters,
}


class SimulationParameters:
    """
    All parameters required for a forward simulation.
    """

    arc_parameters: ArcParameters
    lagrange_kernels: MutableDenseMatrix
    simulated_forces: dict[str, Optional[Parameters]]
    parameter_expressions: dict[str, Expr]
    terminal_parameter_values: dict[str, float]

    def __init__(
        self,
        arc_parameters: ArcParameters,
        simulated_forces: dict[str, Optional[Parameters]],
        parameter_expressions: Optional[dict[str, Expr | str]] = None,
        terminal_parameter_values: Optional[dict[str, float]] = None,
    ) -> None:

        self.arc_parameters = arc_parameters
        self.simulated_forces = simulated_forces
        self.terminal_parameter_values = DEFAULT_TERMINAL_PARAMETER_VALUES | (
            terminal_parameter_values if terminal_parameter_values else {}
        )
        self.parameter_expressions = (
            {r"t": Symbol(r"t")}
            | {
                parameter_name: Symbol(parameter_name)
                for parameter_name in self.terminal_parameter_values
            }
            | (
                {}
                if not parameter_expressions
                else {
                    parameter_name: simplify(expr=expression)
                    for parameter_name, expression in parameter_expressions.items()
                }
            )
        )
        self.lagrange_kernels = self.arc_parameters.generate_lagrange_kernels(
            t=self.parameter_expressions[r"t"]
        )

        for _, force_parameters in simulated_forces.items():

            if force_parameters:

                self.update_expressions(
                    new_expressions=force_parameters.get_parameter_expressions()
                )
                self.update_terminal_parameter_values(
                    new_expressions=force_parameters.get_terminal_parameters()
                )

    def update_expressions(self, new_expressions: dict[str, Expr]) -> None:
        """
        Updates the dictionary of parameter expressions. Need to update the dictionary of terminal
        parameter values too.
        """

        self.parameter_expressions.update(new_expressions)

    def update_terminal_parameter_values(self, new_expressions: dict[str, Expr]) -> None:
        """
        Updates the dictionary of terminal parameter values. Need to update the dictionary of
        parameter expressions too.
        """

        self.terminal_parameter_values.update(new_expressions)

    def update_for_stations(self, stations: dict[str, Parameters]) -> None:
        """
        Input is a dict of stations. Defined here as Parameters (from which Station inherits) to
        avoid circular import.
        """

        for _, station in stations.items():

            self.update_expressions(new_expressions=station.get_parameter_expressions())
            self.update_terminal_parameter_values(new_expressions=station.get_terminal_parameters())

    def save(
        self,
        path: Path = TEST_NO_ITERATIONS_PATH,
        name: str = DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
    ) -> None:
        """
        Saves in (.JSON) file.
        """

        arc_parameters = self.arc_parameters
        save_base_model(
            obj={
                "arc_parameters": {
                    "time_step": arc_parameters.time_step,
                    "arc_start_datetime": arc_parameters.arc_start_datetime.strftime(
                        format=DATETIME_FORMAT
                    ),
                    "arc_length": arc_parameters.arc_length,
                    "lagrange_interpolation_order": arc_parameters.lagrange_interpolation_order,
                    "arc_id": arc_parameters.arc_id,
                    "is_initial": arc_parameters.is_initial,
                },
                "simulated_forces": {
                    force_id: None if not force else force.to_serializable()
                    for force_id, force in self.simulated_forces.items()
                },
                "parameter_expressions": (
                    None
                    if not self.parameter_expressions
                    else {
                        parameter_name: srepr(expression)
                        for parameter_name, expression in self.parameter_expressions.items()
                    }
                ),
                "terminal_parameter_values": self.terminal_parameter_values,
            },
            name=name,
            path=path,
        )


def load_simulation_parameters(
    path: Path = TEST_NO_ITERATIONS_PATH,
    name: str = DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
) -> SimulationParameters:
    """
    From (.JSON) file.
    """

    loaded_dict = load_base_model(name=name + ".json", path=path)
    simulated_forces: dict = loaded_dict["simulated_forces"]

    return SimulationParameters(
        arc_parameters=ArcParameters(
            time_step=loaded_dict["arc_parameters"]["time_step"],
            arc_start_datetime=datetime.strptime(
                loaded_dict["arc_parameters"]["arc_start_datetime"], DATETIME_FORMAT
            ),
            arc_length=loaded_dict["arc_parameters"]["arc_length"],
            lagrange_interpolation_order=loaded_dict["arc_parameters"][
                "lagrange_interpolation_order"
            ],
            arc_id=loaded_dict["arc_parameters"]["arc_id"],
            is_initial=loaded_dict["arc_parameters"]["is_initial"],
        ),
        simulated_forces={
            force_name: (
                None
                if not ALL_FORCES_PARAMETERS[force_name]
                else ALL_FORCES_PARAMETERS[force_name](**force_parameters)
            )
            for force_name, force_parameters in simulated_forces.items()
        },
        parameter_expressions=loaded_dict["parameter_expressions"],
        terminal_parameter_values=loaded_dict["terminal_parameter_values"],
    )
