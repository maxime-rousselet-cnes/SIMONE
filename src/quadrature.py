"""
Integrates the partial derivatives.
"""

from .forward_simulation import propagate_ephemeris
from .parameters import SimulationParameters
from .test_constants import TEST_SIMULATION_PARAMETERS


def test_quadrature(
    simulation_parameters: SimulationParameters = TEST_SIMULATION_PARAMETERS,
) -> None:
    """
    Verifies if the measurements are correctly created in a forward simulation.
    """

    t, y, generalized_symbolic_propagator = propagate_ephemeris(
        simulation_parameters=simulation_parameters,
    )

    # TODO.
    """
    station_theoretical_measurements, _ = generate_measurements(
        t=t,
        y=y,
        stations=TEST_STATIONS,
        observation_timestamps=observation_timestamps,
        simulation_parameters=simulation_parameters,
    )
    """
