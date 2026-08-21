"""
To avoid cyclical definition.
"""

from pathlib import Path

from matplotlib.image import imread
from numpy import flipud, ndarray, pi, zeros_like
from sympy import Expr

STATE_PARAMETERS = r"x_0 y_0 z_0 \dot{x}_0 \dot{y}_0 \dot{z}_0"
DEFAULT_MAX_ITERATIONS = 4
DFAULT_CONVERGENCE_THRESHOLD = 1e-2
TEST_OUTPUT_PATH = Path("tests")
TEST_INVERSION_NAME = "test_inversion"
TEST_NO_ITERATIONS_NAME = "test_no_iterations"
DEFAULT_SIMULATION_PARAMETERS_FILE_NAME = "simulation_parameters"
DEFAULT_STATIONS_FILE_NAME = "stations"
DEFAULT_SIMULATED_MEASUREMENTS_FILE_NAME = "simulated_measurements"
DEFAULT_RESIDUALS_FILE_NAME = "residuals"
DEFAULT_MEASUREMENT_DIRECTORY_NAME = "measurements"
TEST_ARC_ID = "test_arc_id"
TEST_INVERSION_PATH = TEST_OUTPUT_PATH.joinpath(TEST_INVERSION_NAME)
TEST_NO_ITERATIONS_PATH = TEST_OUTPUT_PATH.joinpath(TEST_NO_ITERATIONS_NAME)
TEST_MEASUREMENTS_PATH = TEST_NO_ITERATIONS_PATH.joinpath(DEFAULT_MEASUREMENT_DIRECTORY_NAME)
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
EARTH_IMAGE_NAME = "earth.jpg"
EARTH_RADIUS = 6.371e6
EARTH_IMAGE = flipud(imread(EARTH_IMAGE_NAME)[:, :, :3])
GRAY_EARTH_IMAGE: ndarray = EARTH_IMAGE.mean(axis=2) / 255.0
EARTH_SURFACE_COLOR = zeros_like(GRAY_EARTH_IMAGE)
DARK_PLOT_THRESHOLD, LIGHT_PLOT_THRESHOLD = 0.05, 0.75
EARTH_SURFACE_COLOR[GRAY_EARTH_IMAGE < DARK_PLOT_THRESHOLD] = 0  # Maps dark to deep blue.
EARTH_SURFACE_COLOR[GRAY_EARTH_IMAGE > LIGHT_PLOT_THRESHOLD] = 1  # Maps light to white.
EARTH_GROUND_MASK = (GRAY_EARTH_IMAGE >= DARK_PLOT_THRESHOLD) & (
    GRAY_EARTH_IMAGE <= LIGHT_PLOT_THRESHOLD
)
EARTH_SURFACE_COLOR[EARTH_GROUND_MASK] = 0.5  # Maps the rest to deep green.
EARTH_COLOR_SCALE = [[0.0, "rgb(30,59,117)"], [0.5, "rgb(46,68,21)"], [1.0, "rgb(255,255,255)"]]
DEFAULT_MU = 3.986004418e14
TEST_J2 = 1e-2
DEFAULT_TERMINAL_PARAMETER_VALUES = {
    r"R_{Earth\ radius}": EARTH_RADIUS,
    r"\mu_{gravitational\ parameter}": DEFAULT_MU,
    r"J_2": TEST_J2,
    r"a_{semi-major\ axis}": 7e6,
    r"e_{eccentricity}": 0.05,
    r"i_{inclination}": 50.0,
    r"\Omega_{right\ ascension\ ascending\ node}": 45.0,
    r"\omega_{argument\ of\ periapsis}": 0.0,
    r"\nu_{true\ anomaly}": 0.0,
    r"\theta_{arc\ start\ Earth\ rotation\ angle}": 0.0,
    r"\omega_{Earth\ rotation\ angular\ speed}": 2 * pi / 86164,
}


def radians(angle: float | Expr) -> float | Expr:
    """
    Conversion.
    """

    return pi / 180 * angle


def degrees(angle: float | Expr) -> float | Expr:
    """
    Conversion.
    """

    return 180 / pi * angle
