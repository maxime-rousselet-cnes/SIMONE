"""
To avoid cyclical definition.
"""

from pathlib import Path

from matplotlib.image import imread
from numpy import flipud, ndarray, pi, zeros_like
from sympy import Expr

DEFAULT_MAX_ITERATIONS = 5
DFAULT_CONVERGENCE_THRESHOLD = 1e-2
TEST_OUTPUT_PATH = Path("TESTS")
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
    r"i_{inclination}": 70.0,
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
