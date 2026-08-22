"""
All base functionalities. To test via pytest test.py.
"""

from copy import deepcopy
from datetime import datetime, timedelta
from os import remove
from pathlib import Path
from shutil import rmtree
from typing import Optional

from numpy import arange, array, mean, ndarray
from pandas import DataFrame
from sympy import Symbol

from simone import (
    DEFAULT_MEASUREMENT_DIRECTORY_NAME,
    DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
    DEFAULT_STATIONS_FILE_NAME,
    STATE_PARAMETERS,
    TEST_ARC_ID,
    TEST_INVERSION_PATH,
    TEST_NO_ITERATIONS_PATH,
    TEST_OUTPUT_PATH,
    ArcOutput,
    ArcParameters,
    ParameterizedTestForceParameters,
    Parameters,
    SimulationParameters,
    Station,
    StationPosition,
    TimeTestForceParameters,
    generate_measurements,
    generate_numerically_initial_condition,
    generate_time_dependent_parameter,
    geographic_coordinates_from_cartesian,
    get_measurements,
    get_stations,
    load_arc_output,
    load_simulation_parameters,
    propagate_ephemeris,
    run_single_arc,
    simulate_measurements,
    solve_precise_orbit_determination,
)

TEST_STATION_QUANTITY = 30
TEST_SIGMA_SAFETY_FACTOR = 10
TEST_ARC_LENGTH = 5000.0  # 10000.0
TEST_TIME_STEP = 60.0
TEST_TIME_STEP_FOR_TEST_TIME_PARAMETER = 1000.0
TEST_ARC_START_DATETIME = datetime(
    year=2000, month=1, day=1, hour=0, minute=0, second=0, microsecond=0
)
TEST_ARC_PARAMETERS = ArcParameters(
    time_step=TEST_TIME_STEP,
    arc_start_datetime=TEST_ARC_START_DATETIME,
    arc_length=TEST_ARC_LENGTH,
    lagrange_interpolation_order=4,
    arc_id=TEST_ARC_ID,
    is_initial=True,
)
TIME_TEST_FORCE_PARAMETER_FACTOR = 1e-5
TEST_SIMULATED_FORCES: dict[str, Optional[Parameters]] = {
    "central_body_attraction": None,
    "j2_attraction": None,
    "parameterized_test_force": ParameterizedTestForceParameters(
        dummy_parameter=Symbol(r"p_{dummy\ parameter}"), dummy_parameter_value=1e-5
    ),
    "time_test_force": TimeTestForceParameters(
        time_dependent_parameter=generate_time_dependent_parameter(
            symbol=r"p_{time\ dependent\ parameter}{}",
            arc_start_datetime=TEST_ARC_START_DATETIME,
            datetime_sampling_values=[
                TEST_ARC_START_DATETIME + timedelta(seconds=seconds)
                for seconds in arange(
                    start=0.0,
                    stop=TEST_ARC_LENGTH + TEST_TIME_STEP_FOR_TEST_TIME_PARAMETER,
                    step=TEST_TIME_STEP_FOR_TEST_TIME_PARAMETER,
                )
            ],
            parameter_values=[
                TIME_TEST_FORCE_PARAMETER_FACTOR * seconds
                for seconds in arange(
                    start=0.0,
                    stop=TEST_ARC_LENGTH + TEST_TIME_STEP_FOR_TEST_TIME_PARAMETER,
                    step=TEST_TIME_STEP_FOR_TEST_TIME_PARAMETER,
                )
            ],
        ),
    ),
}
NUMERICAL_TOLERANCE = 1e-9


def test_clear_test_folder(path: Path = TEST_OUTPUT_PATH) -> None:
    """
    Deletes the test folder before all tests.
    """

    if path.exists():

        rmtree(path)


def test_generate_simulation_parameters(
    path: Path = TEST_NO_ITERATIONS_PATH,
    simulation_parameters_file_name: str = DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
    arc_parameters: ArcParameters = TEST_ARC_PARAMETERS,
    simulated_forces: dict[str, Optional[Parameters]] = TEST_SIMULATED_FORCES,
) -> None:
    """
    Generates simulation parameters and save in (.JSON) file to be loaded by other test functions.
    """

    SimulationParameters(
        arc_parameters=arc_parameters,
        simulated_forces=simulated_forces,
    ).save(
        path=path,
        name=simulation_parameters_file_name,
    )
    assert path.joinpath(simulation_parameters_file_name + ".json").exists()


def test_generate_stations(
    output_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = DEFAULT_STATIONS_FILE_NAME,
    # Ensures a virtual station at least has the satellite in visibility at the arc's beginning.
    deterministic_station: bool = True,
    simulation_parameters_path: Path = TEST_NO_ITERATIONS_PATH,
    simulation_parameters_file_name: str = DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
) -> None:
    """
    Generates on-continents virtual stations for simulation/test purposes and saves a corresponding
    (.CSV) file.
    """

    data = {
        "station_id": [],
        "latitude": [],
        "longitude": [],
        "altitude": [],
        "eastward_speed": [],
        "northward_speed": [],
        "vertical_speed": [],
        "range_bias": [],
        "minimal_elevation_angle": [],
    }
    stations = {
        r"station\ test\ id\ "
        + str(k): Station(
            name=r"station\ test\ id\ " + str(k),
            station_position=StationPosition(),
        )
        for k in range(TEST_STATION_QUANTITY)
    }

    if deterministic_station:

        y_0 = generate_numerically_initial_condition(
            simulation_parameters=load_simulation_parameters(
                path=simulation_parameters_path,
                name=simulation_parameters_file_name,
            )
        )
        deterministic_station_latitude, deterministic_station_longitude = (
            geographic_coordinates_from_cartesian(r=y_0[:3])
        )
        stations |= {
            r"station\ deterministic": Station(
                name=r"station\ deterministic",
                station_position=StationPosition(
                    latitude=deterministic_station_latitude,
                    longitude=deterministic_station_longitude,
                ),
            )
        }

    for station_id, station in stations.items():

        data["station_id"] += [station_id]
        data["latitude"] += [station.station_position.latitude]
        data["longitude"] += [station.station_position.longitude]
        data["altitude"] += [station.station_position.altitude]
        data["eastward_speed"] += [station.station_parameters.eastward_speed]
        data["northward_speed"] += [station.station_parameters.northward_speed]
        data["vertical_speed"] += [station.station_parameters.vertical_speed]
        data["range_bias"] += [station.station_parameters.range_bias]
        data["minimal_elevation_angle"] += [station.minimal_elevation_angle]

    output_path.mkdir(exist_ok=True, parents=True)
    DataFrame(data=data).to_csv(
        path_or_buf=output_path.joinpath(station_file_name + ".csv"),
        index=False,
    )


def test_forward_simulation(
    stations_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = DEFAULT_STATIONS_FILE_NAME,
    simulation_parameters_path: Path = TEST_NO_ITERATIONS_PATH,
    simulation_parameters_file_name: str = DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
) -> None:
    """
    Checks if the forward simulation of orbit determination runs for dummy forces.
    """

    stations = get_stations(path=stations_path, station_file_name=station_file_name)
    simulation_parameters = load_simulation_parameters(
        path=simulation_parameters_path,
        name=simulation_parameters_file_name,
    )
    simulation_parameters.update_for_stations(stations=stations)
    t, y, _ = propagate_ephemeris(
        simulation_parameters=simulation_parameters,
    )
    assert len(t) >= TEST_ARC_LENGTH // TEST_TIME_STEP
    assert len(y) == len(t)


def simulate_observations(
    stations_path: Path = TEST_OUTPUT_PATH,
    save_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = DEFAULT_STATIONS_FILE_NAME,
    simulation_parameters_path: Path = TEST_NO_ITERATIONS_PATH,
    simulation_parameters_file_name: str = DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
) -> tuple[SimulationParameters, dict[str, list[float]], Path, ndarray]:
    """
    Verifies if the measurements are correctly created in a forward simulation.
    """

    stations = get_stations(path=stations_path, station_file_name=station_file_name)
    simulation_parameters = load_simulation_parameters(
        path=simulation_parameters_path,
        name=simulation_parameters_file_name,
    )
    simulation_parameters.update_for_stations(stations=stations)
    t, y, _ = propagate_ephemeris(
        simulation_parameters=simulation_parameters,
    )
    path = save_path.joinpath(DEFAULT_MEASUREMENT_DIRECTORY_NAME).joinpath(
        simulation_parameters.arc_parameters.arc_id + ".csv"
    )

    if path.exists():

        remove(path)

    station_theoretical_measurements = simulate_measurements(
        t=t,
        y=y,
        stations=stations,
        simulation_parameters=simulation_parameters,
        path=save_path,
    )

    return simulation_parameters, station_theoretical_measurements, path, y


def test_observations(
    stations_path: Path = TEST_OUTPUT_PATH,
    save_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = DEFAULT_STATIONS_FILE_NAME,
    simulation_parameters_path: Path = TEST_NO_ITERATIONS_PATH,
    simulation_parameters_file_name: str = DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
) -> SimulationParameters:
    """
    Verifies if the measurements are correctly created in a forward simulation.
    """

    _, station_theoretical_measurements, path, _ = simulate_observations(
        stations_path=stations_path,
        save_path=save_path,
        station_file_name=station_file_name,
        simulation_parameters_path=simulation_parameters_path,
        simulation_parameters_file_name=simulation_parameters_file_name,
    )
    assert len(station_theoretical_measurements) > 0
    assert path.exists()


def test_arc_output(
    station_file_name: str = DEFAULT_STATIONS_FILE_NAME,
    simulation_parameters_path: Path = TEST_NO_ITERATIONS_PATH,
    simulation_parameters_file_name: str = DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
    measurements_directory_name: str = DEFAULT_MEASUREMENT_DIRECTORY_NAME,
) -> None:
    """
    Verifies if the measurements are correctly created in a forward simulation.
    """

    stations = get_stations(
        path=simulation_parameters_path.parent, station_file_name=station_file_name
    )
    simulation_parameters = load_simulation_parameters(
        path=simulation_parameters_path,
        name=simulation_parameters_file_name,
    )
    simulation_parameters.update_for_stations(stations=stations)
    simulation_parameters.arc_parameters.is_initial = False
    saved_observation_timestamps, saved_measurement_values = get_measurements(
        path=simulation_parameters_path.parent.joinpath(measurements_directory_name),
        name=simulation_parameters.arc_parameters.arc_id,
    )
    t, y, _ = propagate_ephemeris(
        simulation_parameters=simulation_parameters,
    )
    arc_output = ArcOutput(simulation_parameters=simulation_parameters, t=t, y=y)
    re_computed_measurement_values, _ = generate_measurements(
        arc_output=arc_output,
        stations=stations,
        observation_timestamps=saved_observation_timestamps,
    )

    # Verifies ArcOutput class saves and loads correctly.
    arc_output.update_observations(
        observation_timestamps=saved_observation_timestamps,
        theoretical_measurements=re_computed_measurement_values,
        real_measurements=saved_measurement_values,
    )
    arc_output.save(
        output_path=simulation_parameters_path,
        simulation_parameters_file_name=simulation_parameters_file_name,
    )
    arc_output_verification = load_arc_output(
        iteration_path=simulation_parameters_path,
        arc_id=simulation_parameters.arc_parameters.arc_id,
        simulation_parameters_file_name=simulation_parameters_file_name,
    )
    assert sum(abs(arc_output.t - arc_output_verification.t)) < NUMERICAL_TOLERANCE
    assert sum(sum(abs(arc_output.y - arc_output_verification.y))) < NUMERICAL_TOLERANCE

    for station_id in arc_output.observation_timestamps:

        assert (
            sum(
                abs(
                    array(object=arc_output.observation_timestamps[station_id])
                    - array(object=arc_output_verification.observation_timestamps[station_id])
                )
            )
            < NUMERICAL_TOLERANCE
        )
        assert (
            mean(
                abs(
                    array(object=arc_output.simulated_measurements[station_id])
                    - array(object=arc_output_verification.simulated_measurements[station_id])
                )
            )
            < NUMERICAL_TOLERANCE
        )
        assert (
            mean(
                abs(
                    array(object=arc_output.residuals[station_id])
                    - array(object=arc_output_verification.residuals[station_id])
                )
            )
            < NUMERICAL_TOLERANCE
        )

    # Verifies the extrapolated orbit is consistent with the initial simulation.
    for station_id, measurement_values in saved_measurement_values.items():

        assert len(measurement_values) == len(re_computed_measurement_values[station_id])
        assert (
            sum(abs(arc_output.residuals[station_id]))
            < len(re_computed_measurement_values[station_id])
            * TEST_SIGMA_SAFETY_FACTOR
            * stations[station_id].station_simulation.sigma_noise
        )


def test_quadrature(
    stations_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = DEFAULT_STATIONS_FILE_NAME,
    simulation_parameters_path: Path = TEST_NO_ITERATIONS_PATH,
    simulation_parameters_file_name: str = DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
    parameters_initial_guess: Optional[dict[str, float]] = None,
) -> None:
    """
    Verifies if the measurements are correctly created in a forward simulation.
    """

    path = run_single_arc(
        simulation_parameters=load_simulation_parameters(
            path=simulation_parameters_path,
            name=simulation_parameters_file_name,
        ),
        stations_path=stations_path,
        station_file_name=station_file_name,
        path=simulation_parameters_path,
        parameters_initial_guess=parameters_initial_guess,
    )

    # Verifies the normal equations save correctly.
    assert path.joinpath("a_matrix.json").exists()
    assert path.joinpath("b_second_member.json").exists()
    assert path.joinpath("parameters.json").exists()


def test_inversion(
    stations_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = DEFAULT_STATIONS_FILE_NAME,
    inversion_path: Path = TEST_INVERSION_PATH,
    simulation_parameters_path: Path = TEST_NO_ITERATIONS_PATH,
    simulation_parameters_file_name: str = DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
) -> None:
    """
    Retrieves the J_2 over 1 arcs for 1 satellite.
    """

    simulation_parameters, _, _, _ = simulate_observations(
        stations_path=stations_path,
        save_path=inversion_path,
        station_file_name=station_file_name,
        simulation_parameters_path=simulation_parameters_path,
        simulation_parameters_file_name=simulation_parameters_file_name,
    )
    solve_precise_orbit_determination(
        simulation_parameters_per_arc=[simulation_parameters],
        inversion_path=inversion_path,
        parameters_values_initial_guess_per_arc=[{r"J_2": 0.009}],
        station_file_name=station_file_name,
    )


def test_inversion_multiple_arcs(
    stations_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = DEFAULT_STATIONS_FILE_NAME,
    inversion_path: Path = TEST_OUTPUT_PATH.joinpath("test_inversion_2_arcs"),
    simulation_parameters_path: Path = TEST_NO_ITERATIONS_PATH,
    simulation_parameters_file_name: str = DEFAULT_SIMULATION_PARAMETERS_FILE_NAME,
) -> None:
    """
    Retrieves the J_2 over 2 different arcs for 1 satellite.
    """

    simulation_parameters, _, _, y = simulate_observations(
        stations_path=stations_path,
        save_path=inversion_path,
        station_file_name=station_file_name,
        simulation_parameters_path=simulation_parameters_path,
        simulation_parameters_file_name=simulation_parameters_file_name,
    )
    second_arc_simulation_parameters = deepcopy(x=simulation_parameters)

    # Builds the initial conditions for second arc.
    for parameter, value in zip(STATE_PARAMETERS.split(), y[-1] - y[0]):

        second_arc_simulation_parameters.terminal_parameter_values[parameter] = value

    second_arc_simulation_parameters_file_name = simulation_parameters_file_name + "_2"
    second_arc_simulation_parameters.arc_parameters.arc_id += "_2"
    second_arc_simulation_parameters.save(
        path=simulation_parameters_path, name=second_arc_simulation_parameters_file_name
    )
    second_arc_simulation_parameters, _, _, _ = simulate_observations(
        stations_path=stations_path,
        save_path=inversion_path,
        station_file_name=station_file_name,
        simulation_parameters_path=simulation_parameters_path,
        simulation_parameters_file_name=second_arc_simulation_parameters_file_name,
    )
    solve_precise_orbit_determination(
        simulation_parameters_per_arc=[simulation_parameters, second_arc_simulation_parameters],
        inversion_path=inversion_path,
        parameters_values_initial_guess_per_arc=2 * [{r"J_2": 0.009}],
        station_file_name=station_file_name,
        parameters_to_accumulate=[r"J_2"],
    )


if __name__ == "__main__":

    test_inversion()

# TODO: Sub-function that simulates measurements for a given satellite for n_arcs.
# TODO: Test function that cumulates the J_2 over 1 arcs for 2 satellite.
# TODO: Test function that cumulates the J_2 over 2 arcs for 2 satellite.
