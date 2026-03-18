"""
All base functionalities. To test via pytest test.py.
"""

# TODO: Test function that retrieves dummy parameter.
# TODO: Test function that cumulates the dummy parameter over 2 arcs.

from datetime import datetime, timedelta
from pathlib import Path
from shutil import rmtree
from typing import Optional

from base_models import save_base_model
from numpy import arange, array, concatenate, matmul, mean, ndarray
from pandas import DataFrame
from sympy import Symbol

from simone import (
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
    propagate_partials_and_save,
    save_normal_equations,
    simulate_measurements,
)

TEST_STATION_QUANTITY = 100
TEST_SIGMA_SAFETY_FACTOR = 10
TEST_ARC_LENGTH = 10000.0
TEST_TIME_STEP = 30.0
TEST_TIME_STEP_FOR_TEST_TIME_PARAMETER = 1000.0
TEST_ARC_START_DATETIME = datetime(
    year=2000, month=1, day=1, hour=0, minute=0, second=0, microsecond=0
)
TEST_ARC_PARAMETERS = ArcParameters(
    time_step=TEST_TIME_STEP,
    arc_start_datetime=TEST_ARC_START_DATETIME,
    arc_length=TEST_ARC_LENGTH,
    lagrange_interpolation_order=4,
    arc_id="test_arc_id",
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


def test_clear_test_folder(output_path: Path = TEST_OUTPUT_PATH) -> None:
    """
    Deletes the test folder before all tests.
    """

    if output_path.exists():

        rmtree(output_path)


def test_generate_simulation_parameters(
    output_path: Path = TEST_OUTPUT_PATH,
    simulation_parameters_file_name: str = "simulation_parameters",
) -> None:
    """
    Generates simulation parameters and save in (.JSON) file to be loaded by other test functions.
    """

    SimulationParameters(
        arc_parameters=TEST_ARC_PARAMETERS,
        simulated_forces=TEST_SIMULATED_FORCES,
    ).save(
        output_path=output_path.joinpath(TEST_ARC_PARAMETERS.arc_id),
        name=simulation_parameters_file_name,
    )
    assert (
        output_path.joinpath(TEST_ARC_PARAMETERS.arc_id)
        .joinpath(simulation_parameters_file_name + ".json")
        .exists()
    )


def test_generate_stations(
    output_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = "stations",
    arc_id: str = TEST_ARC_PARAMETERS.arc_id,
    simulation_parameters_file_name: str = "simulation_parameters",
    # Ensures a virtual station at least has the satellite in visibility at the arc's beginning.
    deterministic_station: bool = True,
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
                output_path=output_path,
                arc_id=arc_id,
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
    output_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = "stations",
    arc_id: str = TEST_ARC_PARAMETERS.arc_id,
    simulation_parameters_file_name: str = "simulation_parameters",
) -> None:
    """
    Checks if the forward simulation of orbit determination runs for dummy forces.
    """

    stations = get_stations(stations_path=output_path, station_file_name=station_file_name)
    simulation_parameters = load_simulation_parameters(
        output_path=output_path,
        arc_id=arc_id,
        name=simulation_parameters_file_name,
    )
    simulation_parameters.update_for_stations(stations=stations)
    t, y, _ = propagate_ephemeris(
        simulation_parameters=simulation_parameters,
    )
    assert len(t) >= TEST_ARC_LENGTH // TEST_TIME_STEP
    assert len(y) == len(t)


def test_observations(
    output_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = "stations",
    arc_id: str = TEST_ARC_PARAMETERS.arc_id,
    simulation_parameters_file_name: str = "simulation_parameters",
) -> None:
    """
    Verifies if the measurements are correctly created in a forward simulation.
    """

    stations = get_stations(stations_path=output_path, station_file_name=station_file_name)
    simulation_parameters = load_simulation_parameters(
        output_path=output_path,
        arc_id=arc_id,
        name=simulation_parameters_file_name,
    )
    simulation_parameters.update_for_stations(stations=stations)
    t, y, _ = propagate_ephemeris(
        simulation_parameters=simulation_parameters,
    )
    path = output_path.joinpath(simulation_parameters.arc_parameters.arc_id).joinpath(
        "measurements.csv"
    )

    if path.exists():

        rmtree(path)

    station_theoretical_measurements = simulate_measurements(
        t=t, y=y, stations=stations, simulation_parameters=simulation_parameters
    )

    assert len(station_theoretical_measurements) > 0
    assert path.exists()


def test_arc_output(
    output_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = "stations",
    arc_id: str = TEST_ARC_PARAMETERS.arc_id,
    simulation_parameters_file_name: str = "simulation_parameters",
) -> None:
    """
    Verifies if the measurements are correctly created in a forward simulation.
    """

    stations = get_stations(stations_path=output_path, station_file_name=station_file_name)
    simulation_parameters = load_simulation_parameters(
        output_path=output_path,
        arc_id=arc_id,
        name=simulation_parameters_file_name,
    )
    simulation_parameters.update_for_stations(stations=stations)
    simulation_parameters.arc_parameters.is_initial = False
    saved_observation_timestamps, saved_measurement_values = get_measurements(
        simulation_parameters=simulation_parameters
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
    arc_output.save(output_path=output_path)
    arc_output_verification = load_arc_output(output_path=output_path, arc_id=arc_id)
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
    output_path: Path = TEST_OUTPUT_PATH,
    station_file_name: str = "stations",
    arc_id: str = TEST_ARC_PARAMETERS.arc_id,
    simulation_parameters_file_name: str = "simulation_parameters",
    parameters_to_invert: Optional[dict[str, list[str]]] = None,
) -> None:
    """
    Verifies if the measurements are correctly created in a forward simulation.
    """

    if parameters_to_invert is None:

        parameters_to_invert = {"dynamic": None, "station": None}

    stations = get_stations(stations_path=output_path, station_file_name=station_file_name)
    simulation_parameters = load_simulation_parameters(
        output_path=output_path,
        arc_id=arc_id,
        name=simulation_parameters_file_name,
    )
    simulation_parameters.update_for_stations(stations=stations)
    simulation_parameters.arc_parameters.is_initial = False
    observation_partials: dict[str, ndarray]

    arc_output, observation_partials, parameters_to_invert = propagate_partials_and_save(
        stations=stations,
        simulation_parameters=simulation_parameters,
        parameters_to_invert=parameters_to_invert,
    )

    # Verifies the normal equations save correctly.
    a_matrix = array(object=list(observation_partials.values())).T
    b_second_member = concatenate(list(arc_output.residuals.values()))[:, None]
    n_matrix = array(
        object=matmul(a_matrix.T, a_matrix), dtype=float
    )  # TODO: ponderate by uncertainties.
    s_second_member = array(
        object=matmul(a_matrix.T, b_second_member), dtype=float
    )  # TODO: ponderate by uncertainties.
    path = save_normal_equations(
        n_matrix=n_matrix,
        s_second_member=s_second_member,
        simulation_parameters=simulation_parameters,
    )
    save_base_model(obj=parameters_to_invert, name="parameters", path=path)

    assert path.joinpath("n_matrix.json").exists()
    assert path.joinpath("s_second_member.json").exists()
    assert path.joinpath("parameters.json").exists()
