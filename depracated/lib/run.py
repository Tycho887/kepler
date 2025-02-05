import math
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sgp4.api import Satrec
from datetime import datetime, timedelta
from sgp4.api import WGS72
import sqlite3
import random

class SatellitePropagator:
    """
    A class to propagate a satellite's orbit using TLE data.
    """

    def __init__(self, tle_data: tuple):
        """
        Initialize the SatellitePropagator with TLE data.

        Args:
            tle_data (tuple): A tuple containing the satellite ID, name, and two TLE lines, plus start time.
        """
        self.TLE = tle_data
        self.id = tle_data[0]
        self.name = tle_data[1]
        self.satellite = Satrec.twoline2rv(tle_data[2], tle_data[3], WGS72)
        self.start_time = datetime.fromisoformat(tle_data[4])

        self._position_data = None
        self.error_code = None

    def propagate(self, minutes_into_future: float, step_size: float):
        """
        Public method to propagate satellite position.

        Args:
            minutes_into_future (float): Time to propagate into the future in minutes.
            step_size (float): Step size for propagation in minutes.
        """
        try:
            self._propagate(minutes_into_future, step_size)
        except Exception as e:
            print(e)
            self.error_code = e

    def _propagate(self, minutes_into_future: float, step_size: float):
        """
        Internal method to propagate the satellite's orbit.

        Args:
            minutes_into_future (float): Minutes to propagate into the future.
            step_size (float): Time step size in minutes.
        """
        times, positions, velocities = [], [], []
        current_time = self.start_time

        while (current_time - self.start_time).total_seconds() / 60 <= minutes_into_future:
            jd, fr = self._datetime_to_julian(current_time)
            error_code, position, velocity = self.satellite.sgp4(jd, fr)

            if error_code != 0:
                raise RuntimeError(f"SGP4 propagation error code: {error_code}")

            times.append(current_time)
            positions.append(position)
            velocities.append(velocity)
            current_time += timedelta(minutes=step_size)

        self._position_data = {
            "times": np.array(times),
            "positions_km": np.array(positions,dtype=np.float32)
        }

    @property
    def positions(self):
        """
        Get the position data of the satellite.

        Returns:
            np.ndarray: Array of satellite positions in kilometers.
        """
        if self._position_data is None:
            raise RuntimeError("Satellite position data has not been propagated yet.")
        return self._position_data["positions_km"]

    @property
    def times(self):
        """
        Get the time steps of the satellite propagation.

        Returns:
            np.ndarray: Array of datetime objects corresponding to propagation steps.
        """
        if self._position_data is None:
            raise RuntimeError("Satellite position data has not been propagated yet.")
        return self._position_data["times"]

    @staticmethod
    def _datetime_to_julian(date: datetime):
        """
        Convert a datetime object to Julian Date and fractional day.

        Args:
            date (datetime): The datetime to convert.

        Returns:
            tuple: Julian date integer part and fractional part.
        """
        jd_epoch = 2451545.0  # Julian Date at epoch 2000-01-01 12:00:00
        year, month, day = date.year, date.month, date.day
        hour, minute, second = date.hour, date.minute, date.second

        if month <= 2:
            year -= 1
            month += 12

        A = int(year / 100)
        B = 2 - A + int(A / 4)
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
        fractional_day = (hour + minute / 60 + second / 3600) / 24.0

        jd_int = int(jd)
        fr = jd - jd_int + fractional_day
        return jd_int, fr


def propagate_orbits(satellites: dict, minutes_into_future: int, step_size: float):
    """Propagate the orbits of satellites in parallel using all CPU cores."""
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(sat.propagate, minutes_into_future, step_size): sat
            for sat in satellites.values()
        }
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            sat = futures[future]
            try:
                future.result()
            except RuntimeError as e:
                sat.error_code = e


def fetch_tle_data_single(index: int):
	"""Fetch TLE data for a single satellite."""
	query = f"SELECT * FROM tle_data WHERE id = {index}"
	
	with sqlite3.connect("database/satellites_tle.db") as conn:
		cursor = conn.cursor()
		cursor.execute(query)
		data = cursor.fetchone()
    
	return data

def number_of_satellites():	
	"""Get the number of satellites in the database."""
	query = "SELECT COUNT(*) FROM tle_data"
	
	with sqlite3.connect("database/satellites_tle.db") as conn:
		cursor = conn.cursor()
		cursor.execute(query)
		data = cursor.fetchone()
	
	return data[0]

def initialize_satellites(indexes: list):
	"""Initialize satellites based on given indexes."""
	satellites = {}
	
	for index in indexes:
		satellite = SatellitePropagator(fetch_tle_data_single(index))
		satellites[index] = satellite
	return satellites


def closest_distance(positions1, positions2):
    """
    Calculate the closest distance between two sets of positions.

    Parameters:
    - positions1 (np.ndarray): Array of positions for the first object.
    - positions2 (np.ndarray): Array of positions for the second object.

    Returns:
    - tuple: Closest distance between the two objects and the row index where this occurs.
    """
    distances = np.linalg.norm(positions1 - positions2, axis=1)
    min_distance = np.min(distances)
    min_index = np.argmin(distances)

    return min_distance, min_index


def compute_closest_approaches(satellites: dict, pair_data_dict: dict):
    """Calculate the closest approach for satellite pairs in parallel."""
    print("Computing closest approaches...")

    def process_pair(pair_id, pair_info):
        id1, id2 = pair_info["sats"]
        sat1, sat2 = satellites[id1], satellites[id2]
        positions1, positions2 = sat1.positions, sat2.positions
        min_distance, min_id = closest_distance(positions1, positions2)
        pair_info["min_distance"] = min_distance
        pair_info["min_distance_time"] = sat1.times[min_id]
        pair_info["close_approach"] = False
        return pair_id, pair_info

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_pair, pair_id, pair_info): pair_id
            for pair_id, pair_info in pair_data_dict.items()
        }
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            pair_id, pair_info = future.result()
            pair_data_dict[pair_id] = pair_info

    return pair_data_dict

def detect_threats(satellites: dict, pair_data_dict: dict, threshold: float):
	"""Detect potential threats based on minimum distances."""
	threat_flag = False
	for pair_id, pair_info in pair_data_dict.items():
		if pair_info["min_distance"] < threshold:
			threat_flag = True
			sat1_name = satellites[pair_info["sats"][0]].name
			sat2_name = satellites[pair_info["sats"][1]].name
			pair_info["close_approach"] = True

	if not threat_flag:
		print("No close approaches detected.")

	print(f"Number of positive close approaches: {sum([pair_info['close_approach'] for pair_info in pair_data_dict.values()])}")
	print(f"Number of negative close approaches: {len(pair_data_dict) - sum([pair_info['close_approach'] for pair_info in pair_data_dict.values()])}")
	print(f"Fraction of close approaches: {sum([pair_info['close_approach'] for pair_info in pair_data_dict.values()]) / len(pair_data_dict)}")

	return pair_data_dict

def filter_error_satellites(satellites: dict, pair_data_dict: dict, indexes: list):
	"""Filter out satellites with errors."""


	valid_indexes = [id for id in indexes if satellites[id].error_code is None]
	valid_sats = {id: satellites[id] for id in valid_indexes}

	valid_dict = {}

	for key, value in pair_data_dict.items():
		if value["sats"][0] in valid_indexes and value["sats"][1] in valid_indexes:
			valid_dict[key] = value

	return valid_sats, valid_dict, valid_indexes

def generate_unique_random_integers(N: int, n: int) -> list:
    """
    Generates a list of n unique random integers below a limit N.

    Args:
        n (int): Number of integers to generate.
        N (int): The upper limit (exclusive) for the random integers.

    Returns:
        list: A list of n unique random integers below N.

    Raises:
        ValueError: If n is greater than N, making it impossible to generate unique numbers.
    """
    if n > N:
        raise ValueError("n must be less than or equal to N to ensure unique values.")

    return random.sample(range(N), n)



def generate_satellite_pair_dict(integers: list, num_pairs: int) -> dict:
    """
    Generate a dictionary of random satellite pairs.

    Args:
        integers (list): A list of integers to generate pairs from.
        num_pairs (int): The number of pairs to generate.

    Returns:
        dict: A dictionary with pair IDs as keys and pair information as values.
    """

    # Ensure the number of requested pairs does not exceed the maximum unique combinations
    assert num_pairs <= math.comb(len(integers), 2), (
        "Number of pairs exceeds the number of unique pairs that can be generated."
    )

    pairs_set = set()

    while len(pairs_set) < num_pairs:
        pair = tuple(sorted(random.sample(integers, 2)))
        pairs_set.add(pair)

    pairs_dict = {
        pair_id: {
            "sats": list(pair),
            "min_distance": None,
            "min_distance_time": None,
            "close_approach": False,
        }
        for pair_id, pair in enumerate(pairs_set)
    }

    return pairs_dict


def run_sim(N_sats: int, days: int, step_size: float, daily_error: float):
	
	propagation_error = daily_error * days  # Error threshold in kilometers

	# Generate satellite indexes
	total_satellites = number_of_satellites()
	indexes = generate_unique_random_integers(total_satellites, N_sats)

	# Initialize satellites and pair dictionary
	satellites = initialize_satellites(indexes)
	max_comparisons = math.comb(N_sats, 2)
	pairs = generate_satellite_pair_dict(indexes, max_comparisons)

	# print(pairs)

	# Propagate satellite orbits
	print("Propagating orbits...")
	propagate_orbits(satellites, 60 * 24 * days, step_size)

	# Filter out satellites with errors
	satellites, pairs, indexes = filter_error_satellites(satellites, pairs, indexes)

	# Compute closest approaches
	pairs = compute_closest_approaches(satellites, pairs)

	# Detect potential threats
	pairs = detect_threats(satellites, pairs, propagation_error)
	
	return satellites, pairs, indexes