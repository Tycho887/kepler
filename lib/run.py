from .SGP4 import *
import lib.utility as utils
import lib as db
import math
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def propagate_orbits(satellites: dict, minutes_into_future: int, step_size: float):
    """Propagate the orbits of satellites in parallel."""
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
			

def initialize_satellites(indexes: list):
	"""Initialize satellites based on given indexes."""
	satellites = {}
	for index in indexes:
		satellite = SatellitePropagator(db.fetch_tle_data_single(index))
		satellites[index] = satellite
	return satellites

def compute_closest_approaches(satellites: dict, pair_data_dict: dict):
    """Calculate the closest approach for satellite pairs in parallel."""
    print("Computing closest approaches...")

    def process_pair(pair_id, pair_info):
        id1, id2 = pair_info["sats"]
        sat1, sat2 = satellites[id1], satellites[id2]
        positions1, positions2 = sat1.positions, sat2.positions
        min_distance, min_id = utils.closest_distance(positions1, positions2)
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

def run_sim(N_sats: int, days: int, step_size: float, daily_error: float):
	
	propagation_error = daily_error * days  # Error threshold in kilometers

	# Generate satellite indexes
	total_satellites = db.number_of_satellites()
	indexes = utils.generate_unique_random_integers(total_satellites, N_sats)

	# Initialize satellites and pair dictionary
	satellites = initialize_satellites(indexes)
	max_comparisons = math.comb(N_sats, 2)
	pairs = utils.generate_satellite_pair_dict(indexes, max_comparisons)

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