import numpy as np
from lib.simulation import SatellitePropagator
from lib.plotting import plot_3d, plot_distance
import lib.database as db
from datetime import datetime
import lib.utils as utils
import random
import math

def propagate_orbits(satellites, minutes_into_future, step_size):
	"""Propagate the orbits of satellites."""
	for sat in satellites.values():
		print(f"Propagating orbit for {sat.name}...")
		try:
			sat.propagate(minutes_into_future, step_size)
		except RuntimeError as e:
			print(f"Error propagating orbit for {sat.name}: {e}")
			sat.error_code = e

def initialize_satellites(indexes):
	"""Initialize satellites based on given indexes."""
	satellites = {}
	for index in indexes:
		satellite = SatellitePropagator(db.fetch_tle_data_single(index))
		satellites[index] = satellite
	return satellites

def compute_closest_approaches(satellites, pair_data_dict):
	"""Calculate the closest approach for satellite pairs."""
	for pair_id, pair_info in pair_data_dict.items():
		id1, id2 = pair_info["sats"]
		sat1, sat2 = satellites[id1], satellites[id2]
		positions1, positions2 = sat1.positions, sat2.positions
		
		min_distance, min_id = utils.closest_distance(positions1, positions2)
		pair_info["min_distance"] = min_distance
		pair_info["min_distance_time"] = sat1.times[min_id]
	return pair_data_dict

def detect_threats(satellites, pair_data_dict, threshold):
	"""Detect potential threats based on minimum distances."""
	threat_flag = False
	for pair_id, pair_info in pair_data_dict.items():
		if pair_info["min_distance"] < threshold:
			threat_flag = True
			sat1_name = satellites[pair_info["sats"][0]].name
			sat2_name = satellites[pair_info["sats"][1]].name
			print(f"Close approach detected between {sat1_name} and {sat2_name} at {pair_info['min_distance_time']}")
			print(f"Minimum distance: {pair_info['min_distance']:.2f} km")

	if not threat_flag:
		print("No close approaches detected.")

def filter_error_satellites(satellites, indexes):
	"""Filter out satellites with errors."""
	valid_indexes = [id for id in indexes if satellites[id].error_code is None]
	return {id: satellites[id] for id in valid_indexes}, valid_indexes

if __name__ == "__main__":
	N = 10
	days = 10
	step_size = 0.1  # Step size in minutes
	propagation_error = 1.5 * days  # Error threshold in kilometers

	# Generate satellite indexes
	total_satellites = db.number_of_satellites()
	indexes = utils.generate_unique_random_integers(total_satellites, N)

	# Initialize satellites and pair dictionary
	satellites = initialize_satellites(indexes)
	max_comparisons = math.comb(N, 2)
	pairs = utils.generate_satellite_pair_dict(indexes, max_comparisons)

	# print(pairs)

	# Propagate satellite orbits
	print("Propagating orbits...")
	propagate_orbits(satellites, 60 * 24 * days, step_size)

	# Filter out satellites with errors
	satellites, indexes = filter_error_satellites(satellites, indexes)

	# Compute closest approaches
	pairs = compute_closest_approaches(satellites, pairs)

	# Detect potential threats
	detect_threats(satellites, pairs, propagation_error)
