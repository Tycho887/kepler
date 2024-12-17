import numpy as np

def synodic_period(period1, period2):
	"""
	Computes the synodic period between two orbital periods.
	"""
	return 1 / np.abs(1 / period1 - 1 / period2)


def load_tle(filename):
	"""
	Loads TLE data from a file.
	"""
	with open(filename, "r") as file:
		lines = file.readlines()
		tle = [lines[i:i+3] for i in range(0, len(lines), 3)]
	return tle

def convert_tle_to_orbital_elements(tle):
	"""
	Converts TLE data to orbital elements.
	"""
