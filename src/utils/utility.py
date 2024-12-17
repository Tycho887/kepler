import numpy as np
import math
import json


# Constants
MU_EARTH = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
SECONDS_PER_DAY = 86400.0  # Seconds in a day


def synodic_period(period1, period2):
	"""
	Computes the synodic period between two orbital periods.
	"""
	return 1 / np.abs(1 / period1 - 1 / period2)


def load_tle_data(file_name):
    """
	Load TLE data from a JSON file.
	"""
	try:
		with open(file_name, "r") as json_file:
			data = json.load(json_file)
		return data
	except Exception as e:
		print(f"Error loading TLE data: {e}")
		return None


def tle_to_keplerian(mean_motion, eccentricity, inclination, raan, arg_perigee, mean_anomaly):
    """
    Convert TLE parameters to Keplerian elements.

    Parameters:
    - mean_motion (float): Revolutions per day.
    - eccentricity (float): Eccentricity (decimal format, no implied point).
    - inclination (float): Inclination (degrees).
    - raan (float): Right Ascension of Ascending Node (degrees).
    - arg_perigee (float): Argument of Perigee (degrees).
    - mean_anomaly (float): Mean Anomaly (degrees).

    Returns:
    - dict: Keplerian elements.
    """
    # Mean Motion (n) to radians per second
    n_rad_per_sec = (mean_motion * 2 * math.pi) / SECONDS_PER_DAY
    
    # Semi-Major Axis (a) from Kepler's Third Law
    semi_major_axis = (MU_EARTH / (n_rad_per_sec ** 2)) ** (1 / 3)
    
    # Return Keplerian elements
    keplerian_elements = {
        "Semi-Major Axis (km)": semi_major_axis,
        "Eccentricity": eccentricity,
        "Inclination (deg)": inclination,
        "RAAN (deg)": raan,
        "Argument of Perigee (deg)": arg_perigee,
        "Mean Anomaly (deg)": mean_anomaly
    }
    
    return keplerian_elements

