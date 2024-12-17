import numpy as np
import math
from sgp4.api import Satrec
from sgp4.conveniences import jday
import math



def extract_mean_motion(tle_data):
    """
    Extract the mean motion (revolutions per day) from TLE data.
    """
    satellite = Satrec.twoline2rv(tle_data[2], tle_data[3])
    mean_motion = satellite.no_kozai * 1440 / (2 * math.pi)  # Convert radians per minute to rev/day
    return mean_motion

def synodic_period(tle_data1, tle_data2):
    """
    Calculate the synodic period between two satellites.

    Parameters:
    - tle_data1 (tuple): TLE data for the first satellite (name, line1, line2).
    - tle_data2 (tuple): TLE data for the second satellite (name, line1, line2).

    Returns:
    - float: Synodic period in minutes.
    """
    # Extract mean motions (revolutions per day) from the TLE data
    mean_motion1 = extract_mean_motion(tle_data1)
    mean_motion2 = extract_mean_motion(tle_data2)

    # Compute orbital periods (in minutes)
    period1 = 1440 / mean_motion1
    period2 = 1440 / mean_motion2

    # Calculate the synodic period
    synodic_period = abs(1 / (1 / period1 - 1 / period2))

    return synodic_period