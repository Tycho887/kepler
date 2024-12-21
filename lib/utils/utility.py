import numpy as np
import math
from sgp4.api import Satrec
from sgp4.conveniences import jday
import math
import random
from datetime import datetime

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
            "min_distance_velocity": None
        }
        for pair_id, pair in enumerate(pairs_set)
    }

    return pairs_dict


def datetime_to_julian(date: datetime):
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


if __name__ == "__main__":
    print(generate_satellite_pair_dict([1, 2, 3, 4, 5], 10))
