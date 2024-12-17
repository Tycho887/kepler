import math

# Constants
MU_EARTH = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
SECONDS_PER_DAY = 86400.0  # Seconds in a day

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

# Example Usage with ISS TLE Line 2:
mean_motion = 15.49709696  # From Line 2
eccentricity = 0.0008967   # Decimal implied
inclination = 51.6448      # Degrees
raan = 85.7573             # Degrees
arg_perigee = 60.4576      # Degrees
mean_anomaly = 300.8325    # Degrees

keplerian = tle_to_keplerian(mean_motion, eccentricity, inclination, raan, arg_perigee, mean_anomaly)
print("Keplerian Elements:", keplerian)
