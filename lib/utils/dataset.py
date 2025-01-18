import pandas as pd

def get_TLE_info(SatellitePropagator):
    """
    Extracts orbital parameters and other coefficients from TLE data.
    
    Parameters:
        TLE_data (tuple): Tuple containing satellite name, international designator, and two-line elements.

    Returns:
        dict: Dictionary containing orbital parameters and coefficients.
    """
    # Extract the relevant info for constructing the dataset

    TLE_data = SatellitePropagator.TLE

    line1 = TLE_data[2]
    line2 = TLE_data[3]

    epoch_day = float(line1[20:32])
    ballistic_coefficient = float(line1[33:43])

    inclination = float(line2[8:16])
    right_ascension_of_the_ascending_node = float(line2[17:25])
    eccentricity = float("0." + line2[26:33])  # Add leading 0 and convert to decimal
    argument_of_perigee = float(line2[34:42])
    mean_anomaly = float(line2[43:51])
    mean_motion = float(line2[52:63])

    # Calculate apogee and perigee (assume Earth radius = 6371 km)
    earth_radius_km = 6371.0
    semi_major_axis = (86400 / mean_motion / (2 * 3.141592653589793)) ** (2 / 3)
    apogee = semi_major_axis * (1 + eccentricity) - earth_radius_km
    perigee = semi_major_axis * (1 - eccentricity) - earth_radius_km

    return {
        "epoch day": epoch_day,
        "ballistic coefficient": ballistic_coefficient,
        "inclination": inclination,
        "right ascension of the ascending node": right_ascension_of_the_ascending_node,
        "eccentricity": eccentricity,
        "argument of perigee": argument_of_perigee,
        "mean anomaly": mean_anomaly,
        "mean motion": mean_motion,
        "apogee": apogee,
        "perigee": perigee,
    }

def orbit_difference(tle1, tle2):
    """
    Calculates differences in orbital parameters between two TLE datasets.
    
    Parameters:
        tle1 (tuple): TLE data for the first satellite.
        tle2 (tuple): TLE data for the second satellite.

    Returns:
        dict: Dictionary containing the differences in orbital parameters.
    """
    info1 = get_TLE_info(tle1)
    info2 = get_TLE_info(tle2)

    return {
        "inclination difference": info1["inclination"] - info2["inclination"],
        "eccentricity difference": info1["eccentricity"] - info2["eccentricity"],
        "ascending node difference": info1["right ascension of the ascending node"] - info2["right ascension of the ascending node"],
        "argument of perigee difference": info1["argument of perigee"] - info2["argument of perigee"],
        "mean anomaly difference": info1["mean anomaly"] - info2["mean anomaly"],
        "mean motion difference": info1["mean motion"] - info2["mean motion"],
        "apogee difference": info1["apogee"] - info2["apogee"],
        "perigee difference": info1["perigee"] - info2["perigee"],
    }

def build_dataset(satellites: dict, pairs: dict, indexes: list):
    """
    Constructs a dataset from TLE data of specified satellites and pairs.
    
    Parameters:
        satellites (list): List of TLE datasets for satellites.
        pairs (list): List of pairs of satellite indices to compare.
        indexes (list): Target variable indicating whether a close approach is detected.
    
    Returns:
        pd.DataFrame: DataFrame containing satellite orbital data and differences.
    """
    data = []

    for i, _dict in pairs.items():

        idx1, idx2 = _dict["sats"]

        sat1 = satellites[idx1]
        sat2 = satellites[idx2]

        info1 = get_TLE_info(sat1)
        info2 = get_TLE_info(sat2)
        differences = orbit_difference(sat1, sat2)

        # Combine data into a single flat structure for each pair
        row = {
            **{f"sat1_{key}": value for key, value in info1.items()},
            **{f"sat2_{key}": value for key, value in info2.items()},
            **differences,
            "close_approach": _dict["close_approach"]
        }
        data.append(row)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)
    return df


def store_dataset(df):
    """
    Stores the dataset to a CSV file.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the dataset.
        filename (str): Name of the file to store the dataset.
    """

    filename = f"datasets/tle{len(df)}.csv"
    

    df.to_csv(filename, index=False)