import sqlite3
import pandas as pd
import math
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sgp4.api import Satrec, WGS72
from datetime import datetime, timedelta
import sqlite3
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SatellitePropagator:
    """
    A class to propagate a satellite's orbit using TLE data.
    """
    def __init__(self, tle_data):
        self.id, self.name, tle1, tle2, start_time = tle_data
        self.satellite = Satrec.twoline2rv(tle1, tle2, WGS72)
        self.start_time = datetime.fromisoformat(start_time)
        self._position_data = None
        self.error_code = None

    def propagate(self, duration_minutes, step_size):
        try:
            self._propagate(duration_minutes, step_size)
        except Exception as e:
            logging.error(f"Error propagating satellite {self.id}: {e}")
            self.error_code = e

    def _propagate(self, duration_minutes, step_size):
        times, positions = [], []
        current_time = self.start_time
        
        while (current_time - self.start_time).total_seconds() / 60 <= duration_minutes:
            jd, fr = self._datetime_to_julian(current_time)
            error_code, position, _ = self.satellite.sgp4(jd, fr)
            
            if error_code:
                raise RuntimeError(f"SGP4 error code: {error_code}")
            
            times.append(current_time)
            positions.append(position)
            current_time += timedelta(minutes=step_size)
        
        self._position_data = {"times": np.array(times), "positions_km": np.array(positions, dtype=np.float32)}

    @property
    def positions(self):
        if self._position_data is None:
            raise RuntimeError("Satellite data has not been propagated.")
        return self._position_data["positions_km"]
    
    @property
    def times(self):
        if self._position_data is None:
            raise RuntimeError("Satellite data has not been propagated.")
        return self._position_data["times"]

    @staticmethod
    def _datetime_to_julian(date):
        jd = date.toordinal() + 1721424.5
        fr = (date.hour + date.minute / 60 + date.second / 3600) / 24.0
        return int(jd), fr


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
    logging.info("Computing closest approaches...")

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
        logging.info("No close approaches detected.")

    logging.info(f"Number of positive close approaches: {sum([pair_info['close_approach'] for pair_info in pair_data_dict.values()])}")
    logging.info(f"Number of negative close approaches: {len(pair_data_dict) - sum([pair_info['close_approach'] for pair_info in pair_data_dict.values()])}")
    logging.info(f"Fraction of close approaches: {sum([pair_info['close_approach'] for pair_info in pair_data_dict.values()]) / len(pair_data_dict)}")

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

    # Propagate satellite orbits
    logging.info("Propagating orbits...")
    propagate_orbits(satellites, 60 * 24 * days, step_size)

    # Filter out satellites with errors
    satellites, pairs, indexes = filter_error_satellites(satellites, pairs, indexes)

    # Compute closest approaches
    pairs = compute_closest_approaches(satellites, pairs)

    # Detect potential threats
    pairs = detect_threats(satellites, pairs, propagation_error)
    
    return satellites, pairs, indexes


def query_orbital_data(db_path, indexes):
    """
    Query the orbital_data table from the database.
    
    Parameters:
        db_path (str): Path to the SQLite database.
    
    Returns:
        pd.DataFrame: DataFrame containing the orbital data.
    """
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT *
    FROM orbital_data
    WHERE id IN ({", ".join([str(i) for i in indexes])})
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # set index to satellite_id

    df.set_index("id", inplace=True)

    return df

def calculate_differences(row1, row2):
    """
    Calculate differences in orbital parameters between two satellites.
    
    Parameters:
        row1 (pd.Series): Orbital data for the first satellite.
        row2 (pd.Series): Orbital data for the second satellite.
    
    Returns:
        dict: Dictionary containing the differences in orbital parameters.
    """
    
    # For all numeric columns, calculate the difference

    differences = {}

    for key1, key2 in zip(row1.keys(), row2.keys()):
        if isinstance(row1[key1], (int, float)) and isinstance(row2[key2], (int, float)):
            differences[f"{key1}_difference"] = row1[key1] - row2[key2]

    # Add the synodic period 
    if "period" in row1 and "period" in row2:
        differences["synodic_period"] = 1 / (1 / row1["period"] - 1 / row2["period"])
    else:
        raise ValueError("Missing period data in the orbital data.")

    return differences

def build_dataset(db_path, pairs, indexes):
    """
    Constructs a dataset from orbital data of specified satellites and pairs.
    
    Parameters:
        db_path (str): Path to the SQLite database.
        pairs (dict): Dictionary of pairs of satellite indices to compare.
        indexes (list): Target variable indicating whether a close approach is detected.
    
    Returns:
        pd.DataFrame: DataFrame containing satellite orbital data and differences.
    """
    orbital_data = query_orbital_data(db_path, indexes)


    data = []

    for i, _dict in pairs.items():
        idx1, idx2 = _dict["sats"]

        # Check if satellite IDs exist in the orbital_data table
        if idx1 not in orbital_data.index or idx2 not in orbital_data.index:
            print(f"Satellite ID {idx1} or {idx2} not found in the database.")
            continue

        sat1_data = orbital_data.loc[idx1]
        sat2_data = orbital_data.loc[idx2]

        differences = calculate_differences(sat1_data, sat2_data)

        # Combine data into a single flat structure for each pair
        row = {
            **{f"sat1_{key}": value for key, value in sat1_data.items()},
            **{f"sat2_{key}": value for key, value in sat2_data.items()},
            **differences,
            "close_approach": _dict["close_approach"]
        }
        data.append(row)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)
    return df

def store_dataset(df, filename):
    """
    Stores the dataset to a CSV file.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the dataset.
        filename (str): Name of the file to store the dataset.
    """
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    db_path = "database/satellites_tle.db"
    N = 5
    days = 1
    step_size = 0.1  # Step size in minutes
    daily_error = 2.0  # Error in kilometers per day

    sats, pairs, indexes = run_sim(N, days, step_size, daily_error)

    df = build_dataset(db_path, pairs, indexes)

    store_dataset(df, f"datasets/tle{len(df)}.csv")


