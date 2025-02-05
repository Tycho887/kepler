import sqlite3
import pandas as pd
import math
import numpy as np
import random
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sgp4.api import Satrec, WGS72
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import sqlite3

conn = sqlite3.connect("database/satellites_tle.db")
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM tle_data")
data = cursor.fetchone()
print("Satellite count:", data)
conn.close()

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


def propagate_orbits(satellites, duration_minutes, step_size):
    """Propagate satellite orbits in parallel."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(sat.propagate, duration_minutes, step_size): sat for sat in satellites.values()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()


def fetch_tle_data(index):
    """Fetch TLE data for a satellite using parameterized query."""
    query = "SELECT * FROM tle_data WHERE id = ?"
    with sqlite3.connect("database/satellites_tle.db") as conn:
        return conn.execute(query, (index,)).fetchone()

def get_satellite_count():
    """Get the total number of satellites in the database."""
    with sqlite3.connect("database/satellites_tle.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tle_data")
        return cursor.fetchone()[0]

def initialize_satellites(indexes):
    """Initialize satellites from the database."""
    return {idx: SatellitePropagator(fetch_tle_data(idx)) for idx in indexes}


def closest_distance(positions1, positions2):
    """Calculate the closest distance between two sets of positions."""
    distances = np.linalg.norm(positions1 - positions2, axis=1)
    min_idx = np.argmin(distances)
    return distances[min_idx], min_idx


def compute_closest_approaches(satellites, pairs):
    """Compute closest approaches in parallel."""
    def process_pair(pair_id, pair_info):
        sat1, sat2 = satellites[pair_info["sats"][0]], satellites[pair_info["sats"][1]]
        min_distance, min_idx = closest_distance(sat1.positions, sat2.positions)
        pair_info.update({"min_distance": min_distance, "min_distance_time": sat1.times[min_idx]})
        return pair_id, pair_info

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_pair, pid, pinfo): pid for pid, pinfo in pairs.items()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            pid, pinfo = future.result()
            pairs[pid] = pinfo
    return pairs


def detect_threats(pairs, threshold):
    """Detect potential threats based on minimum distances."""
    for pair in pairs.values():
        pair["close_approach"] = pair["min_distance"] < threshold
    logging.info(f"Total close approaches: {sum(p['close_approach'] for p in pairs.values())}")
    return pairs


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
    query = "SELECT * FROM orbital_data WHERE satellite_id = ?"
    conn = sqlite3.connect(db_path)
    orbital_data = pd.concat([pd.read_sql_query(query, conn, params=(idx,)) for idx in indexes], ignore_index=True)
    conn.close()

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
    """Save dataset to CSV."""
    df.to_csv(filename, index=False)

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

def run_sim(N_sats, days, step_size, error_km):
    total_sats = get_satellite_count()
    indexes = generate_unique_random_integers(total_sats, N_sats)
    satellites = initialize_satellites(indexes)
    pairs = generate_satellite_pair_dict(indexes, math.comb(N_sats, 2))
    
    propagate_orbits(satellites, 1440 * days, step_size)
    pairs = compute_closest_approaches(satellites, pairs)
    pairs = detect_threats(pairs, error_km * days)
    return satellites, pairs, indexes


if __name__ == "__main__":
    sats, pairs, indexes = run_sim(5, 1, 0.1, 2.0)
    df = build_dataset("database/satellites_tle.db", pairs, indexes)
    store_dataset(df, "datasets/tle_dataset.csv")
