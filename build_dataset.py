import sqlite3
import pandas as pd
from lib import run_sim

def query_orbital_data(db_path):
    """
    Query the orbital_data table from the database.
    
    Parameters:
        db_path (str): Path to the SQLite database.
    
    Returns:
        pd.DataFrame: DataFrame containing the orbital data.
    """
    conn = sqlite3.connect(db_path)
    query = """
    SELECT satellite_id, epoch, mean_motion, eccentricity, inclination, raan, 
           argument_of_perigee, mean_anomaly, perigee, apogee, period, 
           semi_major_axis, orbital_energy, orbital_angular_momentum, rev_number, 
           bstar, mean_motion_derivative, cluster
    FROM orbital_data
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
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
    differences = {
        "inclination_diff": row1["inclination"] - row2["inclination"],
        "eccentricity_diff": row1["eccentricity"] - row2["eccentricity"],
        "raan_diff": row1["raan"] - row2["raan"],
        "argument_of_perigee_diff": row1["argument_of_perigee"] - row2["argument_of_perigee"],
        "mean_anomaly_diff": row1["mean_anomaly"] - row2["mean_anomaly"],
        "mean_motion_diff": row1["mean_motion"] - row2["mean_motion"],
        "apogee_diff": row1["apogee"] - row2["apogee"],
        "perigee_diff": row1["perigee"] - row2["perigee"],
        "period_diff": row1["period"] - row2["period"],
        "semi_major_axis_diff": row1["semi_major_axis"] - row2["semi_major_axis"],
        "orbital_energy_diff": row1["orbital_energy"] - row2["orbital_energy"],
        "orbital_angular_momentum_diff": row1["orbital_angular_momentum"] - row2["orbital_angular_momentum"],
        "rev_number_diff": row1["rev_number"] - row2["rev_number"],
        "bstar_diff": row1["bstar"] - row2["bstar"],
        "mean_motion_derivative_diff": row1["mean_motion_derivative"] - row2["mean_motion_derivative"],
        "cluster_diff": row1["cluster"] - row2["cluster"]
    }
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
    orbital_data = query_orbital_data(db_path)
    data = []

    for i, _dict in pairs.items():
        idx1, idx2 = _dict["sats"]

        # Check if satellite IDs exist in the orbital_data table
        if idx1 not in orbital_data["satellite_id"].values:
            print(f"Warning: satellite_id {idx1} not found in orbital_data table. Skipping pair {i}.")
            continue
        if idx2 not in orbital_data["satellite_id"].values:
            print(f"Warning: satellite_id {idx2} not found in orbital_data table. Skipping pair {i}.")
            continue

        sat1_data = orbital_data[orbital_data["satellite_id"] == idx1].iloc[0]
        sat2_data = orbital_data[orbital_data["satellite_id"] == idx2].iloc[0]

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
    N = 10
    days = 10
    step_size = 0.1  # Step size in minutes
    daily_error = 2.0  # Error in kilometers per day

    sats, pairs, indexes = run_sim(N, days, step_size, daily_error)

    df = build_dataset(db_path, pairs, indexes)

    store_dataset(df, f"datasets/tle{len(df)}.csv")