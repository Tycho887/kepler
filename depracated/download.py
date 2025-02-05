import requests
import sqlite3
import json
import os
from datetime import datetime

# Constants
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"  # Active satellites TLE

storage_dir = "database/"

DB_NAME = storage_dir + "satellites_tle.db"
# JSON_FILE = storage_dir + "satellites_tle.json"

import sqlite3

db_name = "database/satellites_tle.db"

db_structure = {
    "sqlite_sequence": {
        "name": "TEXT", 
        "seq": "INTEGER"
    },
    "tle_data": {
        "id": "INTEGER PRIMARY KEY",
        "name": "TEXT",
        "line1": "TEXT",
        "line2": "TEXT",
        "timestamp": "TEXT"},
	"orbital_data": {
        "id": "INTEGER PRIMARY KEY",
        "satellite_id": "INTEGER",
        "epoch": "TEXT",
        "mean_motion": "REAL",
        "eccentricity": "REAL",
        "inclination": "REAL",
		"perigee": "REAL",
		"apogee": "REAL",
		"period": "REAL",
    }
}

# We create a decorator function that handles the connection to the database and closes it after the function is executed.
def connect(function):
    def wrapper(*args, **kwargs):
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        result = function(cursor, *args, **kwargs)
        conn.commit()
        conn.close()
        return result
    return wrapper


@connect
def fetch_tle_data_single(cursor, id):

	id = int(id)

	cursor.execute("SELECT * FROM tle_data WHERE id = ?", (id,))

	return cursor.fetchone()

@connect
def fetch_tle_data_multiple(cursor, ids: list):

	ids = tuple(map(int, ids))

	cursor.execute("SELECT * FROM tle_data WHERE id IN {}".format(ids))

	return cursor.fetchall()

@connect
def number_of_satellites(cursor):
	cursor.execute("SELECT COUNT(*) FROM tle_data")

	return cursor.fetchone()[0]

def fetch_tle_data(url):
    """
    Fetch TLE data from the provided URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching TLE data: {e}")
        return None

def parse_tle_data(tle_text):
    """
    Parse TLE data into a structured list of dictionaries.
    Each satellite entry contains the name, line1, and line2.
    """
    lines = tle_text.strip().split("\n")
    satellites = []

    for i in range(0, len(lines) - 2, 3):  # TLE format: name, line1, line2
        name = lines[i].strip()
        line1 = lines[i+1].strip()
        line2 = lines[i+2].strip()
        satellites.append({
            "name": name,
            "line1": line1,
            "line2": line2
        })
    return satellites

def save_to_sqlite(data, db_name):
    """
    Save parsed TLE data to a SQLite database.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tle_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                line1 TEXT,
                line2 TEXT,
                timestamp TEXT
            )
        """)

        # Insert data into the table
        timestamp = datetime.utcnow().isoformat()
        for satellite in data:
            cursor.execute("""
                INSERT INTO tle_data (name, line1, line2, timestamp)
                VALUES (?, ?, ?, ?)
            """, (satellite["name"], satellite["line1"], satellite["line2"], timestamp))

        conn.commit()
        conn.close()
        print(f"TLE data saved to {db_name}")
    except Exception as e:
        print(f"Error saving to SQLite: {e}")


def get_TLE_info(TLE_data):
    """
    Extracts orbital parameters and other coefficients from TLE data.
    
    Parameters:
        TLE_data (tuple): Tuple containing satellite name, international designator, and two-line elements.

    Returns:
        dict: Dictionary containing orbital parameters and coefficients.
    """
    # Extract the relevant info for constructing the dataset

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
    # Calculate the orbital period
    orbital_period = 2 * 3.141592653589793 * (semi_major_axis ** 1.5) / (mean_motion * 86400)

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
        "orbital period": orbital_period
    }



def main():
    print("Starting TLE data fetch and storage...")

    # Step 1: Fetch TLE data
    tle_text = fetch_tle_data(TLE_URL)
    if not tle_text:
        print("No TLE data fetched. Exiting.")
        return

    # Step 2: Parse TLE data
    satellites = parse_tle_data(tle_text)
    print(f"Fetched and parsed data for {len(satellites)} satellites.")

    # Step 3: Save to SQLite
    save_to_sqlite(satellites, DB_NAME)

if __name__ == "__main__":
    main()
