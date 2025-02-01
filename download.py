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

def save_to_json(data, file_name):
    """
    Save parsed TLE data to a JSON file.
    """
    try:
        with open(file_name, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"TLE data saved to {file_name}")
    except Exception as e:
        print(f"Error saving to JSON: {e}")

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

    # # Step 3: Save to JSON
    # save_to_json(satellites, JSON_FILE)

    # Step 4: Save to SQLite
    save_to_sqlite(satellites, DB_NAME)

if __name__ == "__main__":
    main()
