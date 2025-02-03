import requests
import sqlite3
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple
from skyfield.api import EarthSatellite, load
import math
import tqdm

# Constants
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
STORAGE_DIR = "database/"
DB_NAME = STORAGE_DIR + "satellites_tle.db"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DatabaseHandler:
    """Handles database connections and operations."""

    def __init__(self, db_name: str):
        self.db_name = db_name
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            # Create TLE data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tle_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    line1 TEXT,
                    line2 TEXT,
                    timestamp TEXT
                )
            """)
            # Create orbital data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orbital_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    satellite_id INTEGER,
                    epoch TEXT,
                    mean_motion REAL,
                    eccentricity REAL,
                    inclination REAL,
                    raan REAL,
                    argument_of_perigee REAL,
                    mean_anomaly REAL,
                    perigee REAL,
                    apogee REAL,
                    period REAL,
                    semi_major_axis REAL,
                    orbital_energy REAL,
                    orbital_angular_momentum REAL,
                    rev_number INTEGER,
                    bstar REAL,
                    mean_motion_derivative REAL,
                    cluster INTEGER,
                    FOREIGN KEY (satellite_id) REFERENCES tle_data (id)
                )
            """)
            conn.commit()

    def execute_query(self, query: str, params: Tuple = ()) -> Optional[List]:
        """Execute a SQL query and return the result."""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            return None

    def insert_tle_data(self, data: List[Dict]):
        """Insert TLE data into the database."""
        timestamp = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            for satellite in data:
                cursor.execute("""
                    INSERT INTO tle_data (name, line1, line2, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (satellite["name"], satellite["line1"], satellite["line2"], timestamp))
            conn.commit()
            logging.info(f"Inserted {len(data)} TLE records into the database.")

    def insert_orbital_data(self, orbital_data: List[Dict]):
        """Insert orbital data into the database."""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            for data in orbital_data:
                cursor.execute("""
                    INSERT INTO orbital_data (
                        satellite_id, epoch, mean_motion, eccentricity, inclination,
                        raan, argument_of_perigee, mean_anomaly, perigee, apogee, period, semi_major_axis, orbital_energy, orbital_angular_momentum,
                        rev_number, bstar, mean_motion_derivative, cluster
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data["satellite_id"], data["epoch"], data["mean_motion"], data["eccentricity"],
                    data["inclination"], data["raan"], data["argument_of_perigee"], data["mean_anomaly"],
                    data["perigee"], data["apogee"], data["period"], data["semi_major_axis"], data["orbital_energy"], data["orbital_angular_momentum"], 
                    data["rev_number"], data["bstar"], data["mean_motion_derivative"], data["cluster"]
                ))
            conn.commit()
            logging.info(f"Inserted {len(orbital_data)} orbital records into the database.")

    def tle_data_exists(self) -> bool:
        """Check if TLE data exists in the database."""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tle_data")
            count = cursor.fetchone()[0]
            return count > 0


class TLEProcessor:
    """Processes TLE data and extracts orbital parameters."""

    @staticmethod
    def fetch_tle_data(url: str) -> Optional[str]:
        """Fetch TLE data from the provided URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.error(f"Error fetching TLE data: {e}")
            return None

    @staticmethod
    def parse_tle_data(tle_text: str) -> List[Dict]:
        """Parse TLE data into a structured list of dictionaries."""
        lines = tle_text.strip().split("\n")
        satellites = []
        for i in range(0, len(lines) - 2, 3):  # TLE format: name, line1, line2
            satellites.append({
                "name": lines[i].strip(),
                "line1": lines[i + 1].strip(),
                "line2": lines[i + 2].strip()
            })
        return satellites

    @staticmethod
    def extract_orbital_parameters(line1: str, line2: str) -> Dict:
        """Extract orbital parameters from TLE lines using skyfield."""
        # Create a satellite object from the TLE data
        satellite = EarthSatellite(line1, line2, "Satellite", load.timescale())

        # Extract orbital parameters
        inclination = satellite.model.inclo  # Inclination in radians
        raan = satellite.model.nodeo  # RAAN in radians
        eccentricity = satellite.model.ecco  # Eccentricity
        argument_of_perigee = satellite.model.argpo  # Argument of perigee in radians
        mean_anomaly = satellite.model.mo  # Mean anomaly in radians
        mean_motion = float(line2[52:63])  # Revolutions per day
        bstar = satellite.model.bstar  # Drag coefficient (B*)

        # Calculate additional parameters
        earth_radius_km = 6371.0
        mu = 3.986004418e14  # Standard gravitational parameter (m^3/s^2)

        # Semi-major axis (in km)
        semi_major_axis = (mu / ((mean_motion * 2 * math.pi / 86400) ** 2)) ** (1 / 3) / 1000
        # Apogee and perigee (in km)
        apogee = semi_major_axis * (1 + eccentricity) - earth_radius_km
        perigee = semi_major_axis * (1 - eccentricity) - earth_radius_km

        # Orbital period (in hours)
        period = (2 * math.pi * (semi_major_axis * 1000) ** 1.5) / (mu ** 0.5) / 3600

        # Orbital energy and angular momentum
        orbital_energy = -mu / (2 * semi_major_axis * 1000)
        orbital_angular_momentum = (mu * semi_major_axis * 1000 * (1 - eccentricity ** 2)) ** 0.5

        # True anomaly (using skyfield's built-in functions)
        ts = load.timescale()
        t = ts.now()
        r, v = satellite.at(t).position.km, satellite.at(t).velocity.km_per_s

        return {
            "epoch": line1[18:32],
            "mean_motion": mean_motion,
            "eccentricity": eccentricity,
            "inclination": inclination,
            "raan": raan,
            "argument_of_perigee": argument_of_perigee,
            "mean_anomaly": mean_anomaly,
            "perigee": perigee,
            "apogee": apogee,
            "period": period,
            "semi_major_axis": semi_major_axis,
            "orbital_energy": orbital_energy,
            "orbital_angular_momentum": orbital_angular_momentum,
            "rev_number": int(line2[63:68]),
            "bstar": bstar,
            "mean_motion_derivative": float(line1[33:43]),
            "cluster": 0  # Placeholder for clustering
        }


def main():
    """Main function to fetch, process, and store TLE data."""
    logging.info("Starting TLE data fetch and storage...")

    # Initialize database handler
    db_handler = DatabaseHandler(DB_NAME)

    # Check if TLE data already exists in the database
    if not db_handler.tle_data_exists():
        logging.info("TLE data not found in database. Fetching from API...")
        # Fetch TLE data
        tle_text = TLEProcessor.fetch_tle_data(TLE_URL)
        if not tle_text:
            logging.error("No TLE data fetched. Exiting.")
            return

        # Parse TLE data
        satellites = TLEProcessor.parse_tle_data(tle_text)
        logging.info(f"Fetched and parsed data for {len(satellites)} satellites.")

        # Save TLE data to database
        db_handler.insert_tle_data(satellites)
    else:
        logging.info("TLE data already exists in the database. Skipping API request.")

    # Fetch TLE data from the database
    satellites = db_handler.execute_query("SELECT name, line1, line2 FROM tle_data")
    if not satellites:
        logging.error("No TLE data found in the database. Exiting.")
        return

    # Extract and save orbital data
    orbital_data = []
    for satellite in tqdm.tqdm(satellites, desc="Processing TLE data"):
        params = TLEProcessor.extract_orbital_parameters(satellite[1], satellite[2])
        orbital_data.append({
            "satellite_id": satellite[0],  # Assuming name is unique; use ID if available
            **params
        })
    db_handler.insert_orbital_data(orbital_data)

    logging.info("TLE and orbital data processing complete.")


if __name__ == "__main__":
    main()