import requests
import sqlite3
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple
from skyfield.api import EarthSatellite, load
import math
import tqdm
import concurrent.futures

# Constants
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
STORAGE_DIR = "database/"
DB_NAME = STORAGE_DIR + "satellites_tle.db"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

orbital_data_structure = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "satellite_id": "INTEGER",
    "epoch": "TEXT",
    "mean_motion": "REAL",
    "eccentricity": "REAL",
    "inclination": "REAL",
    "raan": "REAL",
    "argument_of_perigee": "REAL",
    "mean_anomaly": "REAL",
    "perigee": "REAL",
    "apogee": "REAL",
    "period": "REAL",
    "semi_major_axis": "REAL",
    "orbital_energy": "REAL",
    "orbital_angular_momentum": "REAL",
    "rev_number": "INTEGER",
    "bstar": "REAL",
    "mean_motion_derivative": "REAL",
    "cluster": "INTEGER"
}

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
            columns_definition = ",\n    ".join(
                [f"{col} {datatype}" for col, datatype in orbital_data_structure.items()]
            )
            
            query = f"""
                CREATE TABLE IF NOT EXISTS orbital_data (
                    {columns_definition},
                    FOREIGN KEY (satellite_id) REFERENCES tle_data (id)
                )
            """
            
            cursor.execute(query)
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
        """Bulk insert orbital data into the database."""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            column_names = [col for col in orbital_data_structure.keys() if col != "id"]
            placeholders = ", ".join(["?"] * len(column_names))
            column_list = ", ".join(column_names)

            cursor.executemany(f"""
                INSERT INTO orbital_data ({column_list})
                VALUES ({placeholders})
            """, [tuple(data[col] for col in column_names) for data in orbital_data])

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

    ts = load.timescale()  # Load time scale once (avoid per-call overhead)

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
    
    # def compute_orbital_parameters_parallel(satellites):
    #     """Compute orbital parameters in parallel."""
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         results = list(tqdm.tqdm(executor.map(
    #             lambda sat: {"satellite_id": sat[0], **TLEProcessor.extract_orbital_parameters(sat[1], sat[2])},
    #             satellites), total=len(satellites), desc="Parallel Processing"))
    #     return results

    @staticmethod
    def extract_orbital_parameters(line1: str, line2: str) -> Dict:
        """Extract orbital parameters from TLE lines using Skyfield."""
        satellite = EarthSatellite(line1, line2, "Satellite", TLEProcessor.ts)

        # Extract parameters
        model = satellite.model
        inclination, raan, eccentricity = model.inclo, model.nodeo, model.ecco
        argument_of_perigee, mean_anomaly, bstar = model.argpo, model.mo, model.bstar
        mean_motion = float(line2[52:63])
        
        # Constants
        mu = 3.986004418e14  # Standard gravitational parameter (m^3/s^2)
        earth_radius_km = 6371.0

        # Compute semi-major axis
        mean_motion_rad_s = mean_motion * 2 * math.pi / 86400
        semi_major_axis = (mu / (mean_motion_rad_s ** 2)) ** (1 / 3) / 1000

        # Compute perigee and apogee
        apogee = semi_major_axis * (1 + eccentricity) - earth_radius_km
        perigee = semi_major_axis * (1 - eccentricity) - earth_radius_km

        # Compute orbital period (hours)
        period = (2 * math.pi * (semi_major_axis * 1000) ** 1.5) / (mu ** 0.5) / 3600

        # Compute orbital energy and angular momentum
        orbital_energy = -mu / (2 * semi_major_axis * 1000)
        orbital_angular_momentum = (mu * semi_major_axis * 1000 * (1 - eccentricity ** 2)) ** 0.5

        # Avoid unnecessary Skyfield calls
        epoch = line1[18:32]
        rev_number = int(line2[63:68])
        mean_motion_derivative = float(line1[33:43])

        return {
            "epoch": epoch, "mean_motion": mean_motion, "eccentricity": eccentricity,
            "inclination": inclination, "raan": raan, "argument_of_perigee": argument_of_perigee,
            "mean_anomaly": mean_anomaly, "perigee": perigee, "apogee": apogee,
            "period": period, "semi_major_axis": semi_major_axis, "orbital_energy": orbital_energy,
            "orbital_angular_momentum": orbital_angular_momentum, "rev_number": rev_number,
            "bstar": bstar, "mean_motion_derivative": mean_motion_derivative, "cluster": 0
        }


def compute_orbital_parameters_parallel(satellites):
    """Compute orbital parameters in parallel."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(
            lambda sat: {"satellite_id": sat[0], **TLEProcessor.extract_orbital_parameters(sat[1], sat[2])},
            satellites), total=len(satellites), desc="Parallel Processing"))
    return results

def main():
    """Main function to fetch, process, and store TLE data."""
    logging.info("Starting TLE data fetch and storage...")

    # Initialize database
    db_handler = DatabaseHandler(DB_NAME)

    if not db_handler.tle_data_exists():
        logging.info("Fetching new TLE data...")
        tle_text = TLEProcessor.fetch_tle_data(TLE_URL)
        if not tle_text:
            logging.error("No TLE data fetched. Exiting.")
            return

        satellites = TLEProcessor.parse_tle_data(tle_text)
        logging.info(f"Fetched and parsed {len(satellites)} satellites.")

        db_handler.insert_tle_data(satellites)
    else:
        logging.info("Using existing TLE data.")

    # Fetch satellite data from database
    satellites = db_handler.execute_query("SELECT id, line1, line2 FROM tle_data")
    if not satellites:
        logging.error("No TLE data found. Exiting.")
        return

    logging.info(f"Processing {len(satellites)} TLE records...")

    # Process all TLEs efficiently using list comprehension
    orbital_data = compute_orbital_parameters_parallel(satellites)

    # Bulk insert into DB
    db_handler.insert_orbital_data(orbital_data)
    logging.info("Processing complete.")



if __name__ == "__main__":
    main()