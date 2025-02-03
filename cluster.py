import sqlite3
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging
from typing import Optional

DB_NAME = "database/satellites_tle.db"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CollisionAwareClustering:
    """Handles clustering analysis of satellite orbital data to minimize collision risk."""

    def __init__(self, db_name: str):
        self.db_name = db_name

    def load_orbital_data(self) -> Optional[np.ndarray]:
        """Load orbital data from the database."""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        semi_major_axis, inclination, raan, eccentricity
                    FROM orbital_data
                """)
                data = cursor.fetchall()
                return np.array(data)
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            return None

    def perform_clustering(self, data: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Optional[np.ndarray]:
        """Perform DBSCAN clustering on the orbital data."""
        try:
            # Normalize the data
            scaler = StandardScaler()
            data_normalized = scaler.fit_transform(data)

            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(data_normalized)
            return clusters
        except Exception as e:
            logging.error(f"Clustering error: {e}")
            return None

    def save_clusters_to_db(self, clusters: np.ndarray):
        """Save the cluster labels back to the database."""
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                for i, cluster in enumerate(clusters):
                    cursor.execute("""
                        UPDATE orbital_data
                        SET cluster = ?
                        WHERE id = ?
                    """, (int(cluster), i + 1))  # Assuming id starts from 1
                conn.commit()
                logging.info(f"Saved {len(clusters)} cluster labels to the database.")
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")

def main():
    """Main function to perform collision-aware clustering analysis on satellite data."""
    logging.info("Starting collision-aware clustering analysis...")

    # Initialize clustering analysis
    clustering = CollisionAwareClustering(DB_NAME)

    # Load orbital data
    orbital_data = clustering.load_orbital_data()
    if orbital_data is None:
        logging.error("No orbital data found. Exiting.")
        return

    # Perform clustering
    clusters = clustering.perform_clustering(orbital_data, eps=0.5, min_samples=5)
    if clusters is None:
        logging.error("Clustering failed. Exiting.")
        return

    # Save clusters to the database
    clustering.save_clusters_to_db(clusters)

    logging.info("Collision-aware clustering analysis complete.")


if __name__ == "__main__":
    main()