import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from typing import Optional


def load_clustered_data(db_name: str) -> Optional[np.ndarray]:
    """Load clustered data from the database."""
    try:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT semi_major_axis, inclination, raan, cluster
                FROM orbital_data
            """)
            data = cursor.fetchall()
            return np.array(data)
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None

def visualize_clusters_matplotlib(data: np.ndarray):
    """Visualize clusters using Matplotlib."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract features and clusters
    semi_major_axis = data[:, 0]
    inclination = data[:, 1]
    raan = data[:, 2]
    clusters = data[:, 3]

    # Plot each cluster
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        mask = clusters == cluster
        ax.scatter(
            semi_major_axis[mask], inclination[mask], raan[mask],
            label=f"Cluster {int(cluster)}", s=50
        )

    # Set labels
    ax.set_xlabel("Semi-major Axis (km)")
    ax.set_ylabel("Inclination (rad)")
    ax.set_zlabel("RAAN (rad)")
    ax.set_title("Satellite Clusters (3D Visualization)")
    ax.legend()

    plt.show()

def main():
    """Main function to load and visualize clustered data."""
    db_name = "database/satellites_tle.db"
    data = load_clustered_data(db_name)
    if data is None:
        print("No clustered data found. Exiting.")
        return

    visualize_clusters_matplotlib(data)

if __name__ == "__main__":
    main()