import numpy as np
from src.simulation import SatellitePropagator
from src.plotting import plot_3d, plot_distance
from src.database import fetch_tle_data
import numpy as np
from datetime import datetime
from src.utils import synodic_period


def simulate_orbit(tle_data, minutes_into_future, step_size):
	# Initialize the SatellitePropagator
	satellite = SatellitePropagator(tle_data)

	# Propagate the satellite's orbit
	try:
		result = satellite.propagate(minutes_into_future, step_size)
		print("Time:", result["times"])
		print("Position (km):", result["positions_km"])
		print("Velocity (km/s):", result["velocities_km_s"])
	except RuntimeError as e:
		print("Error:", e)
	
	return result["times"], result["positions_km"], result["velocities_km_s"]
	

def distance_over_time(tle_data1, tle_data2, minutes_into_future, step_size):
	# Simulate the orbits of the two satellites
	times1, positions1, velocities1 = simulate_orbit(tle_data1, minutes_into_future, step_size)
	times2, positions2, velocities2 = simulate_orbit(tle_data2, minutes_into_future, step_size)
	
	# Calculate the distance between the two satellites at each time step
	distances = np.linalg.norm(np.array(positions1) - np.array(positions2), axis=1)
	
	# Plot the distance between the two satellites over time
	plot_distance(times1, distances)

if __name__ == "__main__":

	ISS = fetch_tle_data(name="ISS (ZARYA)")
	CSS = fetch_tle_data(name="CSS (TIANHE)")
	TESS = fetch_tle_data(name="TESS")

	# Calculate the synodic period between the two satellites

	synodic_period_minutes = synodic_period(CSS, ISS)

	print("Synodic Period:", synodic_period_minutes, "minutes")

	distance_over_time(CSS, ISS, minutes_into_future=synodic_period_minutes, step_size=1)