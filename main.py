import numpy as np
from src.simulation import compute_position
from src.plotting import plot_3d
from src.database import fetch_tle_data


class Satellite:
	def __init__(self, tle_data: tuple):
		self.name = tle_data[1]
		self.tle_line1 = tle_data[2]
		self.tle_line2 = tle_data[3]

	def get_keplerian(self):
		pass

	def compute_position(self, time_limit, time_step=1):

		times = np.arange(0, time_limit, time_step)
		positions = np.zeros((len(times), 3))

		for i, time in enumerate(times):
			positions[i] = compute_position(self.mean_anomaly, self.eccentricity, self.semi_major_axis, self.inclination, self.longitude_of_ascending_node, self.argument_perigee, time)
		
		return positions

class SimulationPair:
	def __init__(self, satellite1, satellite2):
		self.satellite1 = satellite1
		self.satellite2 = satellite2


if __name__ == "__main__":
	tle_ISS = fetch_tle_data(name="ISS (ZARYA)")
	tle_Hubble = fetch_tle_data(name="HST")

	iss = Satellite(tle_ISS)
	hubble = Satellite(tle_Hubble)