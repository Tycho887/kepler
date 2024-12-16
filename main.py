import numpy
from orbit_computation import *
import matplotlib.pyplot as plt

class Satellite:
	def __init__(self, name: str, semi_major_axis, eccentricity, inclination, longitude_of_ascending_node, argument_perigee, mean_anomaly, epoch):
		self.name = name
		self.semi_major_axis = semi_major_axis
		self.eccentricity = eccentricity
		self.inclination = inclination
		self.argument_perigee = argument_perigee
		self.longitude_of_ascending_node = longitude_of_ascending_node
		self.mean_anomaly = mean_anomaly
		self.epoch = epoch

	def compute_position(self, time_limit, time_step=1):

		times = numpy.arange(0, time_limit, time_step)
		positions = np.zeros((len(times), 3))

		for i, time in enumerate(times):
			positions[i] = compute_position(self.mean_anomaly, self.eccentricity, self.semi_major_axis, self.inclination, self.longitude_of_ascending_node, self.argument_perigee, time)
		
		return positions

class SimulationPair:
	def __init__(self, satellite1, satellite2):
		self.satellite1 = satellite1
		self.satellite2 = satellite2