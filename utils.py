import numpy as np

def synodic_period(period1, period2):
	"""
	Computes the synodic period between two orbital periods.
	"""
	return 1 / np.abs(1 / period1 - 1 / period2)