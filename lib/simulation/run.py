from .SGP4 import *

# def distance_over_time(tle_data1, tle_data2, minutes_into_future, step_size):
# 	# Simulate the orbits of the two satellites
# 	times1, positions1, velocities1 = simulate_orbit(tle_data1, minutes_into_future, step_size)
# 	times2, positions2, velocities2 = simulate_orbit(tle_data2, minutes_into_future, step_size)
	
# 	# Calculate the distance between the two satellites at each time step
# 	distances = np.linalg.norm(np.array(positions1) - np.array(positions2), axis=1)
# 	# minimum distance and relative velocity
# 	min_distance = np.min(distances)
# 	min_distance_index = np.argmin(distances)
# 	min_distance_time = times1[min_distance_index]
# 	min_distance_velocity = np.linalg.norm(np.array(velocities1[min_distance_index]) - np.array(velocities2[min_distance_index]))
	
# 	print("Minimum Distance:", min_distance, "km")
# 	print("Time of Minimum Distance:", min_distance_time)
# 	print("Relative Velocity at Minimum Distance:", min_distance_velocity, "km/s")

# 	# Plot the distance between the two satellites over time
# 	plot_distance(times1, distances)