import numpy as np

def kepler_equation(eccentricity, mean_anomaly, tolerance=1e-6):
    """
    Solves Kepler's equation for eccentric anomaly using the Newton-Raphson method.
    """
    f = lambda E: E - eccentricity * np.sin(E) - mean_anomaly
    f_prime = lambda E: 1 - eccentricity * np.cos(E)
    E = mean_anomaly  # Initial guess
    while abs(f(E)) > tolerance:
        E = E - f(E) / f_prime(E)
    return E

def compute_mean_anomaly(period, time):
    """
    Computes the mean anomaly at a given time.
    """
    return 2 * np.pi * (time % period) / period

def compute_true_anomaly(eccentricity, eccentric_anomaly):
    """
    Converts eccentric anomaly to true anomaly.
    """
    return 2 * np.arctan2(
        np.sqrt(1 + eccentricity) * np.sin(eccentric_anomaly / 2),
        np.sqrt(1 - eccentricity) * np.cos(eccentric_anomaly / 2)
    )

def compute_distance(semi_major_axis, eccentricity, true_anomaly):
    """
    Computes the radial distance from the focal point to the satellite.
    """
    return (semi_major_axis * (1 - eccentricity**2)) / (1 + eccentricity * np.cos(true_anomaly))

def orbital_to_cartesian(semi_major_axis, eccentricity, inclination, raan, arg_periapsis, true_anomaly):
    """
    Converts orbital elements to Cartesian position in 3D space.
    """
    # Distance from focus
    r = compute_distance(semi_major_axis, eccentricity, true_anomaly)

    # Orbital plane coordinates
    x_orb = r * np.cos(true_anomaly)
    y_orb = r * np.sin(true_anomaly)
    z_orb = 0  # In the orbital plane, z = 0

    # Rotation matrices for 3D transformation
    rotation_matrix = (
        np.array([
            [np.cos(raan) * np.cos(arg_periapsis) - np.sin(raan) * np.sin(arg_periapsis) * np.cos(inclination),
             -np.cos(raan) * np.sin(arg_periapsis) - np.sin(raan) * np.cos(arg_periapsis) * np.cos(inclination),
             np.sin(raan) * np.sin(inclination)],
            [np.sin(raan) * np.cos(arg_periapsis) + np.cos(raan) * np.sin(arg_periapsis) * np.cos(inclination),
             -np.sin(raan) * np.sin(arg_periapsis) + np.cos(raan) * np.cos(arg_periapsis) * np.cos(inclination),
             -np.cos(raan) * np.sin(inclination)],
            [np.sin(arg_periapsis) * np.sin(inclination),
             np.cos(arg_periapsis) * np.sin(inclination),
             np.cos(inclination)]
        ])
    )

    # Position in 3D space
    position_orbital = np.array([x_orb, y_orb, z_orb])
    position_3d = np.dot(rotation_matrix, position_orbital)
    return position_3d

def compute_position(period, eccentricity, semi_major_axis, inclination, raan, arg_periapsis, time):
    """
    Computes the satellite's 3D position in space.
    """
    # Mean anomaly
    mean_anomaly = compute_mean_anomaly(period, time)

    # Eccentric anomaly
    eccentric_anomaly = kepler_equation(eccentricity, mean_anomaly)

    # True anomaly
    true_anomaly = compute_true_anomaly(eccentricity, eccentric_anomaly)

    # Cartesian position in 3D space
    position = orbital_to_cartesian(semi_major_axis, eccentricity, inclination, raan, arg_periapsis, true_anomaly)
    return position


# Example Usage
if __name__ == "__main__":
	# Orbital elements (example values)
	period = 90  # Orbital period in minutes
	semi_major_axis = 20000  # Semi-major axis in km (Earth radius + altitude)
	eccentricity = 0.5  # Orbital eccentricity
	inclination = np.radians(90)  # Inclination in radians
	raan = np.radians(45)  # Right Ascension of Ascending Node in radians
	arg_periapsis = np.radians(30)  # Argument of periapsis in radians
		
	N = 100
    
	# Plot the orbit with Earth
    
	times = np.linspace(0, period, N)
    
	positions = np.array([
		compute_position(period, eccentricity, semi_major_axis, inclination, raan, arg_periapsis, t) 
    for t in times])