import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_orbit_3d_with_earth(times, positions):
    """
    Plots the orbit of a satellite in 3D, including a textured Earth visualization.
    
    :param period: Orbital period in minutes
    :param semi_major_axis: Semi-major axis in km
    :param eccentricity: Orbital eccentricity
    :param inclination: Orbital inclination in radians
    :param raan: Right Ascension of Ascending Node in radians
    :param arg_periapsis: Argument of periapsis in radians
    :param num_points: Number of points to plot along the orbit
    """

    # Extract x, y, z coordinates
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Plot the orbit in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the orbit
    ax.plot(x, y, z, label='Orbit', lw=2)

    # Add the Earth as a sphere
    earth_radius = 6371  # Earth's radius in km
    u = np.linspace(0, 2 * np.pi, 100)  # Longitude
    v = np.linspace(0, np.pi, 100)     # Latitude
    x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_radius * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(
        x_earth, y_earth, z_earth, rstride=4, cstride=4, color='blue', alpha=0.6, edgecolor='none'
    )

    # Set labels and aspect
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Satellite Orbit Around Earth')
    ax.legend()
    ax.grid(True)

    # Set aspect ratio for equal scaling
    max_val = max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z)))
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    plt.show()


def plot_distance(times: list, distances: np.ndarray):
    """
    Plots the distance between two satellites over time.
    
    :param times: Array of datetime objects representing the time at each step
    :param positions_1: Array of position vectors for satellite 1
    :param positions_2: Array of position vectors for satellite 2
    """

    # Plot the distance between the two satellites over time
    plt.figure(figsize=(10, 6))
    plt.plot(times, distances)
    plt.xlabel('Time (UTC)')
    plt.ylabel('Distance (km)')
    plt.title('Distance Between Satellites Over Time')
    plt.grid(True)
    plt.show()