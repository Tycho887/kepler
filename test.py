import numpy as np
import matplotlib.pyplot as plt

def keplerian_orbit(a, e, theta):
    """
    Calculate the radial distance r of a Keplerian orbit for a given angle theta.

    Parameters:
        a (float): Semi-major axis of the orbit
        e (float): Eccentricity of the orbit
        theta (numpy array): Array of angles (in radians)

    Returns:
        r (numpy array): Radial distances corresponding to the angles
    """
    r = (a * (1 - e**2)) / (1 + e * np.cos(theta))
    return r

# Define parameters of the orbit
a = 1.0  # Semi-major axis
E = 0.9  # Eccentricity

# Generate angles for the orbit (0 to 2*pi)
theta = np.linspace(0, 2 * np.pi, 500)

# Calculate radial distances
r = keplerian_orbit(a, E, theta)

# Create the polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r, label=f"a={a}, e={E}")
ax.set_title("Keplerian Orbit in Polar Coordinates")
ax.legend()

# Display the plot
plt.show()
