import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

def star_shape(R=1.0, eps=0.3, n=5, points=5000):
    """Returns theta array and r(theta) for the star profile."""
    theta = np.linspace(0, 2*np.pi, points)
    r = R * (1 + eps * np.cos(n * theta))
    return theta, r

def area(theta, r):
    """Polar coordinate area = 1/2 ∫ r^2 dθ"""
    return 0.5 * simpson(r**2, theta)

def perimeter(theta, r):
    """Perimeter = ∫ sqrt(r^2 + (dr/dθ)^2) dθ"""
    dr = np.gradient(r, theta)
    integrand = np.sqrt(r**2 + dr**2)
    return simpson(integrand, theta)

def plot_star(theta, r):
    """Plot the star shape."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    plt.figure(figsize=(6,6))
    plt.plot(x, y)
    plt.axis("equal")
    plt.title("Star Grain Port Shape")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Parameters
    R = 1        # mean radius
    eps = 0.9      # tip amplitude
    n = 6          # number of points

    # Generate shape
    theta, r = star_shape(R, eps, n)

    # Compute geometry
    A = area(theta, r)
    P = perimeter(theta, r)
    AP_ratio = A / P

    print(f"Area     = {A:.6f}")
    print(f"Perimeter = {P:.6f}")
    print(f"A/P       = {AP_ratio:.6f}")

    # Plot
    plot_star(theta, r)
