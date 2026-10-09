import matplotlib.pyplot as plt
import scipy.special as special
import numpy as np

n = 1
k = 1

def harmonic_probe_amp(x, y):
	"""Compute a harmonic probe flow with given wave numbers at time t.

	This function evaluates the harmonic probe flow at the current
	vortex positions `(self.xs, self.ys)` and time `t`. The flow is
	separable in polar coordinates and uses Bessel functions of order
	`n` with the `k`-th zero to define the radial structure.
	"""
	alpha = special.jnp_zeros(n, k)[-1]  # Get the k-th zero of the Bessel function of order n
	r = np.sqrt(x**2 + y**2)
	angles = np.arctan2(y, x)
	J_n = special.jv(n, alpha * r)
	J_n_prime = special.jvp(n, alpha * r)
	du_dr = alpha * J_n_prime * np.cos(n * angles)
	if n == 0:
		du_dtheta = np.zeros_like(du_dr)
	else:
		du_dtheta = -n * J_n * np.sin(n * angles)
	du_dx = du_dr * x / r - du_dtheta * y / r**2
	du_dy = du_dr * y / r + du_dtheta * x / r**2
	return du_dx, du_dy

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
A = harmonic_probe_amp(X, Y)

plt.imshow(np.sqrt(A[0]**2 + A[1]**2), cmap='viridis', extent=(-1, 1, -1, 1))
plt.show()