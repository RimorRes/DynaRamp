from dynaramp.beam import EulerBernoulliBeam
import numpy as np
from matplotlib import pyplot as plt

# TODO: test first beam natural frequencies against know
# TODO: test mode shape = +/- 2 at x=L
# TODO: test mode shape = 0 at x=0
# TODO: test mode orthogonality; M-K diagonal
# TODO: test phi^2 integral = L

def make_beam():
	# Steel rectangular beam
	b = 1e-2
	h = 2e-2
	A = b * h
	rho = 7850
	mu = rho * A  # linear mass density, kg/m
	E = 210e9  # Elastic modulus, Pa
	I = b * h ** 3 / 12  # Second moment of area, m^4
	L = 1  # Length of the beam, m

	return  EulerBernoulliBeam(mu, E, I, L)

Beam = make_beam()
ws, phis = Beam.modes(1, 4)
M, C, K = Beam.modal_matrices(4)

print("Natural frequencies (rad/s):", ws)
xs = np.linspace(0.0, L, 100)

print("M", M)
print("K", K)

for i, phi in enumerate(phis, start=1):
	ys = phi(xs)
	plt.plot(xs, ys, label=f"Mode {i}")

plt.xlabel("x (m)")
plt.ylabel("Mode shape (normalized)")
plt.legend()
plt.show()
