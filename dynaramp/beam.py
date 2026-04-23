import numpy as np
import scipy
from scipy.optimize import fsolve
from scipy import integrate
from matplotlib import pyplot as plt


class EulerBernoulliBeam:
    """
    Euler-Bernoulli beam theory for a homogenous cantilevered beam
    E and I are independent of x.

    EI * d^4w/dx^2 = - mu * d^2w/dt^2 + q(x)
    """
    def __init__(self, lin_mass, e_modulus, second_moment, length):
        self.mu = lin_mass # linear mass density, kg/m
        self.E = e_modulus # Elastic modulus, Pa
        self.I = second_moment # Second moment of area, m^4
        self.L = length  # Length of the beam, m

        self._roots = []  # Cache for the first roots of the characteristic equation

    @property
    def betas(self):
        """ Returns all cached values of beta = root/L"""
        return np.array(self._roots)/self.L

    @property
    def omegas(self):
        """ Returns all cached values of natural frequencies (rad/s)"""
        return self.betas ** 2 * np.sqrt(self.E * self.I / self.mu)

    @staticmethod
    def characteristic_equation(x):
        # x = beta * L
        return np.cosh(x) * np.cos(x) + 1

    def _compute_nth_root(self, n):
        # Compute the nth root of the characteristic equation
        x0 = (2*n - 1) * np.pi/2  # Initial guess for a cantilevered beam
        x_n = float(fsolve(self.characteristic_equation, x0)[0])

        self._roots.append(x_n)

    def modes(self, n_start, n_modes):

        # If neccessary compute more natural frequencies
        n_cache = len(self._roots)
        n_stop = n_start + n_modes - 1
        if n_stop > n_cache:
            for n in range(n_stop - n_cache):
                self._compute_nth_root(n_cache + n + 1)

        # Natural frequencies (rad/s)
        ws = self.omegas[n_start-1:n_stop]

        # Mode shapes (normalization factor = 1)
        phis = []
        for beta in self.betas[n_start-1:n_stop]:
            phi_i = lambda x, b_i=beta :(
                    1 * ((np.cosh(b_i * x) - np.cos(b_i * x))
                         - (np.cos(b_i*self.L) + np.cosh(b_i*self.L)) / (np.sin(b_i*self.L) + np.sinh(b_i*self.L))
                         * (np.sinh(b_i * x) - np.sin(b_i * x)))
            )

            phis.append(phi_i)

        return ws, phis

    def modal_matrices(self, n_modes=None):
        # If neccessary compute more natural frequencies, else used cached
        n_cache = len(self._roots)
        if n_modes is not None:
            if n_modes > n_cache:
                for n in range(n_modes - n_cache):
                    self._compute_nth_root(n_cache + n + 1)
        else:
            n_modes = n_cache

        m_mat = np.zeros((n_modes, n_modes))
        k_mat = np.zeros((n_modes, n_modes))

        # M-K mode orthagonality insures that M and K are diagonal
        # m_ii = int_{O}^{L} mu * phi_i^2(x) dx
        # k_ii = w_i * m_ii
        # Moreover, int_{O}^{L} phi_i^2(x) dx = L, therefore, m_ii = mass

        m_mat = self.mu * self.L * np.eye(n_modes)
        k_mat = np.diag(m_mat @ (self.omegas[:n_modes]**2))

        return m_mat, k_mat

if __name__ == "__main__":
    # Steel rectangular beam
    b = 1e-2
    h = 2e-2
    A = b * h
    rho = 7850
    mu = rho*A  # linear mass density, kg/m
    E = 210e9  # Elastic modulus, Pa
    I = b * h ** 3 / 12  # Second moment of area, m^4
    L = 1  # Length of the beam, m

    Beam = EulerBernoulliBeam(mu, E, I, L)
    ws, phis = Beam.modes(1, 4)
    M, K = Beam.modal_matrices(4)

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
