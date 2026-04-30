import numpy as np
import jax.numpy as jnp
from scipy.optimize import fsolve


class EulerBernoulliBeam:
    """
    Euler-Bernoulli beam theory for a homogenous cantilevered beam
    E and I are independent of x.

    EI * d^4w/dx^2 = - mu * d^2w/dt^2 + q(x)
    """
    def __init__(self, lin_mass, e_modulus, second_moment, length, r_alpha=0.0, r_beta=0.0):
        self.mu = lin_mass # linear mass density, kg/m
        self.E = e_modulus # Elastic modulus, Pa
        self.I = second_moment # Second moment of area, m^4
        self.L = length  # Length of the beam, m
        # Rayleigh damping coefficients
        self.alpha = r_alpha
        self.beta = r_beta

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

    def jax_modes(self, n_start: int, n_modes: int):
        # Same as modes() but mode shape functions use jnp — differentiable by JAX.
        n_cache = len(self._roots)
        n_stop = n_start + n_modes - 1
        if n_stop > n_cache:
            for n in range(n_stop - n_cache):
                self._compute_nth_root(n_cache + n + 1)

        ws = self.omegas[n_start - 1:n_stop]
        phis = []
        for beta in self.betas[n_start - 1:n_stop]:
            b = float(beta)
            L = float(self.L)
            C = float((np.cos(b * L) + np.cosh(b * L)) / (np.sin(b * L) + np.sinh(b * L)))
            phi_i = lambda x, _b=b, _C=C: (
                (jnp.cosh(_b * x) - jnp.cos(_b * x))
                - _C * (jnp.sinh(_b * x) - jnp.sin(_b * x))
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
        # Moreover, int_{O}^{L} phi_i^2(x) dx = L, therefore, m_ii = mass
        # It can be shown that: k_ii = w_i * m_ii
        # The damping term C is computed using Rayleigh damping: C = alpha*M + beta*K

        m_mat = self.mu * self.L * np.eye(n_modes)
        k_mat = np.diag(m_mat @ (self.omegas[:n_modes]**2))
        c_mat = self.alpha * m_mat + self.beta * k_mat

        return m_mat, c_mat, k_mat
