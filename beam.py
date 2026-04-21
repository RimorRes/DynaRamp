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

    @staticmethod
    def characteristic_equation(beta, l):
        return np.cosh(beta*l) * np.cos(beta*l) + 1

    def _betas(self, n_modes):
        # Compute the n-th first roots of the characteristic equation
        betas = []
        for n in range(1, n_modes + 1):
            x0 = (2*n - 1) * np.pi/2  # Initial guess for a cantilevered beam
            b0 = x0 / self.L
            b_i = float(fsolve(self.characteristic_equation, b0, args=(self.L,))[0])

            betas.append(b_i)
        return np.array(betas, dtype=float)

    def modes(self, n_modes):
        betas = self._betas(n_modes)

        # Natural frequencies (rad/s)
        ws = betas ** 2 * np.sqrt(self.E * self.I / self.mu)

        # Mode shapes (normalized)
        phis = []
        for beta in betas:
            phi_i = lambda x, b_i=beta :(
                    1 * ((np.cosh(b_i * x) - np.cos(b_i * x))
                         - (np.cos(b_i*self.L) + np.cosh(b_i*self.L)) / (np.sin(b_i*self.L) + np.sinh(b_i*self.L))
                         * (np.sinh(b_i * x) - np.sin(b_i * x)))
            )

            phis.append(phi_i)

        return ws, phis

    def modal_matrices(self, n_modes):
        ws, phis = self.modes(n_modes)
        m_mat = np.zeros((n_modes, n_modes))
        k_mat = np.zeros((n_modes, n_modes))

        for i in range(n_modes):
            for j in range(n_modes):
                f_m = lambda x: phis[i](x) * phis[j](x)
                f_k = lambda x: phis[i](x) * phis[j](x)

                m_mat[i, j] = integrate.quad(f_m, 0, self.L)[0]
                k_mat[i, j] = integrate.quad(f_k, 0, self.L)[0]

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
    ws, phis = Beam.modes(4)
    M, K = Beam.modal_matrices(4)

    print("Natural frequencies (rad/s):", ws)
    xs = np.linspace(0.0, L, 100)

    print("M", M)
    print("K", K)

    for i, phi in enumerate(phis, start=1):
        ys = phi(xs)
        plt.plot(xs, ys, label=f"Mode {i}")
        f = lambda x: phi(x)**2
        print(integrate.quad(f, 0, L))

    plt.xlabel("x (m)")
    plt.ylabel("Mode shape (normalized)")
    plt.legend()
    plt.show()
