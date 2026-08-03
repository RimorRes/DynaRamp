import numpy as np

from dynaramp.common.vecmath import (
    skew_sym_mat,
    euler_zyx,
    small_rot,
    h_rotation_matrix,
    h_matrix,
    h_dot_matrix,
)


def test_euler_identity():
    assert np.allclose(euler_zyx(0.0, 0.0, 0.0), np.identity(3))


def test_euler_orthonormal():
    rng = np.random.default_rng(0)
    for _ in range(20):
        gamma, psi, phi = rng.uniform(-np.pi, np.pi, size=3)
        a = euler_zyx(gamma, psi, phi)
        assert np.allclose(a @ a.T, np.identity(3), atol=1e-12)
        assert np.isclose(np.linalg.det(a), 1.0, atol=1e-12)


def test_small_rot_matches_euler_first_order():
    # For small (gamma, psi, phi): A_LB ~= I + skew([phi, psi, gamma])
    eps = 1e-4
    gamma, psi, phi = 0.7 * eps, -1.3 * eps, 0.4 * eps
    a = euler_zyx(gamma, psi, phi)
    approx = small_rot([phi, psi, gamma])
    assert np.allclose(a, approx, atol=1e-7)


def test_angular_velocity_map_consistency():
    # skew(H_R @ a_dot) @ A_LB must equal d/dt A_LB (finite difference).
    rng = np.random.default_rng(1)
    dt = 1e-6
    for _ in range(20):
        a0 = rng.uniform(-1.0, 1.0, size=3)          # [gamma, psi, phi]
        a_dot = rng.uniform(-2.0, 2.0, size=3)        # [gamma_dot, psi_dot, phi_dot]

        gamma, psi, phi = a0
        omega_l = h_rotation_matrix(gamma, psi) @ a_dot
        analytic = skew_sym_mat(omega_l) @ euler_zyx(gamma, psi, phi)

        a_plus = a0 + a_dot * dt
        a_minus = a0 - a_dot * dt
        numeric = (euler_zyx(*a_plus) - euler_zyx(*a_minus)) / (2 * dt)

        assert np.allclose(analytic, numeric, atol=1e-6)


def test_h_dot_matches_finite_difference():
    rng = np.random.default_rng(2)
    dt = 1e-6
    for _ in range(20):
        gamma, psi = rng.uniform(-1.0, 1.0, size=2)
        gamma_dot, psi_dot = rng.uniform(-2.0, 2.0, size=2)

        analytic = h_dot_matrix(gamma, psi, gamma_dot, psi_dot)
        h_plus = h_matrix(gamma + gamma_dot * dt, psi + psi_dot * dt)
        h_minus = h_matrix(gamma - gamma_dot * dt, psi - psi_dot * dt)
        numeric = (h_plus - h_minus) / (2 * dt)

        assert np.allclose(analytic, numeric, atol=1e-6)


def test_h_matrix_block_structure():
    h = h_matrix(0.3, -0.4)
    assert h.shape == (6, 6)
    assert np.allclose(h[0:3, 0:3], np.identity(3))
    assert np.allclose(h[0:3, 3:6], 0.0)
    assert np.allclose(h[3:6, 0:3], 0.0)
    assert np.allclose(h[3:6, 3:6], h_rotation_matrix(0.3, -0.4))
