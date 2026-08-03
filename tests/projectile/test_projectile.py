import numpy as np
import pytest

from dynaramp.projectile import Projectile, Slider, ProjectileState
from dynaramp.common.vecmath import h_matrix, h_rotation_matrix


# --------------------------------------------------------------------------- #
# Projectile
# --------------------------------------------------------------------------- #
def test_projectile_coercion_and_defaults():
    m = Projectile(mass=100, inertia_com=np.diag([1.0, 2.0, 3.0]))
    assert isinstance(m.mass, float)
    assert m.inertia_com.shape == (3, 3)
    assert m.com_o1.shape == (3,)
    assert np.allclose(m.com_o1, 0.0)
    assert m.sliders == []


def test_inertia_o1_equals_com_when_no_offset():
    inertia = np.diag([1.0, 2.0, 3.0])
    m = Projectile(mass=5.0, inertia_com=inertia, com_o1=(0, 0, 0))
    assert np.allclose(m.inertia_o1, inertia)


def test_inertia_o1_parallel_axis():
    # I_C = I3, m = 2, r = O1->C = [1,0,0]
    # I_O1 = I3 + 2*(|r|^2 I3 - r r^T) = I3 + 2*diag(0,1,1) = diag(1,3,3)
    m = Projectile(mass=2.0, inertia_com=np.identity(3), com_o1=(1.0, 0.0, 0.0))
    assert np.allclose(m.inertia_o1, np.diag([1.0, 3.0, 3.0]))
    # inertia about O1 must stay symmetric
    assert np.allclose(m.inertia_o1, m.inertia_o1.T)


def test_inertia_o1_matches_manual_formula():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 3))
    inertia = a @ a.T  # symmetric positive semidefinite
    r = np.array([0.3, -1.2, 0.7])
    mass = 7.5
    m = Projectile(mass=mass, inertia_com=inertia, com_o1=r)
    expected = inertia + mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))
    assert np.allclose(m.inertia_o1, expected)


def test_slider_coercion():
    s = Slider(position=(0.1, 0.2, 0.3), radius=2)
    assert s.position.shape == (3,)
    assert isinstance(s.radius, float)
    m = Projectile(mass=1.0, inertia_com=np.identity(3), sliders=[s])
    assert len(m.sliders) == 1


# --------------------------------------------------------------------------- #
# ProjectileState
# --------------------------------------------------------------------------- #
def test_state_accessors():
    x = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    y = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    st = ProjectileState(x, y)
    assert np.allclose(st.s_b, x[0:3])
    assert np.allclose(st.a_b, x[3:6])
    assert (st.x_r, st.y_l, st.z_l) == (1.0, 2.0, 3.0)
    assert (st.gamma, st.psi, st.phi) == (0.1, 0.2, 0.3)
    assert np.allclose(st.s_b_dot, y[0:3])
    assert np.allclose(st.omega_lb, y[3:6])
    assert st.v_p_prime == 4.0


def test_state_from_config_rates_matches_H():
    x = np.array([1.0, 0.5, -0.5, 0.2, -0.3, 0.4])
    x_dot = np.array([2.0, -1.0, 0.5, 0.7, 0.1, -0.2])
    st = ProjectileState.from_config_rates(x, x_dot)
    # translational part is identity
    assert np.allclose(st.s_b_dot, x_dot[0:3])
    # rotational part is H_R @ euler_rates
    assert np.allclose(st.omega_lb, h_rotation_matrix(x[3], x[4]) @ x_dot[3:6])
    assert np.allclose(st.y, h_matrix(x[3], x[4]) @ x_dot)


def test_state_config_rates_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.uniform(-1.0, 1.0, size=6)
        # keep psi away from gimbal lock
        x[4] = rng.uniform(-1.0, 1.0)
        x_dot = rng.uniform(-2.0, 2.0, size=6)
        st = ProjectileState.from_config_rates(x, x_dot)
        assert np.allclose(st.config_rates(), x_dot, atol=1e-12)
