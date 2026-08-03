import numpy as np
import pytest

import dynaramp.mstmm as dyn
from dynaramp.projectile import GuideModalField, ProjectileKinematics, ProjectileState
from dynaramp.common.vecmath import small_rot, h_matrix, h_dot_matrix


def _build_cantilever():
    topo = dyn.TopologyHandler()
    length = 10
    h, b = 1, 2
    i_y = b * (h ** 3) / 12
    i_z = (b ** 3) * h / 12
    beam = dyn.EulerBernoulliBeam(
        e_id="beam", length=length, density=7800, youngs_mod=210e9,
        shear_mod=80e9, area=b * h, i_y=i_y, i_z=i_z,
    )
    topo.add_elements(beam)
    tip = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    root = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])
    topo.add_root(beam, root, output_pos=(length, 0, 0))
    topo.add_tip(beam, tip, input_pos=(0, 0, 0))
    topo.make_tree()
    return dyn.System(topo), "beam"


@pytest.fixture(scope="module")
def field():
    system, elem = _build_cantilever()
    modes = system.natural_modes(3, omega_max=300, search_res=3000)
    assert len(modes) >= 1
    return GuideModalField.from_elements(system, [elem], modes)


X0 = 3.7


def _r_o1(field, x, p):
    """Independent first-principles position of O1 in K_I (model oracle for FD checks)."""
    a_ir = field.a_ir
    x_r, y_l, z_l = x[0], x[1], x[2]
    ms = field.evaluate(x_r)
    theta_rp = a_ir.T @ (ms.phi_theta @ p)
    a_il = a_ir @ small_rot(theta_rp)
    r_pp = a_ir @ np.array([x_r, 0.0, 0.0]) + ms.phi_r @ p
    return r_pp + a_il @ np.array([0.0, y_l, z_l])


def test_block_shapes(field):
    n = field.n_modes
    st = ProjectileState(np.array([X0, 0.05, -0.03, 0.1, -0.2, 0.15]), np.zeros(6))
    kin = ProjectileKinematics(field).evaluate(st, np.zeros(n), np.zeros(n))
    assert kin.l_to1.shape == (3, n)
    assert kin.l_rp.shape == (3, n)
    assert kin.j_to1.shape == (3, 6)
    assert kin.j_ro1.shape == (3, 6)
    assert kin.zeta_to1.shape == (3,)
    assert kin.zeta_ro1.shape == (3,)
    assert kin.a_ib.shape == (3, 3)


def test_rigid_canister_limit(field):
    # p = p_dot = 0: no deformation. Convective terms vanish, A_IL = A_IR = I.
    n = field.n_modes
    omega_lb = np.array([0.3, -0.1, 0.2])
    y = np.concatenate([[1.5, 0.0, 0.0], omega_lb])  # sliding + body rotation
    st = ProjectileState(np.array([X0, 0.04, -0.02, 0.2, -0.1, 0.05]), y)
    kin = ProjectileKinematics(field).evaluate(st, np.zeros(n), np.zeros(n))

    assert np.allclose(kin.a_il, np.identity(3), atol=1e-12)
    assert np.allclose(kin.omega_il, 0.0, atol=1e-12)
    assert np.allclose(kin.zeta_to1, 0.0, atol=1e-12)
    assert np.allclose(kin.zeta_ro1, 0.0, atol=1e-12)
    # With A_IL = I, I_omega_IB reduces to the body rotation A_LB @ omega_LB via J_RO1.
    assert np.allclose(kin.omega_ib, kin.a_il @ omega_lb, atol=1e-12)


def test_omega_identity(field):
    # I_omega_IB == I_omega_IL + A_IL @ L_omega_LB (Eq. A10), exact by construction.
    n = field.n_modes
    rng = np.random.default_rng(3)
    p = 1e-2 * rng.standard_normal(n)
    p_dot = 1e-2 * rng.standard_normal(n)
    y = np.array([1.2, 0.3, -0.2, 0.4, -0.5, 0.6])
    st = ProjectileState(np.array([X0, 0.05, -0.04, 0.1, -0.2, 0.3]), y)
    kin = ProjectileKinematics(field).evaluate(st, p, p_dot)
    assert np.allclose(kin.omega_ib, kin.omega_il + kin.a_il @ y[3:6], atol=1e-12)


def test_translational_velocity_finite_difference(field):
    # I_r_dot_O1 (central FD of the position oracle) == L_TO1 p_dot + J_TO1 y.
    n = field.n_modes
    rng = np.random.default_rng(4)
    p0 = 1e-3 * rng.standard_normal(n)
    p_dot = 1e-3 * rng.standard_normal(n)
    x0 = np.array([X0, 0.03, -0.02, 0.08, -0.05, 0.10])
    x_dot = np.array([1.4, 0.2, -0.15, 0.1, -0.07, 0.05])
    y0 = h_matrix(x0[3], x0[4]) @ x_dot
    st = ProjectileState(x0, y0)

    kin = ProjectileKinematics(field).evaluate(st, p0, p_dot)
    v_ana = kin.l_to1 @ p_dot + kin.j_to1 @ y0

    dt = 1e-6
    v_num = (_r_o1(field, x0 + x_dot * dt, p0 + p_dot * dt)
             - _r_o1(field, x0 - x_dot * dt, p0 - p_dot * dt)) / (2 * dt)
    assert np.allclose(v_ana, v_num, atol=1e-6)


def test_translational_acceleration_finite_difference(field):
    # Linear-in-t trajectory => x_ddot = 0, p_ddot = 0, so y_dot = H_dot x_dot.
    # I_r_ddot_O1 (2nd central FD) == J_TO1 y_dot + zeta_TO1.
    n = field.n_modes
    rng = np.random.default_rng(5)
    p0 = 1e-3 * rng.standard_normal(n)
    p_dot = 1e-3 * rng.standard_normal(n)
    x0 = np.array([X0, 0.03, -0.02, 0.08, -0.05, 0.10])
    x_dot = np.array([1.4, 0.2, -0.15, 0.1, -0.07, 0.05])
    y0 = h_matrix(x0[3], x0[4]) @ x_dot
    y_dot0 = h_dot_matrix(x0[3], x0[4], x_dot[3], x_dot[4]) @ x_dot
    st = ProjectileState(x0, y0)

    kin = ProjectileKinematics(field).evaluate(st, p0, p_dot)
    a_ana = kin.j_to1 @ y_dot0 + kin.zeta_to1

    dt = 1e-4
    a_num = (_r_o1(field, x0 + x_dot * dt, p0 + p_dot * dt)
             - 2 * _r_o1(field, x0, p0)
             + _r_o1(field, x0 - x_dot * dt, p0 - p_dot * dt)) / dt ** 2
    assert np.allclose(a_ana, a_num, atol=1e-4)
