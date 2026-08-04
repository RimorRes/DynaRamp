import numpy as np
import pytest

from dynaramp.projectile import Projectile, ProjectileEOM, ProjectileKinematicState
from dynaramp.common.vecmath import euler_zyx


def _synthetic_kin(n=2, seed=0):
    rng = np.random.default_rng(seed)
    a_il = euler_zyx(*rng.uniform(-0.5, 0.5, size=3))
    a_ib = a_il @ euler_zyx(*rng.uniform(-0.5, 0.5, size=3))
    return ProjectileKinematicState(
        a_il=a_il,
        a_ib=a_ib,
        omega_il=rng.standard_normal(3),
        omega_ib=rng.standard_normal(3),
        r_o1=rng.standard_normal(3),
        r_dot_o1=rng.standard_normal(3),
        l_to1=rng.standard_normal((3, n)),
        j_to1=rng.standard_normal((3, 6)),
        zeta_to1=rng.standard_normal(3),
        l_rp=rng.standard_normal((3, n)),
        j_ro1=rng.standard_normal((3, 6)),
        zeta_ro1=rng.standard_normal(3),
    )


def _spd_inertia(seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((3, 3))
    return a @ a.T + 3 * np.identity(3)


def test_block_shapes():
    n = 3
    kin = _synthetic_kin(n=n, seed=1)
    projectile = Projectile(mass=120.0, inertia_com=_spd_inertia(2), com_o1=(0.5, 0.1, -0.2))
    eom = ProjectileEOM.assemble(kin, projectile)
    assert eom.m_tp.shape == (3, n)
    assert eom.m_rp.shape == (3, n)
    assert eom.m_ty.shape == (3, 6)
    assert eom.m_ry.shape == (3, 6)
    assert eom.h_t.shape == (3,)
    assert eom.h_r.shape == (3,)


def test_blocks_match_equations_cross_product():
    # Independent reconstruction of Eqs. 45-48 using np.cross for the vector terms.
    n = 2
    kin = _synthetic_kin(n=n, seed=3)
    projectile = Projectile(mass=87.0, inertia_com=_spd_inertia(4), com_o1=(0.4, -0.3, 0.2))
    eom = ProjectileEOM.assemble(kin, projectile)

    m = projectile.mass
    r = kin.a_ib @ projectile.com_o1
    i_o1 = kin.a_ib @ projectile.inertia_o1 @ kin.a_ib.T
    w = kin.omega_ib

    def cross_cols(a, mat):
        return np.column_stack([np.cross(a, mat[:, k]) for k in range(mat.shape[1])])

    m_tp = m * (kin.l_to1 - cross_cols(r, kin.l_rp))
    m_ty = m * (kin.j_to1 - cross_cols(r, kin.j_ro1))
    h_t = m * (np.cross(r, kin.zeta_ro1) - np.cross(w, np.cross(w, r)) - kin.zeta_to1)
    m_rp = i_o1 @ kin.l_rp + m * cross_cols(r, kin.l_to1)
    m_ry = i_o1 @ kin.j_ro1 + m * cross_cols(r, kin.j_to1)
    h_r = -i_o1 @ kin.zeta_ro1 - m * np.cross(r, kin.zeta_to1) - np.cross(w, i_o1 @ w)

    assert np.allclose(eom.m_tp, m_tp)
    assert np.allclose(eom.m_ty, m_ty)
    assert np.allclose(eom.h_t, h_t)
    assert np.allclose(eom.m_rp, m_rp)
    assert np.allclose(eom.m_ry, m_ry)
    assert np.allclose(eom.h_r, h_r)


def test_translational_eom_reduces_to_newton_when_com_at_o1():
    # com_o1 = 0 => LHS - h_T == m * (L_TO1 p_ddot + J_TO1 y_dot + zeta_TO1) = m * a_O1.
    n = 2
    kin = _synthetic_kin(n=n, seed=5)
    projectile = Projectile(mass=42.0, inertia_com=_spd_inertia(6), com_o1=(0, 0, 0))
    eom = ProjectileEOM.assemble(kin, projectile)

    rng = np.random.default_rng(7)
    p_ddot = rng.standard_normal(n)
    y_dot = rng.standard_normal(6)

    lhs = eom.m_tp @ p_ddot + eom.m_ty @ y_dot - eom.h_t
    a_o1 = kin.l_to1 @ p_ddot + kin.j_to1 @ y_dot + kin.zeta_to1
    assert np.allclose(lhs, projectile.mass * a_o1)


def test_rotational_eom_reduces_to_euler_when_com_at_o1():
    # com_o1 = 0 => LHS - h_R == I_O1 omega_dot_IB + omega_IB x (I_O1 omega_IB).
    n = 2
    kin = _synthetic_kin(n=n, seed=8)
    inertia = _spd_inertia(9)
    projectile = Projectile(mass=42.0, inertia_com=inertia, com_o1=(0, 0, 0))
    eom = ProjectileEOM.assemble(kin, projectile)

    rng = np.random.default_rng(10)
    p_ddot = rng.standard_normal(n)
    y_dot = rng.standard_normal(6)

    i_o1 = kin.a_ib @ inertia @ kin.a_ib.T
    omega_dot_ib = kin.l_rp @ p_ddot + kin.j_ro1 @ y_dot + kin.zeta_ro1

    lhs = eom.m_rp @ p_ddot + eom.m_ry @ y_dot - eom.h_r
    euler = i_o1 @ omega_dot_ib + np.cross(kin.omega_ib, i_o1 @ kin.omega_ib)
    assert np.allclose(lhs, euler)
