import numpy as np
import pytest

import dynaramp.mstmm as dyn
from dynaramp.projectile import GuideModalField, ProjectileEOM
from dynaramp.contact import GuideReaction
from dynaramp.simulation import modal_contact_force, modal_damping_stiffness, assemble_and_solve


def _build():
    topo = dyn.TopologyHandler()
    length = 10
    h, b = 1, 2
    beam = dyn.EulerBernoulliBeam(
        e_id="beam", length=length, density=7800, youngs_mod=210e9,
        shear_mod=80e9, area=b * h, i_y=b * (h ** 3) / 12, i_z=(b ** 3) * h / 12,
    )
    topo.add_elements(beam)
    tip = np.array([0, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    root = np.array([None, None, None, None, None, None, 0, 0, 0, 0, 0, 0])
    topo.add_root(beam, root, output_pos=(length, 0, 0))
    topo.add_tip(beam, tip, input_pos=(0, 0, 0))
    topo.make_tree()
    return dyn.System(topo)


@pytest.fixture(scope="module")
def setup():
    system = _build()
    modes = system.natural_modes(3, omega_max=300, search_res=3000)
    field = GuideModalField.from_elements(system, ["beam"], modes)
    modal_masses = system.calc_system_modal_masses(modes)
    return field, modal_masses


def test_modal_force_single_reaction_matches_projection(setup):
    field, mm = setup
    x = 3.7
    q = np.array([10.0, -5.0, 3.0])
    m = np.array([0.5, 0.2, -0.1])
    f_g = modal_contact_force(field, mm, [GuideReaction(x, q, m)])

    phi_r, phi_theta = field.phi(x)
    expected = (phi_r.T @ q + phi_theta.T @ m) / mm  # <f, V^s> / M_s
    assert f_g.shape == (field.n_modes,)
    assert np.allclose(f_g, expected)


def test_modal_force_zero_reactions(setup):
    field, mm = setup
    assert np.allclose(modal_contact_force(field, mm, []), 0.0)


def test_modal_force_is_linear_in_reactions(setup):
    field, mm = setup
    r1 = GuideReaction(3.0, np.array([1.0, 0.0, 0.0]), np.zeros(3))
    r2 = GuideReaction(5.0, np.array([0.0, 2.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    f_both = modal_contact_force(field, mm, [r1, r2])
    f1 = modal_contact_force(field, mm, [r1])
    f2 = modal_contact_force(field, mm, [r2])
    assert np.allclose(f_both, f1 + f2)


# --------------------------------------------------------------------------- #
# Coupled assembler (Eq. 67)
# --------------------------------------------------------------------------- #
def test_modal_damping_stiffness_diagonals():
    freqs = np.array([10.0, 25.0, 40.0])
    c_g, k_g = modal_damping_stiffness(freqs, rayleigh=(0.5, 1e-4))
    assert np.allclose(k_g, freqs ** 2)
    assert np.allclose(c_g, 0.5 + 1e-4 * freqs ** 2)


def _synthetic_eom(n, seed=0):
    rng = np.random.default_rng(seed)
    m_p = rng.standard_normal((6, n))                 # [M_Tp; M_Rp]
    a_y = rng.standard_normal((6, 6))
    m_y = a_y @ a_y.T + 6.0 * np.identity(6)          # SPD -> invertible projectile mass block
    return ProjectileEOM(
        m_tp=m_p[:3], m_ty=m_y[:3], m_rp=m_p[3:], m_ry=m_y[3:],
        h_t=rng.standard_normal(3), h_r=rng.standard_normal(3),
    )


def test_assemble_and_solve_satisfies_equation_67():
    n = 3
    rng = np.random.default_rng(1)
    eom = _synthetic_eom(n, seed=2)
    c_g, k_g = rng.uniform(0, 1, n), rng.uniform(1, 10, n)
    p, p_dot, f_g = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
    cq, cm, eq, em = (rng.standard_normal(3) for _ in range(4))

    p_ddot, y_dot = assemble_and_solve(eom, c_g, k_g, p, p_dot, f_g, cq, cm, eq, em)
    assert p_ddot.shape == (n,) and y_dot.shape == (6,)

    # Top block row [I | 0] gives the vehicle acceleration directly: p_ddot = b_g.
    assert np.allclose(p_ddot, f_g - c_g * p_dot - k_g * p)
    # Projectile block rows are satisfied.
    assert np.allclose(eom.m_tp @ p_ddot + eom.m_ty @ y_dot, eom.h_t + cq + eq)
    assert np.allclose(eom.m_rp @ p_ddot + eom.m_ry @ y_dot, eom.h_r + cm + em)
